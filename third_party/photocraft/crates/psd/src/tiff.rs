//! Photoshop layer data embedded in TIFF files (layered TIFF).
//!
//! Photoshop keeps a layered TIFF's layers in two private tags, described by Adobe's
//! *Photoshop TIFF Technical Notes* (March 2002) together with the PSD specification:
//!
//! * **37724 `ImageSourceData`**: the string `"Adobe Photoshop Document Data Block\0"`
//!   (`"Adobe Photoshop Document Data V0002\0"` for PSB-sized data, whose long keys carry
//!   8-byte lengths) followed by tagged blocks in the usual `8BIM` + key + length + data form,
//!   each padded to a multiple of 4. `Layr` (8-bit), `Lr16` and `Lr32` hold the layer info, the
//!   same structure as a PSD's layer info section; `LMsk` holds the global layer mask; every other
//!   block is a global additional-layer-information block (`Patt`, `Anno`, `FMsk`, `Txt2`, …).
//! * **34377**: the PSD image resources (`8BIM` resource blocks, without the section length).
//!
//! **Byte order.** Adobe's note is silent, but Photoshop files show (and open-source readers such as psdtags
//! implement) that the block structure follows the TIFF's byte order: in an Intel-order (`II`)
//! file every integer and double is little-endian and every four-character signature and key is
//! stored reversed (`MIB8`, `ryaL`), while the pixel samples inside channel data stay big-endian
//! as in a PSD, and RLE row-count tables follow the container. [`ImageSourceData::from_bytes`]
//! normalizes either order to the big-endian PSD model, and [`ImageSourceData::to_bytes`] writes
//! either order, by transcoding every structure it knows field by field ([`transcode`]).
//! A block the transcoder does not know is never dropped on the way in: its bytes are kept
//! verbatim with the signature `MIB8` (`46B8` for `8B64`) marking that they are in little-endian
//! layout ([`TaggedBlock::is_foreign_order`]), so a later save to a little-endian TIFF writes it
//! back exactly. Only a save to the *other* byte order has to drop such a block, with a warning:
//! a byte-swapped block passed through unchanged would be read as garbage by every other
//! application (and would end the block list for readers that stop at a bad signature).

use crate::descriptor::MAX_DEPTH;
use crate::error::{PsdError, Result};
use crate::file::{GlobalLayerMask, LayerInfoPlacement, PsdFile};
use crate::header::{Header, Version};
use crate::image_data::ImageData;
use crate::io::{Reader, WriteExt};
use crate::layer::{LayerInfo, LayerMask, mask_parameter_bytes};
use crate::resources::ImageResource;
use crate::tagged::{TaggedBlock, read_blocks, uses_long_length, write_blocks};

/// The byte order of a TIFF container (and so of the Photoshop data inside it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ByteOrder {
    /// Motorola order (`MM`); the PSD byte order.
    #[default]
    Big,
    /// Intel order (`II`).
    Little,
}

impl ByteOrder {
    fn is_little(self) -> bool {
        matches!(self, ByteOrder::Little)
    }
}

/// Header string of 37724 data with PSD-sized (4-byte) lengths.
pub const SIGNATURE: &[u8; 36] = b"Adobe Photoshop Document Data Block\0";
/// Header string of 37724 data with PSB-sized (8-byte) lengths for the long keys.
pub const SIGNATURE_PSB: &[u8; 36] = b"Adobe Photoshop Document Data V0002\0";

/// The layer-info block key Photoshop uses for a bit depth.
pub fn layer_info_key(depth: u16) -> [u8; 4] {
    match depth {
        16 => *b"Lr16",
        32 => *b"Lr32",
        _ => *b"Layr",
    }
}

/// The contents of an `ImageSourceData` tag, in the big-endian PSD model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageSourceData {
    /// PSD (4-byte lengths) or PSB (8-byte lengths for the long keys).
    pub version: Version,
    /// The byte order the data was read in (what [`Self::to_bytes`] should normally write back).
    pub byte_order: ByteOrder,
    /// The layers, from the `Layr` / `Lr16` / `Lr32` block.
    pub layer_info: Option<LayerInfo>,
    /// Which key held the layer info (`Layr` when there was none).
    pub layer_info_key: [u8; 4],
    /// The `LMsk` block.
    pub global_layer_mask: Option<GlobalLayerMask>,
    /// Every other block, in stored order.
    pub global_blocks: Vec<TaggedBlock>,
    /// Bytes after the last block that do not form a block.
    pub trailing: Vec<u8>,
}

impl Default for ImageSourceData {
    fn default() -> Self {
        ImageSourceData {
            version: Version::Psd,
            byte_order: ByteOrder::Big,
            layer_info: None,
            layer_info_key: *b"Layr",
            global_layer_mask: None,
            global_blocks: Vec::new(),
            trailing: Vec::new(),
        }
    }
}

fn block_order(rest: &[u8]) -> Result<ByteOrder> {
    match rest.get(..4) {
        None | Some(b"8BIM") | Some(b"8B64") => Ok(ByteOrder::Big),
        Some(b"MIB8") | Some(b"46B8") => Ok(ByteOrder::Little),
        Some(other) => {
            Err(PsdError::InvalidSignature { expected: "8BIM or MIB8 (Photoshop image source data block)", found: [other[0], other[1], other[2], other[3]] })
        }
    }
}

impl ImageSourceData {
    /// Parses a 37724 tag. Returns the data together with warnings about blocks that were
    /// dropped because they could not be converted from the file's byte order.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, Vec<String>)> {
        let (version, rest) = match data.get(..36) {
            Some(s) if s == SIGNATURE => (Version::Psd, &data[36..]),
            Some(s) if s == SIGNATURE_PSB => (Version::Psb, &data[36..]),
            _ => return Err(PsdError::invalid("not Photoshop image source data (missing \"Adobe Photoshop Document Data Block\")")),
        };
        let byte_order = block_order(rest)?;
        let mut warnings = Vec::new();
        let big;
        let rest = if byte_order.is_little() {
            let (b, w) = transcode::transcode_blocks(rest, version, ByteOrder::Little, ByteOrder::Big)?;
            warnings = w;
            big = b;
            &big[..]
        } else {
            rest
        };
        let (mut blocks, trailing) = read_blocks(&mut Reader::new(rest), version)?;
        // Blocks in a TIFF are padded to 4 (`to_bytes` re-pads them), so zero padding is the
        // canonical default here, as even-length padding is in a PSD.
        for b in &mut blocks {
            if b.padding.as_ref().is_some_and(|p| p.len() <= 3 && p.iter().all(|&x| x == 0)) {
                b.padding = None;
            }
        }
        let mut layer_info = None;
        let mut layer_info_key = *b"Layr";
        if let Some(i) = blocks.iter().position(|b| matches!(&b.key, b"Layr" | b"Lr16" | b"Lr32")) {
            let b = blocks.remove(i);
            layer_info = Some(LayerInfo::read_body(&b.data, version)?);
            layer_info_key = b.key;
        }
        let global_layer_mask = blocks.iter().position(|b| &b.key == b"LMsk").map(|i| GlobalLayerMask { data: blocks.remove(i).data });
        Ok((ImageSourceData { version, byte_order, layer_info, layer_info_key, global_layer_mask, global_blocks: blocks, trailing }, warnings))
    }

    /// Serializes the data in `order`. Returns the bytes and warnings about blocks that were
    /// dropped because they could not be converted to a little-endian layout.
    pub fn to_bytes(&self, order: ByteOrder) -> Result<(Vec<u8>, Vec<String>)> {
        let version = self.version;
        let mut blocks = Vec::with_capacity(self.global_blocks.len() + 2);
        if let Some(info) = &self.layer_info {
            // Pad to 4 without cloning the layers (and all their pixel data) just to set `padding`.
            let pad = info.pad_len(version, 4)?;
            let mut data = Vec::new();
            data.try_reserve(info.channel_data_len().saturating_add(pad)).map_err(|_| PsdError::LimitExceeded("not enough memory for the layer data"))?;
            info.write_unpadded(&mut data, version)?;
            data.resize(data.len() + pad, 0);
            blocks.push(padded_block(self.layer_info_key, data));
        }
        if let Some(m) = &self.global_layer_mask {
            blocks.push(padded_block(*b"LMsk", m.data.clone()));
        }
        for b in &self.global_blocks {
            let mut b = b.clone();
            if b.padding.is_none() {
                // Padded to 4 like the others; the signature stays (a foreign-order marker included).
                b.padding = Some(vec![0; (4 - b.data.len() % 4) % 4]);
            }
            blocks.push(b);
        }
        let mut body = Vec::new();
        write_blocks(&mut body, &blocks, version, 2)?;
        body.put(&self.trailing);
        let (body, warnings) =
            if order.is_little() { transcode::transcode_blocks(&body, version, ByteOrder::Big, ByteOrder::Little)? } else { (body, Vec::new()) };
        let mut out = Vec::with_capacity(36 + body.len());
        out.put(if version.is_psb() { SIGNATURE_PSB } else { SIGNATURE });
        out.put(&body);
        Ok((out, warnings))
    }

    /// The layers, global mask and global blocks of a PSD as image source data (byte order Big).
    pub fn from_psd(file: &PsdFile) -> Self {
        let key = match &file.layer_info_placement {
            LayerInfoPlacement::GlobalBlock { key, .. } => *key,
            LayerInfoPlacement::Section => layer_info_key(file.header.depth),
        };
        ImageSourceData {
            version: file.header.version,
            byte_order: ByteOrder::Big,
            layer_info: file.layer_info.clone(),
            layer_info_key: key,
            global_layer_mask: file.global_layer_mask.clone(),
            global_blocks: file.global_blocks.clone(),
            trailing: file.layer_mask_trailing.clone(),
        }
    }

    /// Assembles a PSD model from this data plus the parts a TIFF stores natively: the header
    /// (dimensions, depth, colour mode, channel count), the image resources (tag 34377) and the
    /// merged image (the TIFF's pixels).
    pub fn into_psd(self, header: Header, resources: Vec<ImageResource>, image_data: ImageData) -> PsdFile {
        let placement = if matches!(&self.layer_info_key, b"Lr16" | b"Lr32") && self.layer_info.is_some() {
            LayerInfoPlacement::GlobalBlock { index: 0, signature: *b"8BIM", key: self.layer_info_key, padding: None }
        } else {
            LayerInfoPlacement::Section
        };
        PsdFile {
            header,
            color_mode_data: Vec::new(),
            resources,
            layer_info: self.layer_info,
            layer_info_placement: placement,
            global_layer_mask: self.global_layer_mask,
            global_blocks: self.global_blocks,
            layer_mask_trailing: self.trailing,
            image_data,
        }
    }
}

/// A block padded to a multiple of 4, as Photoshop writes them in TIFFs.
fn padded_block(key: [u8; 4], data: Vec<u8>) -> TaggedBlock {
    let pad = (4 - data.len() % 4) % 4;
    TaggedBlock { signature: *b"8BIM", key, data, padding: Some(vec![0; pad]) }
}

/// Parses a 34377 tag (image resources). Little-endian resource blocks (`MIB8`) are converted;
/// resource payloads other than the typed numeric ones are kept as stored.
pub fn resources_from_bytes(data: &[u8]) -> Result<(Vec<ImageResource>, Vec<String>)> {
    let order = block_order(data)?;
    let mut warnings = Vec::new();
    let big;
    let data = if order.is_little() {
        let (b, w) = transcode::transcode_resources(data, ByteOrder::Little, ByteOrder::Big)?;
        warnings = w;
        big = b;
        &big[..]
    } else {
        data
    };
    let mut r = Reader::new(data);
    let mut out = Vec::new();
    while !r.is_empty() {
        // Photoshop pads the tag to an even length; a lone trailing byte is not a resource.
        if r.remaining() < 12 && r.peek_rest().iter().all(|&b| b == 0) {
            break;
        }
        out.push(ImageResource::read(&mut r)?);
    }
    Ok((out, warnings))
}

/// Serializes image resources for a 34377 tag (big-endian, which every reader expects).
pub fn resources_to_bytes(resources: &[ImageResource]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for r in resources {
        r.write(&mut out)?;
    }
    Ok(out)
}

/// Field-by-field byte-order conversion of Photoshop structures.
///
/// Every function reads one structure in the source order and writes it in the target order,
/// so output lengths equal input lengths and every length field stays valid. Layouts follow the
/// Adobe PSD specification; the walker only needs to know where the integers, doubles and
/// four-character keys are, not what they mean.
pub mod transcode {
    use super::*;

    /// Converts the tagged blocks after the 37724 header string from `from` to `to`.
    pub fn transcode_blocks(data: &[u8], version: Version, from: ByteOrder, to: ByteOrder) -> Result<(Vec<u8>, Vec<String>)> {
        let mut t = Tx::new(data, version, from, to);
        t.blocks()?;
        Ok(t.finish())
    }

    /// Converts a layer-info body (the data of a `Layr` / `Lr16` / `Lr32` block) between orders.
    pub fn transcode_layer_info(data: &[u8], version: Version, from: ByteOrder, to: ByteOrder) -> Result<(Vec<u8>, Vec<String>)> {
        let mut t = Tx::new(data, version, from, to);
        t.layer_info()?;
        Ok(t.finish())
    }

    /// Converts image-resource blocks (a 34377 tag) between orders.
    pub fn transcode_resources(data: &[u8], from: ByteOrder, to: ByteOrder) -> Result<(Vec<u8>, Vec<String>)> {
        let mut t = Tx::new(data, Version::Psd, from, to);
        t.resources()?;
        Ok(t.finish())
    }

    /// A block signature as stored in `little` or big-endian data. Big-endian data (the model)
    /// also carries the `MIB8` / `46B8` markers of blocks kept from a little-endian file.
    fn is_sig(b: &[u8], little: bool) -> bool {
        if little {
            b.starts_with(b"MIB8") || b.starts_with(b"46B8")
        } else {
            b.starts_with(b"8BIM") || b.starts_with(b"8B64") || TaggedBlock::is_foreign_order_signature(b)
        }
    }

    fn height(top: i32, bottom: i32) -> usize {
        usize::try_from(bottom.saturating_sub(top)).unwrap_or(0)
    }

    /// Per layer record: the channel (id, length) list and the heights of the layer, mask and
    /// real-mask rects.
    type RecordShape = (Vec<(i16, u64)>, [usize; 3]);

    struct Tx<'a> {
        r: Reader<'a>,
        out: Vec<u8>,
        from: ByteOrder,
        to: ByteOrder,
        version: Version,
        warnings: Vec<String>,
        depth: usize,
    }

    macro_rules! copy_int {
        ($name:ident, $ty:ty) => {
            fn $name(&mut self) -> Result<$ty> {
                let v = self.r.$name()?;
                self.out.extend_from_slice(&if self.to.is_little() { v.to_le_bytes() } else { v.to_be_bytes() });
                Ok(v)
            }
        };
    }

    impl<'a> Tx<'a> {
        fn new(data: &'a [u8], version: Version, from: ByteOrder, to: ByteOrder) -> Self {
            let r = if from.is_little() { Reader::new_little(data) } else { Reader::new(data) };
            Tx { r, out: Vec::with_capacity(data.len()), from, to, version, warnings: Vec::new(), depth: 0 }
        }

        fn finish(self) -> (Vec<u8>, Vec<String>) {
            (self.out, self.warnings)
        }

        fn same_order(&self) -> bool {
            self.from == self.to
        }

        fn remaining(&self) -> usize {
            self.r.remaining()
        }

        copy_int!(u16, u16);
        copy_int!(i16, i16);
        copy_int!(u32, u32);
        copy_int!(i32, i32);
        copy_int!(u64, u64);
        copy_int!(i64, i64);
        copy_int!(f64, f64);

        fn u8(&mut self) -> Result<u8> {
            let v = self.r.u8()?;
            self.out.push(v);
            Ok(v)
        }

        fn f32(&mut self) -> Result<()> {
            let b = self.r.array::<4>()?;
            let v = if self.from.is_little() { f32::from_le_bytes(b) } else { f32::from_be_bytes(b) };
            self.out.extend_from_slice(&if self.to.is_little() { v.to_le_bytes() } else { v.to_be_bytes() });
            Ok(())
        }

        /// Bytes copied as they are.
        fn raw(&mut self, n: usize) -> Result<()> {
            let b = self.r.bytes(n)?;
            self.out.extend_from_slice(b);
            Ok(())
        }

        fn rest(&mut self) -> Result<()> {
            let n = self.remaining();
            self.raw(n)
        }

        /// A four-character signature or key: stored reversed in little-endian data. Returns the
        /// canonical (big-endian) spelling.
        fn key(&mut self) -> Result<[u8; 4]> {
            let mut k = self.r.array::<4>()?;
            if self.from.is_little() {
                k.reverse();
            }
            let mut w = k;
            if self.to.is_little() {
                w.reverse();
            }
            self.out.extend_from_slice(&w);
            Ok(k)
        }

        fn len(&mut self, long: bool) -> Result<u64> {
            if long { self.u64() } else { Ok(u64::from(self.u32()?)) }
        }

        fn u16s(&mut self, n: usize) -> Result<()> {
            for _ in 0..n {
                self.u16()?;
            }
            Ok(())
        }

        /// Up to `n` `u16`s: a block shorter than its full layout keeps what it has.
        fn u16s_upto(&mut self, n: usize) -> Result<()> {
            for _ in 0..n {
                if self.remaining() < 2 {
                    break;
                }
                self.u16()?;
            }
            Ok(())
        }

        fn i32s(&mut self, n: usize) -> Result<()> {
            for _ in 0..n {
                self.i32()?;
            }
            Ok(())
        }

        fn f64s(&mut self, n: usize) -> Result<()> {
            for _ in 0..n {
                self.f64()?;
            }
            Ok(())
        }

        /// `u16`s to the end (a lone odd byte is copied).
        fn u16s_to_end(&mut self) -> Result<()> {
            while self.remaining() >= 2 {
                self.u16()?;
            }
            self.rest()
        }

        fn u32s_to_end(&mut self) -> Result<()> {
            while self.remaining() >= 4 {
                self.u32()?;
            }
            self.rest()
        }

        /// A Photoshop Unicode string: u32 count + UTF-16 units.
        fn unicode(&mut self) -> Result<()> {
            let n = self.u32()?;
            self.r.check_count(u64::from(n), 2)?;
            self.u16s(n as usize)
        }

        /// A Pascal string padded to `align`.
        fn pascal(&mut self, align: usize) -> Result<()> {
            let n = usize::from(self.u8()?);
            self.raw(n)?;
            let total = 1 + n;
            self.raw((align - total % align) % align)
        }

        /// Transcodes the next `len` bytes with `f`; whatever `f` leaves is copied verbatim.
        fn region(&mut self, len: u64, f: impl FnOnce(&mut Tx<'_>) -> Result<()>) -> Result<()> {
            let data = self.r.bytes_u64(len)?;
            let mut child = Tx::new(data, self.version, self.from, self.to);
            child.depth = self.depth;
            f(&mut child)?;
            child.rest()?;
            if child.out.len() != data.len() {
                return Err(PsdError::invalid("byte-order conversion changed a structure's length"));
            }
            self.warnings.append(&mut child.warnings);
            self.out.extend_from_slice(&child.out);
            Ok(())
        }

        /// Reads a length field (4 bytes, or 8 when `long`), transcodes that many bytes with `f`
        /// and writes the length of the result: blocks dropped inside shrink the field.
        fn sized_region(&mut self, long: bool, f: impl FnOnce(&mut Tx<'_>) -> Result<()>) -> Result<()> {
            self.sized_region_counting(long, false, f)
        }

        /// [`Self::sized_region`] for a u32 length that counts its own 4 bytes.
        fn sized_region_inclusive(&mut self, f: impl FnOnce(&mut Tx<'_>) -> Result<()>) -> Result<()> {
            self.sized_region_counting(false, true, f)
        }

        fn sized_region_counting(&mut self, long: bool, inclusive: bool, f: impl FnOnce(&mut Tx<'_>) -> Result<()>) -> Result<()> {
            let mut len = self.r.len_field(long)?;
            let at = self.out.len();
            self.out.extend_from_slice(if long { &[0u8; 8] } else { &[0u8; 4] });
            if inclusive {
                len = len.checked_sub(4).ok_or_else(|| PsdError::invalid("length shorter than its own field"))?;
            }
            let data = self.r.bytes_u64(len)?;
            let mut child = Tx::new(data, self.version, self.from, self.to);
            child.depth = self.depth;
            f(&mut child)?;
            child.rest()?;
            self.warnings.append(&mut child.warnings);
            self.patch_len(at, long, child.out.len() as u64 + if inclusive { 4 } else { 0 })?;
            self.out.extend_from_slice(&child.out);
            Ok(())
        }

        /// Tries `f` on the rest of the data; it counts as a match only when at most three
        /// trailing zero bytes are left. On a miss nothing is consumed or written.
        fn attempt(&mut self, f: impl FnOnce(&mut Tx<'_>) -> Result<()>) -> bool {
            let (reader, written) = (self.r.clone(), self.out.len());
            let ok = f(self).is_ok() && self.remaining() <= 3 && self.r.peek_rest().iter().all(|&b| b == 0);
            if !ok {
                self.r = reader;
                self.out.truncate(written);
            }
            ok
        }

        /// Blocks whose layout is not known by key are very often a versioned descriptor,
        /// optionally behind a u32 version (every Photoshop block introduced since CS6 is):
        /// recognized by parsing to the end.
        fn descriptor_shaped(&mut self) -> bool {
            self.attempt(|t| t.versioned_descriptor())
                || self.attempt(|t| {
                    t.u32()?;
                    t.versioned_descriptor()
                })
        }

        /// Writes `n` into the length placeholder at `at`.
        fn patch_len(&mut self, at: usize, long: bool, n: u64) -> Result<()> {
            if long {
                let b = if self.to.is_little() { n.to_le_bytes() } else { n.to_be_bytes() };
                if let Some(slot) = self.out.get_mut(at..at + 8) {
                    slot.copy_from_slice(&b);
                }
            } else {
                let n = u32::try_from(n).map_err(|_| PsdError::LimitExceeded("structure too large for a 32-bit length"))?;
                let b = if self.to.is_little() { n.to_le_bytes() } else { n.to_be_bytes() };
                if let Some(slot) = self.out.get_mut(at..at + 4) {
                    slot.copy_from_slice(&b);
                }
            }
            Ok(())
        }

        /// An RLE row-count table (`rows` entries of 2 bytes, 4 in PSB data).
        fn rle_counts(&mut self, rows: usize) -> Result<()> {
            for _ in 0..rows {
                if self.version.is_psb() {
                    if self.remaining() < 4 {
                        break;
                    }
                    self.u32()?;
                } else {
                    if self.remaining() < 2 {
                        break;
                    }
                    self.u16()?;
                }
            }
            Ok(())
        }

        /// Encoded channel data for a plane of `rows` rows: the compression field, then the RLE
        /// row counts (container order) and the samples (always big-endian: copied).
        fn channel_data(&mut self, rows: usize, long_counts: bool) -> Result<()> {
            let saved = self.version;
            if long_counts {
                self.version = Version::Psb;
            }
            let compression = self.u16()?;
            let res = if compression == 1 { self.rle_counts(rows) } else { Ok(()) };
            self.version = saved;
            res?;
            self.rest()
        }

        // ---- image resources (34377) ----

        fn resources(&mut self) -> Result<()> {
            while self.remaining() >= 12 && is_sig(self.r.peek_rest(), self.from.is_little()) {
                self.key()?;
                let id = self.u16()?;
                self.pascal(2)?;
                let odd = self.r.clone().u32()? % 2 == 1;
                self.sized_region(false, |t| t.resource_data(id))?;
                if odd && self.remaining() > 0 {
                    self.raw(1)?;
                }
            }
            self.rest()
        }

        /// Typed numeric resources; every other payload is a byte blob (ICC, XMP, EXIF, …) or
        /// unknown and is copied as stored.
        fn resource_data(&mut self, id: u16) -> Result<()> {
            match id {
                // Resolution info: fixed 16.16 h res, u16 unit, u16 width unit, then the same for v.
                1005 => {
                    self.u32()?;
                    self.u16()?;
                    self.u16()?;
                    self.u32()?;
                    self.u16()?;
                    self.u16()?;
                }
                // Layer state (u16 target), layer group info (u16 per layer).
                1024 | 1026 => self.u16s_to_end()?,
                // Global angle / altitude.
                1037 | 1049 => {
                    self.u32()?;
                }
                // Layer selection ids: u16 count + u32 ids.
                1069 => {
                    self.u16()?;
                    self.u32s_to_end()?;
                }
                // Version info: u32 version, u8 has real merged data, two Unicode strings, u32.
                1057 => {
                    self.u32()?;
                    self.u8()?;
                    self.unicode()?;
                    self.unicode()?;
                    self.u32()?;
                }
                // Grid and guides: u32 version, 2 × u32 grid, u32 count, (u32 location, u8 direction)…
                1032 => {
                    self.u32()?;
                    self.u32()?;
                    self.u32()?;
                    let n = self.u32()?;
                    for _ in 0..n {
                        if self.remaining() < 5 {
                            break;
                        }
                        self.u32()?;
                        self.u8()?;
                    }
                }
                // Path resources: 26-byte records.
                2000..=2997 | 2999 => self.path_records()?,
                _ => {}
            }
            Ok(())
        }

        // ---- tagged blocks ----

        /// Tagged blocks to the end; bytes that do not form a block are copied.
        fn blocks(&mut self) -> Result<()> {
            loop {
                let rest = self.r.peek_rest();
                if rest.len() < 12 || !is_sig(rest, self.from.is_little()) {
                    return self.rest();
                }
                self.block()?;
            }
        }

        fn block(&mut self) -> Result<()> {
            let before = self.out.len();
            // A `MIB8` / `46B8` marker in big-endian data: a block kept verbatim from a
            // little-endian file. Written back as it was into little-endian output (its `8BIM`
            // signature reversed, as every signature there), copied as is otherwise.
            if !self.from.is_little() && TaggedBlock::is_foreign_order_signature(self.r.peek_rest()) {
                let sig = self.r.array::<4>()?;
                let mut key = self.r.array::<4>()?;
                let long = uses_long_length(self.version, &{
                    let mut k = key;
                    if self.to.is_little() {
                        k.reverse();
                    }
                    k
                });
                let len = self.r.len_field(long)?;
                let data = self.r.bytes_u64(len)?;
                if self.to.is_little() {
                    key.reverse();
                }
                self.out.extend_from_slice(&sig);
                self.out.extend_from_slice(&key);
                let n = self.out.len();
                self.out.extend_from_slice(if long { &[0u8; 8] } else { &[0u8; 4] });
                self.patch_len(n, long, len)?;
                self.out.extend_from_slice(data);
                let rest = self.r.peek_rest();
                let pad = (0..=3usize).find(|&k| k == rest.len() || (k < rest.len() && is_sig(&rest[k..], self.from.is_little()))).unwrap_or(0);
                let pad_bytes = self.r.bytes(pad)?;
                self.out.extend_from_slice(pad_bytes);
                return Ok(());
            }
            let sig_at = self.out.len();
            let sig = self.key()?;
            let key = self.key()?;
            let long = uses_long_length(self.version, &key);
            let len = self.r.len_field(long)?;
            let len_at = self.out.len();
            self.out.extend_from_slice(if long { &[0u8; 8] } else { &[0u8; 4] });
            let data = self.r.bytes_u64(len)?;
            let mut child = Tx::new(data, self.version, self.from, self.to);
            child.depth = self.depth + 1;
            let handled = if child.depth > MAX_DEPTH {
                Err(PsdError::LimitExceeded("tagged block nesting too deep"))
            } else if data.is_empty() {
                Ok(true)
            } else {
                child.block_body(&key)
            };
            // Padding: the smallest k in 0..=3 after which the next block (or the end) starts.
            let rest = self.r.peek_rest();
            let pad = (0..=3usize).find(|&k| k == rest.len() || (k < rest.len() && is_sig(&rest[k..], self.from.is_little()))).unwrap_or(0);
            let pad_bytes = self.r.bytes(pad)?;
            match handled {
                Ok(true) => {
                    child.rest()?;
                    self.warnings.append(&mut child.warnings);
                    self.patch_len(len_at, long, child.out.len() as u64)?;
                    self.out.extend_from_slice(&child.out);
                    self.out.extend_from_slice(pad_bytes);
                }
                Ok(false) if self.same_order() => {
                    self.patch_len(len_at, long, data.len() as u64)?;
                    self.out.extend_from_slice(data);
                    self.out.extend_from_slice(pad_bytes);
                }
                // Little-endian in, big-endian model: keep the bytes as they are, marked.
                Ok(false) if self.from.is_little() => {
                    let marker: &[u8; 4] = if &sig == b"8B64" { b"46B8" } else { b"MIB8" };
                    self.out[sig_at..sig_at + 4].copy_from_slice(marker);
                    self.patch_len(len_at, long, data.len() as u64)?;
                    self.out.extend_from_slice(data);
                    self.out.extend_from_slice(pad_bytes);
                }
                Ok(false) => {
                    self.out.truncate(before);
                    self.warnings.push(format!(
                        "{} block ({} bytes) dropped: its layout is unknown and cannot be converted to the file's byte order",
                        String::from_utf8_lossy(&key),
                        data.len()
                    ));
                }
                Err(e) if self.same_order() => return Err(PsdError::invalid(format!("{} block: {e}", String::from_utf8_lossy(&key)))),
                Err(e) if self.from.is_little() => {
                    let marker: &[u8; 4] = if &sig == b"8B64" { b"46B8" } else { b"MIB8" };
                    self.out[sig_at..sig_at + 4].copy_from_slice(marker);
                    self.patch_len(len_at, long, data.len() as u64)?;
                    self.out.extend_from_slice(data);
                    self.out.extend_from_slice(pad_bytes);
                    self.warnings.push(format!("{} block ({} bytes) kept as stored, in little-endian layout: {e}", String::from_utf8_lossy(&key), data.len()));
                }
                Err(e) => {
                    self.out.truncate(before);
                    self.warnings.push(format!("{} block ({} bytes) dropped: {e}", String::from_utf8_lossy(&key), data.len()));
                }
            }
            Ok(())
        }

        /// Transcodes a block's data by key. `Ok(false)` for keys without a known layout.
        fn block_body(&mut self, key: &[u8; 4]) -> Result<bool> {
            match key {
                b"Layr" | b"Lr16" | b"Lr32" => self.layer_info()?,
                b"luni" => self.unicode()?,
                b"lsct" | b"lsdk" => {
                    self.u32()?;
                    if self.remaining() >= 8 {
                        self.key()?;
                        self.key()?;
                    }
                    if self.remaining() >= 4 {
                        self.u32()?;
                    }
                }
                // lyvr layer version, sn2P and vowv (shape and vector flags).
                b"lyid" | b"lspf" | b"lyvr" | b"sn2P" | b"vowv" => {
                    self.u32()?;
                }
                // Text-engine global data: a textual structure, no byte order.
                b"Txt2" => self.rest()?,
                b"lnsr" => {
                    self.key()?;
                }
                // Advanced Blending flags and fill opacity: one byte, then padding (Adobe spec,
                // "Additional Layer Information"; `lmgm`/`vmgm` = layer/vector mask hides effects).
                b"clbl" | b"infx" | b"knko" | b"tsly" | b"lmgm" | b"vmgm" | b"iOpa" => {
                    self.u8()?;
                }
                b"lclr" => self.u16s_upto(4)?,
                b"fxrp" => {
                    while self.remaining() >= 8 {
                        self.f64()?;
                    }
                }
                b"brst" => self.u32s_to_end()?,
                b"LMsk" => self.u16s_upto(6)?,
                b"FMsk" => self.u16s_to_end()?,
                b"shmd" => self.metadata_setting()?,
                // Effects (lfxs: the effects as applied to a group), content credentials (CAI).
                b"lfx2" | b"lmfx" | b"lfxs" | b"CAI " => {
                    self.u32()?;
                    self.versioned_descriptor()?;
                }
                b"SoCo" | b"GdFl" | b"PtFl" | b"vstk" | b"blwh" | b"vibA" | b"CgEd" | b"artb" | b"artd" | b"cinf" | b"cust" | b"PxSD" | b"GenI" | b"OCIO"
                | b"extn" | b"pths" => {
                    self.versioned_descriptor()?;
                }
                // Color Lookup: u16 version, then the descriptor.
                b"clrL" => {
                    self.u16()?;
                    self.versioned_descriptor()?;
                }
                b"lnk2" | b"lnk3" | b"lnkD" | b"lnkE" => self.linked_layers()?,
                b"Anno" => self.annotations()?,
                b"vogk" => {
                    self.u32()?;
                    self.versioned_descriptor()?;
                }
                b"vscg" => {
                    self.key()?;
                    self.versioned_descriptor()?;
                }
                b"SoLd" | b"SoLE" => {
                    self.key()?;
                    self.u32()?;
                    self.versioned_descriptor()?;
                }
                b"PlLd" => {
                    self.key()?;
                    self.u32()?;
                    self.pascal(1)?;
                    for _ in 0..4 {
                        self.u32()?;
                    }
                    self.f64s(8)?;
                    self.u32()?;
                    self.versioned_descriptor()?;
                }
                // Type: version, transform, text version + descriptor, warp version + descriptor,
                // and the bounds as four 32-bit values (the specification says doubles; Photoshop
                // writes 16 bytes).
                b"TySh" => {
                    self.u16()?;
                    self.f64s(6)?;
                    self.u16()?;
                    self.versioned_descriptor()?;
                    self.u16()?;
                    self.versioned_descriptor()?;
                    self.i32s(4)?;
                }
                b"vmsk" | b"vsms" => {
                    self.u32()?;
                    self.u32()?;
                    self.path_records()?;
                }
                b"Patt" | b"Pat2" | b"Pat3" => {
                    while self.remaining() >= 4 {
                        let len = self.r.clone().u32()?;
                        if len == 0 {
                            self.u32()?;
                            continue;
                        }
                        self.sized_region(false, Tx::pattern)?;
                        // Each pattern is padded to a multiple of 4.
                        let pad = ((4 - len % 4) % 4) as usize;
                        self.raw(pad.min(self.remaining()))?;
                    }
                }
                b"FEid" | b"FXid" => self.filter_effects()?,
                b"lrFX" => self.legacy_effects()?,
                b"levl" => {
                    self.u16()?;
                    for _ in 0..29 * 5 {
                        if self.remaining() < 2 {
                            break;
                        }
                        self.u16()?;
                    }
                    if self.remaining() >= 4 {
                        self.key()?;
                    }
                    self.u16s_to_end()?;
                }
                b"curv" => self.curves()?,
                b"hue2" => {
                    self.u16()?;
                    self.u8()?;
                    self.u8()?;
                    self.u16s_to_end()?;
                }
                b"brit" => self.u16s_upto(3)?,
                b"thrs" | b"post" => {
                    self.u16()?;
                }
                b"nvrt" => {}
                b"expA" => {
                    self.u16()?;
                    for _ in 0..3 {
                        self.f32()?;
                    }
                }
                b"blnc" => self.u16s_upto(9)?,
                b"mixr" | b"selc" => self.u16s_to_end()?,
                b"phfl" => match self.u16()? {
                    2 => {
                        self.u16s(5)?;
                        self.u32()?;
                        self.u8()?;
                    }
                    3 => {
                        for _ in 0..4 {
                            self.u32()?;
                        }
                        self.u8()?;
                    }
                    v => return Err(PsdError::Unsupported(format!("photo filter version {v}"))),
                },
                b"grdm" => self.gradient_map()?,
                _ if !self.same_order() => return Ok(self.descriptor_shaped()),
                _ => return Ok(false),
            }
            Ok(true)
        }

        /// `lnk2` / `lnk3` / `lnkD` / `lnkE` (linked and embedded smart-object files): items of
        /// u64 length + data, each padded to 4. Item: type (`liFD` embedded, `liFE` external,
        /// `liFA` alias), version 1–7, Pascal uuid, Unicode name, file type and creator keys, u64
        /// data length, open-descriptor flag (+ descriptor), per-type fields, the file bytes for
        /// `liFD`, then the fields later versions added.
        fn linked_layers(&mut self) -> Result<()> {
            while self.remaining() >= 8 {
                let len = self.r.clone().u64()?;
                self.sized_region(true, |t| {
                    let kind = t.key()?;
                    let version = t.u32()?;
                    t.pascal(1)?;
                    t.unicode()?;
                    t.key()?;
                    t.key()?;
                    let data_len = t.u64()?;
                    if t.u8()? != 0 {
                        t.versioned_descriptor()?;
                    }
                    if &kind == b"liFE" && version > 3 {
                        t.i32()?;
                        for _ in 0..4 {
                            t.u8()?;
                        }
                        t.f64()?;
                    }
                    if &kind == b"liFE" {
                        t.u64()?;
                    }
                    if &kind == b"liFA" {
                        t.raw(8)?;
                    }
                    let data_len = usize::try_from(data_len).map_err(|_| PsdError::LimitExceeded("linked file length"))?;
                    if &kind == b"liFD" {
                        t.raw(data_len)?;
                    }
                    if version >= 5 {
                        t.unicode()?;
                    }
                    if version >= 6 {
                        t.f64()?;
                    }
                    if version >= 7 {
                        t.u8()?;
                    }
                    if &kind == b"liFE" && version == 2 {
                        t.raw(data_len)?;
                    }
                    Ok(())
                })?;
                let pad = ((4 - len % 4) % 4) as usize;
                self.raw(pad.min(self.remaining()))?;
            }
            Ok(())
        }

        /// `Anno`: u16 major, u16 minor, u32 count, then per annotation a self-counting u32
        /// length, type key, open and flags bytes, u16 optional blocks, icon and popup rects, a
        /// colour, three even-padded Pascal strings, and a self-counting data block (key, u32
        /// size, text with its own byte-order mark).
        fn annotations(&mut self) -> Result<()> {
            self.u16()?;
            self.u16()?;
            let n = self.u32()?;
            for _ in 0..n {
                if self.remaining() < 4 {
                    break;
                }
                self.sized_region_inclusive(|t| {
                    t.key()?;
                    t.u8()?;
                    t.u8()?;
                    t.u16()?;
                    t.i32s(8)?;
                    t.u16s(5)?;
                    t.pascal(2)?;
                    t.pascal(2)?;
                    t.pascal(2)?;
                    t.sized_region_inclusive(|d| {
                        d.key()?;
                        d.u32()?;
                        Ok(())
                    })
                })?;
            }
            Ok(())
        }

        // ---- layer info ----

        fn layer_info(&mut self) -> Result<()> {
            let count = self.i16()?;
            let n = usize::from(count.unsigned_abs());
            self.r.check_count(n as u64, 34)?;
            // Per layer: the channel ids and lengths, and the heights of the layer, mask and
            // real-mask rects (what each channel's RLE count table spans).
            let mut records: Vec<RecordShape> = Vec::with_capacity(n);
            for _ in 0..n {
                let top = self.i32()?;
                self.i32()?;
                let bottom = self.i32()?;
                self.i32()?;
                let nch = self.u16()?;
                if nch > crate::layer::MAX_LAYER_CHANNELS {
                    return Err(PsdError::LimitExceeded("too many channels in layer"));
                }
                let mut chans = Vec::with_capacity(usize::from(nch));
                let psb = self.version.is_psb();
                for _ in 0..nch {
                    let id = self.i16()?;
                    let len = self.len(psb)?;
                    chans.push((id, len));
                }
                let sig = self.key()?;
                if &sig != b"8BIM" {
                    return Err(PsdError::InvalidSignature { expected: "8BIM (layer blend mode)", found: sig });
                }
                self.key()?;
                for _ in 0..4 {
                    self.u8()?;
                }
                let mut heights = [height(top, bottom), 0, 0];
                let mut mask_heights = (0usize, 0usize);
                self.sized_region(false, |t| {
                    let mut mh = (0, 0);
                    t.sized_region(false, |m| {
                        mh = m.mask()?;
                        Ok(())
                    })?;
                    mask_heights = mh;
                    // Blending ranges are bytes (black low/high, white low/high per channel), so
                    // Photoshop stores them in the same order in either byte order: an Intel-order
                    // layered TIFF holds the default ranges as `00 00 ff ff`. Swapping them as u32s
                    // turned every range into `ff ff 00 00`, which hides every pixel of the layer.
                    t.sized_region(false, |x| x.rest())?;
                    t.pascal(4)?;
                    t.blocks()
                })?;
                heights[1] = mask_heights.0;
                heights[2] = mask_heights.1;
                records.push((chans, heights));
            }
            for (chans, heights) in &records {
                for &(id, len) in chans {
                    if len == 0 {
                        continue;
                    }
                    if len == 1 {
                        return Err(PsdError::invalid("channel data length 1"));
                    }
                    let rows = match id {
                        -2 => heights[1],
                        -3 => heights[2],
                        _ => heights[0],
                    };
                    self.region(len, |t| t.channel_data(rows, false))?;
                }
            }
            self.rest()
        }

        /// A layer mask record. Returns the heights of the mask and real-mask rects.
        fn mask(&mut self) -> Result<(usize, usize)> {
            let whole = self.r.peek_rest();
            if whole.is_empty() {
                return Ok((0, 0));
            }
            let top = self.i32()?;
            self.i32()?;
            let bottom = self.i32()?;
            self.i32()?;
            self.u8()?;
            let flags = self.u8()?;
            let mut real_h = 0;
            if self.remaining() <= 2 {
                return Ok((height(top, bottom), 0));
            }
            let has_parameters = flags & LayerMask::FLAG_PARAMETERS != 0;
            // The helper only looks at lengths and the parameter-flags byte, so it reads the
            // same in either order.
            let real_first = has_parameters && LayerMask::real_before_parameters(whole);
            let mut read_real = |t: &mut Tx<'_>| -> Result<()> {
                if t.remaining() >= 18 {
                    t.u8()?;
                    t.u8()?;
                    let rt = t.i32()?;
                    t.i32()?;
                    let rb = t.i32()?;
                    t.i32()?;
                    real_h = height(rt, rb);
                }
                Ok(())
            };
            if real_first {
                read_real(self)?;
            }
            if has_parameters {
                let pf = self.u8()?;
                let need = mask_parameter_bytes(pf);
                if need > self.remaining() {
                    return Err(PsdError::UnexpectedEof { offset: self.r.pos(), needed: need - self.remaining() });
                }
                if pf & 1 != 0 {
                    self.u8()?;
                }
                if pf & 2 != 0 {
                    self.f64()?;
                }
                if pf & 4 != 0 {
                    self.u8()?;
                }
                if pf & 8 != 0 {
                    self.f64()?;
                }
            }
            if !real_first {
                read_real(self)?;
            }
            Ok((height(top, bottom), real_h))
        }

        // ---- descriptors ----

        fn versioned_descriptor(&mut self) -> Result<()> {
            self.u32()?;
            self.descriptor()
        }

        fn descriptor(&mut self) -> Result<()> {
            if self.depth > MAX_DEPTH {
                return Err(PsdError::LimitExceeded("descriptor nesting too deep"));
            }
            self.depth += 1;
            self.unicode()?;
            self.id()?;
            let n = self.u32()?;
            self.r.check_count(u64::from(n), 10)?;
            for _ in 0..n {
                self.id()?;
                self.value()?;
            }
            self.depth -= 1;
            Ok(())
        }

        fn id(&mut self) -> Result<()> {
            let n = self.u32()?;
            if n == 0 {
                self.key()?;
            } else {
                self.raw(usize::try_from(n).map_err(|_| PsdError::LimitExceeded("id length"))?)?;
            }
            Ok(())
        }

        fn class(&mut self) -> Result<()> {
            self.unicode()?;
            self.id()
        }

        fn value(&mut self) -> Result<()> {
            if self.depth > MAX_DEPTH {
                return Err(PsdError::LimitExceeded("descriptor nesting too deep"));
            }
            let ty = self.key()?;
            match &ty {
                b"obj " => {
                    let n = self.u32()?;
                    self.r.check_count(u64::from(n), 8)?;
                    for _ in 0..n {
                        self.reference_item()?;
                    }
                }
                b"Objc" | b"GlbO" => self.descriptor()?,
                b"VlLs" => {
                    let n = self.u32()?;
                    self.r.check_count(u64::from(n), 4)?;
                    self.depth += 1;
                    for _ in 0..n {
                        self.value()?;
                    }
                    self.depth -= 1;
                }
                b"doub" => {
                    self.f64()?;
                }
                b"UntF" => {
                    self.key()?;
                    self.f64()?;
                }
                b"UnFl" => {
                    self.key()?;
                    let n = self.u32()?;
                    self.r.check_count(u64::from(n), 8)?;
                    self.f64s(n as usize)?;
                }
                b"TEXT" => self.unicode()?,
                b"enum" => {
                    self.id()?;
                    self.id()?;
                }
                b"long" => {
                    self.i32()?;
                }
                b"comp" => {
                    self.i64()?;
                }
                b"bool" => {
                    self.u8()?;
                }
                b"type" | b"GlbC" => self.class()?,
                b"alis" | b"Pth " | b"tdta" => {
                    let n = self.u32()?;
                    self.raw(usize::try_from(n).map_err(|_| PsdError::LimitExceeded("descriptor data length"))?)?;
                }
                b"ObAr" => {
                    self.u32()?;
                    self.descriptor()?;
                }
                other => return Err(PsdError::Unsupported(format!("descriptor OSType {:?}", String::from_utf8_lossy(other)))),
            }
            Ok(())
        }

        fn reference_item(&mut self) -> Result<()> {
            let ty = self.key()?;
            match &ty {
                b"prop" => {
                    self.class()?;
                    self.id()?;
                }
                b"Clss" => self.class()?,
                b"Enmr" => {
                    self.class()?;
                    self.id()?;
                    self.id()?;
                }
                b"rele" => {
                    self.class()?;
                    self.i32()?;
                }
                b"Idnt" | b"indx" => {
                    self.u32()?;
                }
                b"name" => {
                    self.class()?;
                    self.unicode()?;
                }
                other => return Err(PsdError::Unsupported(format!("reference item type {:?}", String::from_utf8_lossy(other)))),
            }
            Ok(())
        }

        // ---- other block layouts ----

        /// `shmd`: u32 count, then per item signature, key, copy flag, 3 bytes, u32 length, data.
        /// An item whose layout is unknown is dropped on its own (and the count patched) rather
        /// than taking the whole block with it.
        fn metadata_setting(&mut self) -> Result<()> {
            let n = self.r.u32()?;
            self.r.check_count(u64::from(n), 16)?;
            let count_at = self.out.len();
            self.out.extend_from_slice(&[0; 4]);
            let mut kept = 0u32;
            for _ in 0..n {
                let before = self.out.len();
                self.key()?;
                let key = self.key()?;
                self.u8()?;
                self.raw(3)?;
                let same = self.same_order();
                let res = self.sized_region(false, |t| match &key {
                    b"cmls" | b"mlst" | b"cust" | b"tmln" | b"sgrp" | b"cinf" => t.versioned_descriptor(),
                    // mdyn (dynamic metadata) is one u32.
                    b"mdyn" => t.u32s_to_end(),
                    _ if same => Ok(()),
                    _ if t.descriptor_shaped() => Ok(()),
                    _ => Err(PsdError::Unsupported(format!("metadata item {:?}", String::from_utf8_lossy(&key)))),
                });
                match res {
                    Ok(()) => kept += 1,
                    Err(e) if same => return Err(e),
                    Err(e) => {
                        self.out.truncate(before);
                        self.warnings.push(format!("shmd item {} dropped: {e}", String::from_utf8_lossy(&key)));
                    }
                }
            }
            self.patch_len(count_at, false, u64::from(kept))
        }

        /// 26-byte path records: a u16 selector then, for knots, six fixed-point coordinates.
        fn path_records(&mut self) -> Result<()> {
            while self.remaining() >= 26 {
                let selector = self.u16()?;
                match selector {
                    1 | 2 | 4 | 5 => self.i32s(6)?,
                    // Length records: knot count, path operation, then fields Photoshop writes as
                    // 16-bit values (zero in practice).
                    0 | 3 => self.u16s(12)?,
                    // Clipboard: four fixed-point coordinates and a resolution.
                    7 => {
                        self.i32s(5)?;
                        self.raw(4)?;
                    }
                    8 => {
                        self.u16()?;
                        self.raw(22)?;
                    }
                    _ => self.raw(24)?,
                }
            }
            Ok(())
        }

        /// One pattern of a `Patt` block (the data after its u32 length).
        fn pattern(t: &mut Tx<'_>) -> Result<()> {
            t.u32()?;
            let mode = t.u32()?;
            t.u16()?;
            t.u16()?;
            t.unicode()?;
            t.pascal(1)?;
            if mode == 2 {
                t.raw(768)?;
                // Some writers follow the palette with 4 extra bytes before the array list version.
                let next_is_version = t.r.clone().u32().is_ok_and(|v| v == 3);
                if !next_is_version && t.remaining() >= 4 {
                    t.raw(4)?;
                }
            }
            t.u32()?;
            t.sized_region(false, |v| {
                v.i32s(4)?;
                let count = v.u32()?;
                for _ in 0..count.saturating_add(2) {
                    if v.remaining() < 4 {
                        break;
                    }
                    if v.u32()? == 0 {
                        continue;
                    }
                    if v.r.clone().u32()? == 0 {
                        v.u32()?;
                        continue;
                    }
                    v.sized_region(false, |c| {
                        c.u32()?;
                        let ct = c.i32()?;
                        c.i32()?;
                        let cb = c.i32()?;
                        c.i32()?;
                        c.u16()?;
                        let comp = c.u8()?;
                        if comp == 1 {
                            let saved = c.version;
                            c.version = Version::Psd;
                            let res = c.rle_counts(height(ct, cb));
                            c.version = saved;
                            res?;
                        }
                        Ok(())
                    })?;
                }
                Ok(())
            })
        }

        /// `FEid` / `FXid`: version, u64 length, items with per-channel planes (4-byte RLE counts).
        fn filter_effects(&mut self) -> Result<()> {
            self.u32()?;
            self.sized_region(true, |t| {
                while !t.r.is_empty() {
                    t.pascal(1)?;
                    t.u32()?;
                    t.sized_region(true, |it| {
                        let top = it.i32()?;
                        it.i32()?;
                        let bottom = it.i32()?;
                        it.i32()?;
                        let rect_h = height(top, bottom);
                        it.u32()?;
                        let max = it.u32()?;
                        if max > crate::filter_effects::MAX_SLOTS {
                            return Err(PsdError::LimitExceeded("filter effects: too many channels"));
                        }
                        for _ in 0..max + 2 {
                            if it.u32()? != 0 {
                                it.effects_plane(rect_h)?;
                            }
                        }
                        Ok(())
                    })?;
                    if !t.r.is_empty() && t.u8()? != 0 {
                        let top = t.i32()?;
                        t.i32()?;
                        let bottom = t.i32()?;
                        t.i32()?;
                        t.effects_plane(height(top, bottom))?;
                    }
                }
                Ok(())
            })
        }

        fn effects_plane(&mut self, rows: usize) -> Result<()> {
            self.sized_region(true, |p| p.channel_data(rows, true))
        }

        /// `lrFX`: version, count, then effects (`cmnS`, `dsdw`, `isdw`, `oglw`, `iglw`, `bevl`,
        /// `sofi`), each a signature, key, u32 size and fixed fields.
        fn legacy_effects(&mut self) -> Result<()> {
            self.u16()?;
            let n = self.u16()?;
            for _ in 0..n {
                if self.remaining() < 12 {
                    break;
                }
                self.key()?;
                let key = self.key()?;
                self.sized_region(false, |t| {
                    // A colour: u16 space + 4 × u16 components.
                    let colour = |t: &mut Tx<'_>| t.u16s(5);
                    let blend = |t: &mut Tx<'_>| -> Result<()> {
                        t.key()?;
                        t.key()?;
                        Ok(())
                    };
                    match &key {
                        b"cmnS" => {
                            t.u32()?;
                            t.u8()?;
                            t.u16()?;
                        }
                        b"dsdw" | b"isdw" => {
                            t.u32()?;
                            t.i32s(4)?;
                            colour(t)?;
                            blend(t)?;
                            t.u8()?;
                            t.u8()?;
                            t.u8()?;
                            if t.remaining() >= 10 {
                                colour(t)?;
                            }
                        }
                        b"oglw" => {
                            t.u32()?;
                            t.i32s(2)?;
                            colour(t)?;
                            blend(t)?;
                            t.u8()?;
                            t.u8()?;
                            if t.remaining() >= 10 {
                                colour(t)?;
                            }
                        }
                        b"iglw" => {
                            t.u32()?;
                            t.i32s(2)?;
                            colour(t)?;
                            blend(t)?;
                            t.u8()?;
                            t.u8()?;
                            if t.remaining() >= 11 {
                                t.u8()?;
                                colour(t)?;
                            }
                        }
                        b"bevl" => {
                            t.u32()?;
                            t.i32s(3)?;
                            blend(t)?;
                            blend(t)?;
                            colour(t)?;
                            colour(t)?;
                            for _ in 0..6 {
                                t.u8()?;
                            }
                            if t.remaining() >= 20 {
                                colour(t)?;
                                colour(t)?;
                            }
                        }
                        b"sofi" => {
                            t.u32()?;
                            blend(t)?;
                            colour(t)?;
                            t.u8()?;
                            t.u8()?;
                            if t.remaining() >= 10 {
                                colour(t)?;
                            }
                        }
                        _ => {}
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }

        /// `curv`: pad byte, version, channel bitmap, per set bit a point list; then an optional
        /// `Crv ` extension with per-channel point lists.
        fn curves(&mut self) -> Result<()> {
            self.u8()?;
            self.u16()?;
            let bits = self.u32()?;
            for bit in 0..32 {
                if bits & (1 << bit) == 0 {
                    continue;
                }
                if self.remaining() < 2 {
                    break;
                }
                let n = self.u16()?;
                self.u16s(usize::from(n) * 2)?;
            }
            if self.remaining() >= 10 {
                self.key()?;
                self.u16()?;
                let n = self.u32()?;
                for _ in 0..n {
                    if self.remaining() < 4 {
                        break;
                    }
                    self.u16()?;
                    let pts = self.u16()?;
                    self.u16s(usize::from(pts) * 2)?;
                }
            }
            Ok(())
        }

        /// `grdm`: version, reverse, dither, [method key], name, colour stops, transparency
        /// stops, then the noise-gradient fields (a u32 seed among u16s).
        fn gradient_map(&mut self) -> Result<()> {
            let version = self.u16()?;
            self.u8()?;
            self.u8()?;
            if version == 3 {
                self.key()?;
            }
            self.unicode()?;
            let n = self.u16()?;
            for _ in 0..n {
                if self.remaining() < 20 {
                    break;
                }
                self.u32()?;
                self.u32()?;
                self.u16s(6)?;
            }
            let nt = self.u16()?;
            for _ in 0..nt {
                if self.remaining() < 10 {
                    break;
                }
                self.u32()?;
                self.u32()?;
                self.u16()?;
            }
            // expansion, interpolation, length, mode, u32 seed, then u16 fields.
            for _ in 0..4 {
                if self.remaining() < 2 {
                    return Ok(());
                }
                self.u16()?;
            }
            if self.remaining() >= 4 {
                self.u32()?;
            }
            self.u16s_to_end()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::transcode::*;
    use super::*;
    use crate::compression::Compression;
    use crate::descriptor::{Descriptor, Id, UnicodeString, Value, VersionedDescriptor};
    use crate::header::ColorMode;
    use crate::testgen;

    /// Keys the generator plants that the transcoder (rightly) does not know.
    const UNKNOWN: [&[u8; 4]; 3] = [b"zOdd", b"zNop", b"Txt2"];

    /// `d` without the blocks a little-endian round trip drops, and without the layer-info
    /// padding (which follows the body length, so it changes when blocks are dropped).
    fn known(d: &ImageSourceData) -> ImageSourceData {
        let mut d = d.clone();
        d.byte_order = ByteOrder::Big;
        d.global_blocks.retain(|b| !UNKNOWN.contains(&&b.key));
        if d.layer_info.is_none() {
            d.layer_info_key = *b"Layr";
        }
        if let Some(info) = &mut d.layer_info {
            info.padding = None;
            for l in &mut info.layers {
                l.blocks.retain(|b| !UNKNOWN.contains(&&b.key));
            }
        }
        d
    }

    fn roundtrip(d: &ImageSourceData) -> ImageSourceData {
        let (big, w) = d.to_bytes(ByteOrder::Big).unwrap();
        assert!(w.is_empty(), "{w:?}");
        let (little, w) = d.to_bytes(ByteOrder::Little).unwrap();
        assert!(w.iter().all(|w| UNKNOWN.iter().any(|k| w.starts_with(std::str::from_utf8(*k).unwrap()))), "{w:?}");
        if w.is_empty() {
            assert_eq!(big.len(), little.len());
        }
        if big.len() > 36 {
            assert_ne!(big, little);
            assert!(little[36..].starts_with(b"MIB8"));
        }
        let (back_big, w) = ImageSourceData::from_bytes(&big).unwrap();
        assert!(w.is_empty(), "{w:?}");
        let (back_little, w) = ImageSourceData::from_bytes(&little).unwrap();
        assert!(w.is_empty(), "{w:?}");
        if little.len() > 36 {
            assert_eq!(back_little.byte_order, ByteOrder::Little);
        }
        assert_eq!(known(&back_little), known(&back_big));
        // The little-endian bytes are a fixed point (from the second generation on when blocks
        // were dropped, since the layer-info padding then follows the shorter body).
        let (little2, _) = back_little.to_bytes(ByteOrder::Little).unwrap();
        let little3 = ImageSourceData::from_bytes(&little2).unwrap().0.to_bytes(ByteOrder::Little).unwrap().0;
        assert_eq!(little3, little2);
        back_big
    }

    #[test]
    fn generated_files_survive_both_orders() {
        for case in testgen::all_cases() {
            let d = ImageSourceData::from_psd(&case.file);
            let back = roundtrip(&d);
            assert_eq!(back.version, d.version, "{}", case.name);
            assert_eq!(known(&back), known(&d), "{}", case.name);
            // Nothing but the planted unknown blocks is lost on the little-endian path.
            let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
            let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
            assert_eq!(known(&back), known(&d), "{}", case.name);
        }
    }

    #[test]
    fn same_order_transcoding_is_the_identity() {
        for case in testgen::all_cases() {
            let d = ImageSourceData::from_psd(&case.file);
            let (big, _) = d.to_bytes(ByteOrder::Big).unwrap();
            let (same, w) = transcode_blocks(&big[36..], d.version, ByteOrder::Big, ByteOrder::Big).unwrap();
            assert!(w.is_empty());
            assert_eq!(same, big[36..].to_vec(), "{}", case.name);
        }
    }

    fn sample_layers() -> ImageSourceData {
        let file = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
        ImageSourceData::from_psd(&file)
    }

    #[test]
    fn signature_selects_the_version() {
        let mut d = sample_layers();
        let (b, _) = d.to_bytes(ByteOrder::Big).unwrap();
        assert!(b.starts_with(SIGNATURE));
        d.version = Version::Psb;
        let (b, _) = d.to_bytes(ByteOrder::Big).unwrap();
        assert!(b.starts_with(SIGNATURE_PSB));
        assert_eq!(ImageSourceData::from_bytes(&b).unwrap().0.version, Version::Psb);
        roundtrip(&d);
    }

    #[test]
    fn layer_key_follows_depth() {
        assert_eq!(layer_info_key(8), *b"Layr");
        assert_eq!(layer_info_key(16), *b"Lr16");
        assert_eq!(layer_info_key(32), *b"Lr32");
        for (depth, comp) in [(16, Compression::ZipPrediction), (32, Compression::ZipPrediction), (16, Compression::Rle)] {
            let file = testgen::layered(Version::Psd, ColorMode::Rgb, depth, comp);
            let d = ImageSourceData::from_psd(&file);
            assert_eq!(d.layer_info_key, layer_info_key(depth));
            let back = roundtrip(&d);
            assert_eq!(back.layer_info_key, d.layer_info_key);
            // The 16/32-bit model goes back into a PSD as a global block, like Photoshop's.
            let psd = back.into_psd(file.header.clone(), file.resources.clone(), file.image_data.clone());
            assert!(matches!(psd.layer_info_placement, LayerInfoPlacement::GlobalBlock { .. }));
            let again = PsdFile::from_bytes(&psd.to_bytes().unwrap()).unwrap();
            assert_eq!(known(&ImageSourceData::from_psd(&again)).layer_info, known(&d).layer_info);
            let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
            let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
            assert_eq!(known(&back).layer_info, known(&d).layer_info, "depth {depth}");
        }
    }

    #[test]
    fn blocks_are_padded_to_four() {
        let d = sample_layers();
        let (b, _) = d.to_bytes(ByteOrder::Big).unwrap();
        let mut at = 36;
        while at < b.len() {
            assert_eq!(&b[at..at + 4], b"8BIM");
            let len = u32::from_be_bytes(b[at + 8..at + 12].try_into().unwrap()) as usize;
            at += 12 + len;
            at += (4 - at % 4) % 4;
        }
        assert_eq!(at, b.len());
    }

    fn block_rt(key: [u8; 4], data: Vec<u8>) -> (Vec<u8>, Vec<u8>) {
        let d = ImageSourceData { global_blocks: vec![TaggedBlock::new(key, data.clone())], ..Default::default() };
        let (big, _) = d.to_bytes(ByteOrder::Big).unwrap();
        let (little, w) = d.to_bytes(ByteOrder::Little).unwrap();
        assert!(w.is_empty(), "{w:?}");
        let (back, w) = ImageSourceData::from_bytes(&little).unwrap();
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(back.global_blocks[0].data, data);
        assert_eq!(back.to_bytes(ByteOrder::Big).unwrap().0, big);
        (big, little)
    }

    #[test]
    fn descriptor_blocks_swap_keys_and_numbers() {
        let mut desc = Descriptor::new("null");
        desc.items.push((Id::new("Opct"), Value::UnitFloat { unit: *b"#Prc", value: 75.5 }));
        desc.items.push((Id::new("Nm  "), Value::Text(UnicodeString::new_nul("Drop Shadow"))));
        desc.items.push((Id::new("enab"), Value::Boolean(true)));
        desc.items.push((Id::new("Md  "), Value::Enumerated { type_id: Id::new("BlnM"), value: Id::new("Mltp") }));
        desc.items.push((Id::new("list"), Value::List(vec![Value::Integer(-3), Value::Double(1.5), Value::LargeInteger(7)])));
        desc.items.push((Id::new("longKeyName"), Value::Descriptor(Descriptor::new("innr"))));
        let mut data = 0u32.to_be_bytes().to_vec();
        data.extend(VersionedDescriptor::new(desc).to_bytes());
        let (big, little) = block_rt(*b"lfx2", data);
        // Keys are reversed and the version field swapped in the little-endian bytes.
        assert!(little[36..].starts_with(b"MIB82xfl"));
        let body_at = 48;
        assert_eq!(&little[body_at..body_at + 4], &[0, 0, 0, 0]);
        assert_eq!(&little[body_at + 4..body_at + 8], &[16, 0, 0, 0]);
        assert_eq!(&big[body_at + 4..body_at + 8], &[0, 0, 0, 16]);
        assert!(little.windows(4).any(|w| w == b"crP#"), "unit key reversed");
        assert!(little.windows(11).any(|w| w == b"longKeyName"), "string ids are not reversed");
    }

    #[test]
    fn type_block_layout() {
        let mut data = Vec::new();
        data.put_u16(1);
        for v in [1.0, 0.0, 0.0, 1.0, 12.5, -3.0] {
            data.put_f64(v);
        }
        data.put_u16(50);
        data.extend(VersionedDescriptor::new(Descriptor::new("TxLr")).to_bytes());
        data.put_u16(1);
        data.extend(VersionedDescriptor::new(Descriptor::new("warp")).to_bytes());
        for v in [0.0, 0.0, 100.0, 20.0] {
            data.put_f64(v);
        }
        let (_, little) = block_rt(*b"TySh", data);
        assert!(little[48..].starts_with(&[1, 0]));
        assert_eq!(&little[50..58], &1.0f64.to_le_bytes());
    }

    #[test]
    fn legacy_effects_and_adjustments() {
        // lrFX with a drop shadow and a solid fill.
        let mut fx = Vec::new();
        fx.put_u16(0);
        fx.put_u16(3);
        fx.put(b"8BIMcmnS");
        fx.put_u32(7);
        fx.put_u32(0);
        fx.put_u8(1);
        fx.put_u16(0);
        fx.put(b"8BIMdsdw");
        let mut sh = Vec::new();
        sh.put_u32(2);
        for v in [5 << 16, 75 << 16, 120 << 16, 7 << 16] {
            sh.put_i32(v);
        }
        for _ in 0..5 {
            sh.put_u16(0);
        }
        sh.put(b"8BIMmul ");
        sh.put_u8(1);
        sh.put_u8(1);
        sh.put_u8(191);
        sh.put_u16(0);
        sh.put_u16(0x1234);
        sh.put_u16(0);
        sh.put_u16(0);
        sh.put_u16(0);
        fx.put_u32(sh.len() as u32);
        fx.put(&sh);
        fx.put(b"8BIMsofi");
        let mut so = Vec::new();
        so.put_u32(2);
        so.put(b"8BIMnorm");
        so.put_u16(0);
        so.put_u16(0xffff);
        so.put_u16(0);
        so.put_u16(0);
        so.put_u16(0);
        so.put_u8(255);
        so.put_u8(1);
        so.put_u16(0);
        so.put_u16(0xffff);
        so.put_u16(0);
        so.put_u16(0);
        so.put_u16(0);
        fx.put_u32(so.len() as u32);
        fx.put(&so);
        let (_, little) = block_rt(*b"lrFX", fx);
        assert!(little.windows(8).any(|w| w == b"MIB8wdsd"), "effect keys reversed");
        assert!(little.windows(2).any(|w| w == [0x34, 0x12]), "colour component swapped");

        // levl: version + 29 records + the Lvls extension.
        let mut lv = Vec::new();
        lv.put_u16(2);
        for i in 0..29 * 5 {
            lv.put_u16(i as u16);
        }
        lv.put(b"Lvls");
        lv.put_u16(3);
        lv.put_u16(33);
        for i in 0..4 * 5 {
            lv.put_u16(1000 + i as u16);
        }
        let (_, little) = block_rt(*b"levl", lv);
        assert!(little.windows(4).any(|w| w == b"slvL"));

        // curv with the Crv extension.
        let mut cv = vec![0u8];
        cv.put_u16(1);
        cv.put_u32(0b101);
        cv.put_u16(2);
        cv.put_u16(0);
        cv.put_u16(0);
        cv.put_u16(255);
        cv.put_u16(255);
        cv.put_u16(1);
        cv.put_u16(128);
        cv.put_u16(100);
        cv.put(b"Crv ");
        cv.put_u16(4);
        cv.put_u32(1);
        cv.put_u16(2);
        cv.put_u16(1);
        cv.put_u16(7);
        cv.put_u16(9);
        block_rt(*b"curv", cv);

        for (key, data) in [
            (*b"hue2", vec![0, 2, 0, 0, 0, 10, 0, 20, 0, 30, 0, 40, 0, 50, 0, 60]),
            (*b"brit", vec![0, 10, 0, 20, 0, 30, 1, 0]),
            (*b"thrs", vec![0, 128, 0, 0]),
            (*b"post", vec![0, 4, 0, 0]),
            (*b"nvrt", vec![]),
            (*b"expA", [1u16.to_be_bytes().to_vec(), 0.5f32.to_be_bytes().to_vec(), 0.0f32.to_be_bytes().to_vec(), 1.0f32.to_be_bytes().to_vec()].concat()),
            (*b"blnc", vec![0; 20]),
            (*b"mixr", vec![0, 1, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0]),
            (*b"selc", vec![0, 1, 0, 0, 0, 5, 0, 6, 0, 7, 0, 8]),
            (*b"phfl", vec![0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 25, 1, 0]),
            (*b"phfl", vec![0, 2, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 0, 0, 25, 1, 0]),
            (*b"lclr", vec![0, 1, 0, 0, 0, 0, 0, 0]),
            (*b"fxrp", [1.5f64.to_be_bytes().to_vec(), 2.5f64.to_be_bytes().to_vec()].concat()),
            (*b"brst", vec![0, 0, 0, 1, 0, 0, 0, 2]),
            (*b"FMsk", vec![0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 50]),
            (*b"lsct", [3u32.to_be_bytes().to_vec(), b"8BIMpass".to_vec()].concat()),
            (*b"lspf", vec![0, 0, 0, 5]),
            (*b"lnsr", b"layr".to_vec()),
            (*b"knko", vec![1, 0, 0, 0]),
            (*b"knko", vec![2, 0, 0, 0]),
            (*b"infx", vec![1, 0, 0, 0]),
            (*b"clbl", vec![0, 0, 0, 0]),
            (*b"tsly", vec![0, 0, 0, 0]),
            (*b"lmgm", vec![1, 0, 0, 0]),
            (*b"vmgm", vec![1, 0, 0, 0]),
            (*b"iOpa", vec![128, 0, 0, 0]),
        ] {
            let (big, little) = block_rt(key, data.clone());
            if data.len() >= 2 && key != *b"nvrt" {
                assert_ne!(big, little, "{}", String::from_utf8_lossy(&key));
            }
        }
    }

    #[test]
    fn global_layer_mask_block() {
        let d = ImageSourceData { global_layer_mask: Some(GlobalLayerMask { data: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 128, 0] }), ..Default::default() };
        let (little, w) = d.to_bytes(ByteOrder::Little).unwrap();
        assert!(w.is_empty());
        assert!(little[36..].starts_with(b"MIB8ksML"));
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        assert_eq!(back.global_layer_mask, d.global_layer_mask);
        assert_eq!(back.global_layer_mask.unwrap().opacity(), Some(50));
    }

    #[test]
    fn gradient_map_layout() {
        let mut g = Vec::new();
        g.put_u16(3);
        g.put_u8(0);
        g.put_u8(1);
        g.put(b"Lnr ");
        g.put_u32(2);
        g.put_u16(u16::from(b'a'));
        g.put_u16(0);
        g.put_u16(2);
        for (loc, mid) in [(0u32, 50u32), (4096, 50)] {
            g.put_u32(loc);
            g.put_u32(mid);
            g.put_u16(0);
            g.put_u16(0x1111);
            g.put_u16(0x2222);
            g.put_u16(0x3333);
            g.put_u16(0);
            g.put_u16(0);
        }
        g.put_u16(2);
        for loc in [0u32, 4096] {
            g.put_u32(loc);
            g.put_u32(50);
            g.put_u16(100);
        }
        g.put_u16(2);
        g.put_u16(1);
        g.put_u16(32);
        g.put_u16(0);
        g.put_u32(0xdead_beef);
        for _ in 0..12 {
            g.put_u16(0);
        }
        let (_, little) = block_rt(*b"grdm", g);
        assert!(little.windows(4).any(|w| w == 0xdead_beefu32.to_le_bytes()));
        assert!(little.windows(4).any(|w| w == b" rnL"));
    }

    #[test]
    fn vector_mask_records() {
        let mut v = Vec::new();
        v.put_u32(3);
        v.put_u32(0);
        // Closed length record, then two linked knots.
        v.put_u16(0);
        v.put_u16(2);
        v.put_i16(1);
        v.put(&[0; 20]);
        for sel in [1u16, 1] {
            v.put_u16(sel);
            for i in 0..6 {
                v.put_i32((i + 1) * 0x0010_0000);
            }
        }
        let (_, little) = block_rt(*b"vmsk", v);
        assert!(little.windows(4).any(|w| w == [0, 0, 0x10, 0]), "fixed-point coordinate swapped");
    }

    #[test]
    fn pattern_block() {
        let plane: Vec<u8> = (0..16u8).collect();
        let p = crate::patterns::PsdPattern {
            mode: 3,
            width: 4,
            height: 4,
            name: "dots".into(),
            id: "abc".into(),
            palette: None,
            depth: 8,
            channels: vec![plane.clone(), plane.clone(), plane.clone()],
            alpha: Some(plane),
        };
        let data = crate::patterns::write_pattern_block(std::slice::from_ref(&p)).unwrap();
        block_rt(*b"Patt", data.clone());
        // Little-endian pattern data parses back into the same pattern.
        let d = ImageSourceData { global_blocks: vec![TaggedBlock::new(*b"Patt", data)], ..Default::default() };
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        assert_eq!(crate::patterns::parse_pattern_block(&back.global_blocks[0].data).unwrap(), vec![p]);
    }

    #[test]
    fn filter_effects_block() {
        use crate::filter_effects::{EffectsPlane, FilterEffects, FilterEffectsItem};
        let plane = EffectsPlane::encode(Compression::Rle, &[1, 2, 3, 4, 5, 6], 3, 2, 8).unwrap();
        let mask = EffectsPlane::encode(Compression::Raw, &[9; 6], 3, 2, 8).unwrap();
        let rect = crate::layer::Rect { top: 0, left: 0, bottom: 2, right: 3 };
        let fx = FilterEffects {
            version: 1,
            items: vec![FilterEffectsItem {
                id: "x".into(),
                version: 1,
                rect,
                depth: 8,
                max_channels: 1,
                slots: vec![Some(plane.clone()), None, Some(plane)],
                mask: Some((rect, mask)),
            }],
        };
        let data = fx.to_bytes();
        block_rt(*b"FEid", data.clone());
        let d = ImageSourceData { global_blocks: vec![TaggedBlock::new(*b"FEid", data)], ..Default::default() };
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        assert_eq!(FilterEffects::parse(&back.global_blocks[0].data).unwrap(), fx);
    }

    #[test]
    fn metadata_setting_items() {
        let desc = VersionedDescriptor::new(Descriptor::new("null")).to_bytes();
        let items = vec![crate::metadata::MetadataItem::new(*b"cmls", desc)];
        block_rt(*b"shmd", crate::metadata::write_shmd(&items));
        // An unknown item cannot be converted: it is dropped on its own, the rest stays.
        let items = vec![
            crate::metadata::MetadataItem::new(*b"zzzz", vec![1, 2, 3, 4, 5, 6, 7, 8]),
            crate::metadata::MetadataItem::new(*b"mdyn", vec![0, 0, 0, 1]),
            crate::metadata::MetadataItem::new(*b"cmls", VersionedDescriptor::new(Descriptor::new("null")).to_bytes()),
        ];
        let d = ImageSourceData { global_blocks: vec![TaggedBlock::new(*b"shmd", crate::metadata::write_shmd(&items))], ..Default::default() };
        let (little, w) = d.to_bytes(ByteOrder::Little).unwrap();
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].starts_with("shmd item zzzz dropped"), "{}", w[0]);
        let (back, w) = ImageSourceData::from_bytes(&little).unwrap();
        assert!(w.is_empty());
        let back_items = crate::metadata::parse_shmd(&back.global_blocks[0].data).unwrap();
        assert_eq!(back_items, items[1..].to_vec());
    }

    #[test]
    fn unknown_descriptor_shaped_blocks_are_converted() {
        // A key the transcoder has never heard of, holding a versioned descriptor (as every
        // block Photoshop added since CS6 does): converted, not dropped.
        let mut desc = Descriptor::new("null");
        desc.items.push((Id::new("Nm  "), Value::Text(UnicodeString::new_nul("x"))));
        let data = VersionedDescriptor::new(desc.clone()).to_bytes();
        block_rt(*b"Qqqq", data);
        let mut with_version = 3u32.to_be_bytes().to_vec();
        with_version.extend(VersionedDescriptor::new(desc).to_bytes());
        block_rt(*b"Qqqq", with_version);
        // Empty data needs no conversion either.
        block_rt(*b"Mt32", Vec::new());
    }

    #[test]
    fn linked_layer_items() {
        // One embedded file (liFD, version 7) as photocraft-io writes it.
        let mut item = Vec::new();
        item.put(b"liFD");
        item.put_u32(7);
        item.put_u8(4);
        item.put(b"uuid");
        let units: Vec<u16> = "a.png\0".encode_utf16().collect();
        item.put_u32(units.len() as u32);
        for u in units {
            item.put_u16(u);
        }
        item.put(b"png ");
        item.put(b"8BIM");
        let bytes = [0x89, b'P', b'N', b'G', 1, 2, 3];
        item.put_u64(bytes.len() as u64);
        item.put_u8(0);
        item.put(&bytes);
        item.put_u32(0);
        item.put_f64(0.0);
        item.put_u8(0);
        let mut data = (item.len() as u64).to_be_bytes().to_vec();
        data.put(&item);
        while !data.len().is_multiple_of(4) {
            data.push(0);
        }
        let (_, little) = block_rt(*b"lnk2", data);
        assert!(little.windows(4).any(|w| w == b"DFil"), "item type key reversed");
        assert!(little.windows(7).any(|w| w == bytes), "file bytes untouched");
    }

    #[test]
    fn annotation_block() {
        let mut a = Vec::new();
        a.put_u16(2);
        a.put_u16(1);
        a.put_u32(1);
        let mut item = Vec::new();
        item.put(b"txtA");
        item.put_u8(0);
        item.put_u8(1);
        item.put_u16(0);
        for v in [10, 20, 30, 40, 50, 60, 70, 80] {
            item.put_i32(v);
        }
        item.put_u16(0);
        for v in [0xffff, 0, 0, 0] {
            item.put_u16(v);
        }
        for s in [&b"me"[..], b"note", b"2026"] {
            item.put_u8(s.len() as u8);
            item.put(s);
            if (1 + s.len()) % 2 == 1 {
                item.put_u8(0);
            }
        }
        let text = [0xfe, 0xff, 0, b'h', 0, b'i'];
        let mut d = Vec::new();
        d.put(b"txtC");
        d.put_u32(text.len() as u32);
        d.put(&text);
        item.put_u32(d.len() as u32 + 4);
        item.put(&d);
        a.put_u32(item.len() as u32 + 4);
        a.put(&item);
        let (_, little) = block_rt(*b"Anno", a);
        assert!(little.windows(4).any(|w| w == b"Atxt"));
        assert!(little.windows(6).any(|w| w == text), "text kept as stored");
    }

    #[test]
    fn unknown_blocks_kept_in_big_dropped_in_little() {
        let mut d = sample_layers();
        d.global_blocks.push(TaggedBlock::new(*b"Zzzz", vec![1, 2, 3, 4, 5]));
        let (big, w) = d.to_bytes(ByteOrder::Big).unwrap();
        assert!(w.is_empty());
        let (back, w) = ImageSourceData::from_bytes(&big).unwrap();
        assert!(w.is_empty());
        assert_eq!(back.global_blocks, d.global_blocks);
        // Big-endian data of unknown layout cannot go into a little-endian file.
        let (little, w) = d.to_bytes(ByteOrder::Little).unwrap();
        assert_eq!(w.iter().filter(|w| w.starts_with("Zzzz block (5 bytes) dropped")).count(), 1, "{w:?}");
        let (back, w) = ImageSourceData::from_bytes(&little).unwrap();
        assert!(w.is_empty());
        assert_eq!(back.global_blocks.iter().map(|b| b.key).collect::<Vec<_>>(), vec![*b"Patt", *b"FMsk"]);
        assert_eq!(known(&back).layer_info, known(&d).layer_info);
    }

    #[test]
    fn unknown_blocks_from_a_little_endian_file_are_kept_with_their_byte_order() {
        let d = sample_layers();
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        // A little-endian file with a block of unknown layout, among known ones.
        let mut raw = little.clone();
        raw.extend_from_slice(b"MIB8zzzZ");
        raw.extend_from_slice(&6u32.to_le_bytes());
        raw.extend_from_slice(&[9, 8, 7, 6, 5, 4, 0, 0]);
        raw.extend_from_slice(b"MIB8diyl");
        raw.extend_from_slice(&4u32.to_le_bytes());
        raw.extend_from_slice(&7u32.to_le_bytes());
        let (back, w) = ImageSourceData::from_bytes(&raw).unwrap();
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(known(&back).layer_info, known(&d).layer_info);
        let kept = back.global_blocks.iter().find(|b| &b.key == b"Zzzz").expect("kept");
        assert!(kept.is_foreign_order());
        assert_eq!(kept.signature, *b"MIB8");
        assert_eq!(kept.data, [9, 8, 7, 6, 5, 4]);
        // The known block after it was still converted.
        assert_eq!(back.global_blocks.iter().find(|b| &b.key == b"lyid").unwrap().data, 7u32.to_be_bytes());
        // Written back to a little-endian TIFF, the bytes are exactly what was read.
        let (again, w) = back.to_bytes(ByteOrder::Little).unwrap();
        assert!(w.is_empty(), "{w:?}");
        let tail = &raw[raw.len() - 18 - 16..];
        assert!(again.windows(tail.len()).any(|x| x == tail), "foreign-order block restored verbatim");
        assert_eq!(ImageSourceData::from_bytes(&again).unwrap().0.global_blocks, back.global_blocks);
        // The big-endian (PSD) model keeps the marker, so a PSD writer can tell it apart.
        let (big, _) = back.to_bytes(ByteOrder::Big).unwrap();
        assert!(big.windows(8).any(|x| x == b"MIB8Zzzz"));
        assert_eq!(ImageSourceData::from_bytes(&big).unwrap().0.global_blocks, back.global_blocks);
    }

    #[test]
    fn rle_counts_follow_the_container_and_samples_do_not() {
        let file = testgen::layered(Version::Psd, ColorMode::Rgb, 16, Compression::Rle);
        let d = ImageSourceData::from_psd(&file);
        let (big, _) = d.to_bytes(ByteOrder::Big).unwrap();
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        let d = known(&d);
        let (a, b) = (d.layer_info.as_ref().unwrap(), back.layer_info.as_ref().unwrap());
        for (la, lb) in a.layers.iter().zip(&b.layers) {
            for (ca, cb) in la.channels.iter().zip(&lb.channels) {
                assert_eq!(ca, cb);
                let (w, h) = la.channel_rect(ca.id).size().unwrap();
                assert_eq!(ca.decode(w, h, 16, Version::Psd).unwrap(), cb.decode(w, h, 16, Version::Psd).unwrap());
            }
        }
        // The encoded samples (after the count table) are identical bytes in both files, while
        // the count table itself is swapped.
        let (rec, ch) = a
            .layers
            .iter()
            .flat_map(|l| l.channels.iter().filter(|c| c.id >= -1 && c.compression == Some(Compression::Rle)).map(move |c| (l, c)))
            .find(|(l, c)| l.rect.height() > 1 && c.data.len() > l.rect.height() as usize * 2 + 8)
            .unwrap();
        let table = rec.rect.height() as usize * 2;
        let samples = &ch.data[table..];
        assert!(big.windows(samples.len()).any(|w| w == samples));
        assert!(little.windows(samples.len()).any(|w| w == samples));
        let counts = &ch.data[..table];
        let swapped: Vec<u8> = counts.chunks(2).flat_map(|c| [c[1], c[0]]).collect();
        assert!(little.windows(table + samples.len()).any(|w| w[..table] == swapped[..] && w[table..] == *samples));
    }

    #[test]
    fn blending_ranges_are_bytes_in_either_order() {
        let mut file = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
        let ranges: Vec<u8> = [0u8, 0, 255, 255].repeat(10);
        let marked: Vec<u8> = (1u8..=40).collect();
        for (i, l) in file.layer_info.as_mut().unwrap().layers.iter_mut().enumerate() {
            l.blending_ranges.data = if i == 0 { marked.clone() } else { ranges.clone() };
        }
        let d = ImageSourceData::from_psd(&file);
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        // Written verbatim (what Photoshop writes in an `II` TIFF), not swapped per u32.
        assert!(little.windows(marked.len()).any(|w| w == marked));
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        for (a, b) in d.layer_info.as_ref().unwrap().layers.iter().zip(&back.layer_info.as_ref().unwrap().layers) {
            assert_eq!(a.blending_ranges, b.blending_ranges);
        }
    }

    #[test]
    fn psb_sized_data_uses_long_lengths_and_counts() {
        let file = testgen::layered(Version::Psb, ColorMode::Cmyk, 8, Compression::Rle);
        let d = ImageSourceData::from_psd(&file);
        assert_eq!(d.version, Version::Psb);
        let back = roundtrip(&d);
        assert_eq!(back.layer_info, d.layer_info);
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        assert!(little[36..].starts_with(b"MIB8"));
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        assert_eq!(known(&back).layer_info, known(&d).layer_info);
    }

    #[test]
    fn resources_both_orders() {
        let res = vec![
            ImageResource::new(1005, vec![0, 72, 0, 0, 0, 1, 0, 1, 0, 72, 0, 0, 0, 1, 0, 1]),
            ImageResource::new(1037, vec![0, 0, 0, 120]),
            ImageResource::new(1039, vec![1, 2, 3]),
            ImageResource { signature: *b"8BIM", id: 1026, name: b"grp".to_vec(), data: vec![0, 1, 0, 1, 0, 0] },
        ];
        let big = resources_to_bytes(&res).unwrap();
        let (back, w) = resources_from_bytes(&big).unwrap();
        assert!(w.is_empty());
        assert_eq!(back, res);
        let (little, w) = transcode_resources(&big, ByteOrder::Big, ByteOrder::Little).unwrap();
        assert!(w.is_empty());
        assert!(little.starts_with(b"MIB8"));
        assert_eq!(little.len(), big.len());
        assert_eq!(&little[4..6], &[0xed, 3]);
        let (back, w) = resources_from_bytes(&little).unwrap();
        assert!(w.is_empty());
        assert_eq!(back, res);
        // The ICC blob is untouched by the swap.
        assert!(little.windows(3).any(|w| w == [1, 2, 3]));
        // An odd trailing pad byte is tolerated.
        let mut padded = big.clone();
        padded.push(0);
        assert_eq!(resources_from_bytes(&padded).unwrap().0, res);
    }

    #[test]
    fn malformed_input_errors_without_panicking() {
        assert!(ImageSourceData::from_bytes(b"").is_err());
        assert!(ImageSourceData::from_bytes(b"Adobe Photoshop Document Data Block").is_err());
        let mut bad = SIGNATURE.to_vec();
        bad.extend_from_slice(b"XXXXLayr");
        assert!(ImageSourceData::from_bytes(&bad).is_err());
        let d = sample_layers();
        for order in [ByteOrder::Big, ByteOrder::Little] {
            let (bytes, _) = d.to_bytes(order).unwrap();
            for cut in (0..bytes.len()).step_by(7).chain(bytes.len().saturating_sub(8)..bytes.len()) {
                let _ = ImageSourceData::from_bytes(&bytes[..cut]);
            }
            // Bit flips in the structure (not the signature).
            for i in (36..bytes.len()).step_by(13) {
                let mut b = bytes.clone();
                b[i] ^= 0x5a;
                let _ = ImageSourceData::from_bytes(&b);
            }
        }
        // A block that claims more data than exists.
        let mut huge = SIGNATURE.to_vec();
        huge.extend_from_slice(b"8BIMluni");
        huge.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(ImageSourceData::from_bytes(&huge).is_err());
        assert!(resources_from_bytes(b"8BIM\x03\xed\x00").is_err());
    }

    #[test]
    fn into_psd_roundtrips_through_the_psd_writer() {
        let file = testgen::layered(Version::Psd, ColorMode::Grayscale, 8, Compression::Zip);
        let d = ImageSourceData::from_psd(&file);
        let (little, _) = d.to_bytes(ByteOrder::Little).unwrap();
        let (back, _) = ImageSourceData::from_bytes(&little).unwrap();
        let psd = back.into_psd(file.header.clone(), file.resources.clone(), file.image_data.clone());
        let bytes = psd.to_bytes().unwrap();
        let again = PsdFile::from_bytes(&bytes).unwrap();
        let expected = known(&ImageSourceData::from_psd(&file));
        assert_eq!(known(&ImageSourceData::from_psd(&again)).layer_info, expected.layer_info);
        assert_eq!(again.global_blocks, expected.global_blocks);
        assert_eq!(again.layer_tree().len(), file.layer_tree().len());
    }
}
