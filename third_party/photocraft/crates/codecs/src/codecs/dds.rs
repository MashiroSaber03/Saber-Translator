//! DirectDraw Surface (DDS) reader: the top mip level of a 2D texture, either uncompressed
//! (8 to 32 bits per pixel described by channel masks, RGB or luminance) or block-compressed
//! BC1/BC2/BC3 (DXT1/DXT3/DXT5), with or without the DX10 header. Read-only. Cubemaps,
//! texture arrays and volume textures are refused rather than flattened to one face or slice.

use crate::SampleType;
use crate::error::CodecError;
use crate::format::Format;
use crate::image::{ChannelLayout, Image};
use crate::options::Limits;

/// `DDS ` plus the 124-byte DDS_HEADER.
const HEADER: usize = 128;
/// DDS_HEADER_DXT10, present when the FourCC is `DX10`.
const DX10_HEADER: usize = 20;

const DDSD_DEPTH: u32 = 0x80_0000;
const DDPF_ALPHAPIXELS: u32 = 0x1;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;
const DDSCAPS2_CUBEMAP: u32 = 0x200;
const DDSCAPS2_VOLUME: u32 = 0x20_0000;
const DIMENSION_TEXTURE2D: u32 = 3;
const MISC_TEXTURECUBE: u32 = 0x4;

fn bad(reason: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(Format::Dds, reason)
}
fn unsupported(reason: impl Into<String>) -> CodecError {
    CodecError::unsupported(Format::Dds, reason)
}
fn le32(b: &[u8], at: usize) -> Result<u32, CodecError> {
    let bytes = b.get(at..at + 4).and_then(|s| <[u8; 4]>::try_from(s).ok()).ok_or_else(|| bad("truncated header"))?;
    Ok(u32::from_le_bytes(bytes))
}

/// A FourCC as text when it is printable, else as the number legacy writers store there.
fn fourcc_name(fourcc: u32) -> String {
    let bytes = fourcc.to_le_bytes();
    if bytes.iter().all(|c| c.is_ascii_graphic() || *c == b' ') { bytes.iter().map(|&c| char::from(c)).collect() } else { fourcc.to_string() }
}

/// One channel of an uncompressed pixel: where it sits and its largest value.
#[derive(Clone, Copy)]
struct Field {
    shift: u32,
    max: u32,
}

impl Field {
    /// `None` for an empty mask; an error for one with gaps or wider than the pixel.
    fn new(mask: u32, bits: u32) -> Result<Option<Field>, CodecError> {
        if mask == 0 {
            return Ok(None);
        }
        let shift = mask.trailing_zeros();
        let max = mask >> shift;
        if max & max.wrapping_add(1) != 0 || (bits < 32 && mask >> bits != 0) {
            return Err(bad(format!("channel mask {mask:#x} is not a contiguous field of a {bits}-bit pixel")));
        }
        Ok(Some(Field { shift, max }))
    }

    fn get(self, pixel: u32) -> u8 {
        let v = u64::from((pixel >> self.shift) & self.max);
        let max = u64::from(self.max);
        ((v * 255 + max / 2) / max) as u8
    }
}

enum Encoding {
    /// Uncompressed: bytes per pixel, then red (or luminance), green, blue and alpha.
    Masks {
        bytes: usize,
        gray: bool,
        fields: [Option<Field>; 4],
    },
    Bc1,
    Bc2,
    Bc3,
}

impl Encoding {
    fn masks(bits: u32, masks: [u32; 4], gray: bool) -> Result<Encoding, CodecError> {
        if !matches!(bits, 8 | 16 | 24 | 32) {
            return Err(bad(format!("{bits} bits per pixel is not 8, 16, 24 or 32")));
        }
        let mut fields = [None; 4];
        for (field, mask) in fields.iter_mut().zip(masks) {
            *field = Field::new(mask, bits)?;
        }
        let colour = if gray { fields[0].is_some() } else { fields[..3].iter().any(Option::is_some) };
        if !colour {
            return Err(bad("the pixel format has no colour channel masks"));
        }
        Ok(Encoding::Masks { bytes: bits as usize / 8, gray, fields })
    }

    fn layout(&self) -> ChannelLayout {
        match self {
            Encoding::Masks { gray: true, fields, .. } if fields[3].is_some() => ChannelLayout::GrayA,
            Encoding::Masks { gray: true, .. } => ChannelLayout::Gray,
            Encoding::Masks { fields, .. } if fields[3].is_none() => ChannelLayout::Rgb,
            _ => ChannelLayout::Rgba,
        }
    }

    /// Bytes of the top mip level.
    fn level_size(&self, width: u32, height: u32) -> Option<usize> {
        let (w, h) = (usize::try_from(width).ok()?, usize::try_from(height).ok()?);
        let block = match self {
            Encoding::Masks { bytes, .. } => return w.checked_mul(*bytes)?.checked_mul(h),
            Encoding::Bc1 => 8,
            Encoding::Bc2 | Encoding::Bc3 => 16,
        };
        w.div_ceil(4).checked_mul(h.div_ceil(4))?.checked_mul(block)
    }
}

/// The legacy pixel format: a FourCC, or channel masks.
fn legacy_encoding(b: &[u8]) -> Result<Encoding, CodecError> {
    let (flags, fourcc, bits) = (le32(b, 80)?, le32(b, 84)?, le32(b, 88)?);
    let alpha = if flags & DDPF_ALPHAPIXELS != 0 { le32(b, 104)? } else { 0 };
    if flags & DDPF_FOURCC != 0 {
        return match &fourcc.to_le_bytes() {
            b"DXT1" => Ok(Encoding::Bc1),
            b"DXT3" => Ok(Encoding::Bc2),
            b"DXT5" => Ok(Encoding::Bc3),
            b"DXT2" | b"DXT4" => Err(unsupported("premultiplied-alpha DXT2/DXT4 textures are not supported")),
            _ => Err(unsupported(format!("compression `{}` is not supported (DXT1, DXT3 and DXT5 are)", fourcc_name(fourcc)))),
        };
    }
    if flags & DDPF_RGB != 0 {
        return Encoding::masks(bits, [le32(b, 92)?, le32(b, 96)?, le32(b, 100)?, alpha], false);
    }
    if flags & DDPF_LUMINANCE != 0 {
        return Encoding::masks(bits, [le32(b, 92)?, 0, 0, alpha], true);
    }
    Err(unsupported(format!("pixel format flags {flags:#x} are not RGB, luminance or a supported compression")))
}

/// The DX10 header's DXGI format, refusing anything but a single 2D texture.
fn dx10_encoding(b: &[u8]) -> Result<Encoding, CodecError> {
    let (dxgi, dimension, misc, array) = (le32(b, HEADER)?, le32(b, HEADER + 4)?, le32(b, HEADER + 8)?, le32(b, HEADER + 12)?);
    if misc & MISC_TEXTURECUBE != 0 {
        return Err(unsupported("cubemaps are not supported"));
    }
    if dimension != DIMENSION_TEXTURE2D {
        return Err(unsupported(format!("only 2D textures are supported (resource dimension {dimension})")));
    }
    if array > 1 {
        return Err(unsupported(format!("texture arrays are not supported ({array} textures)")));
    }
    const RGBA: [u32; 4] = [0xff, 0xff00, 0xff_0000, 0xff00_0000];
    const BGRA: [u32; 4] = [0xff_0000, 0xff00, 0xff, 0xff00_0000];
    const BGRX: [u32; 4] = [0xff_0000, 0xff00, 0xff, 0];
    match dxgi {
        27..=29 => Encoding::masks(32, RGBA, false),
        87 | 90 | 91 => Encoding::masks(32, BGRA, false),
        88 | 92 | 93 => Encoding::masks(32, BGRX, false),
        70..=72 => Ok(Encoding::Bc1),
        73..=75 => Ok(Encoding::Bc2),
        76..=78 => Ok(Encoding::Bc3),
        _ => Err(unsupported(format!("DXGI format {dxgi} is not supported (BC1, BC2, BC3 and 8-bit RGBA or BGRA are)"))),
    }
}

pub(crate) fn decode(b: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    if !b.starts_with(b"DDS ") {
        return Err(bad("missing the `DDS ` signature"));
    }
    if le32(b, 4)? != 124 {
        return Err(bad("header size is not 124"));
    }
    let (flags, height, width, depth, caps2) = (le32(b, 8)?, le32(b, 12)?, le32(b, 16)?, le32(b, 24)?, le32(b, 112)?);
    if caps2 & DDSCAPS2_CUBEMAP != 0 {
        return Err(unsupported("cubemaps are not supported"));
    }
    if caps2 & DDSCAPS2_VOLUME != 0 || (flags & DDSD_DEPTH != 0 && depth > 1) {
        return Err(unsupported("volume textures are not supported"));
    }
    let dx10 = le32(b, 80)? & DDPF_FOURCC != 0 && &le32(b, 84)?.to_le_bytes() == b"DX10";
    let (encoding, start) = if dx10 { (dx10_encoding(b)?, HEADER + DX10_HEADER) } else { (legacy_encoding(b)?, HEADER) };
    let layout = encoding.layout();
    limits.check(width, height, layout, SampleType::U8)?;
    let size = encoding.level_size(width, height).ok_or_else(|| bad("pixel data size overflow"))?;
    let data = b.get(start..).and_then(|d| d.get(..size)).ok_or_else(|| bad(format!("truncated pixel data: {size} bytes needed")))?;
    let (w, h) = (width as usize, height as usize);
    let channels = layout.channels();
    let total = w.checked_mul(h).and_then(|n| n.checked_mul(channels)).ok_or_else(|| bad("pixel size overflow"))?;
    let mut pixels = vec![0u8; total];
    match encoding {
        Encoding::Masks { bytes, gray, fields } => {
            for (src, dst) in data.chunks_exact(bytes).zip(pixels.chunks_exact_mut(channels)) {
                let pixel = src.iter().rev().fold(0u32, |v, &byte| v << 8 | u32::from(byte));
                let channel = |i: usize| fields[i].map_or(0, |f| f.get(pixel));
                let values = if gray { [channel(0), channel(3), 0, 0] } else { [channel(0), channel(1), channel(2), channel(3)] };
                dst.copy_from_slice(&values[..channels]);
            }
        }
        Encoding::Bc1 | Encoding::Bc2 | Encoding::Bc3 => {
            let block_size = if matches!(encoding, Encoding::Bc1) { 8 } else { 16 };
            let blocks_x = w.div_ceil(4);
            for (i, block) in data.chunks_exact(block_size).enumerate() {
                let texels = match encoding {
                    Encoding::Bc1 => bc1(block, true),
                    Encoding::Bc2 => with_alpha(bc1(block.get(8..).unwrap_or_default(), false), bc2_alpha(block)),
                    _ => with_alpha(bc1(block.get(8..).unwrap_or_default(), false), bc3_alpha(block)),
                };
                let (bx, by) = ((i % blocks_x) * 4, (i / blocks_x) * 4);
                for (t, texel) in texels.iter().enumerate() {
                    let (x, y) = (bx + t % 4, by + t / 4);
                    if x < w && y < h {
                        let at = (y * w + x) * 4;
                        if let Some(dst) = pixels.get_mut(at..at + 4) {
                            dst.copy_from_slice(texel);
                        }
                    }
                }
            }
        }
    }
    Image::from_u8(width, height, layout, pixels)
}

fn rgb565(c: u16) -> [u8; 3] {
    let scale = |v: u16, max: u16| ((u32::from(v) * 255 + u32::from(max) / 2) / u32::from(max)) as u8;
    [scale(c >> 11, 31), scale((c >> 5) & 63, 63), scale(c & 31, 31)]
}

/// A BC1 colour block. BC2 and BC3 colour blocks always have four colours; only BC1's
/// `c0 <= c1` order selects three colours and transparent black.
fn bc1(block: &[u8], one_bit_alpha: bool) -> [[u8; 4]; 16] {
    let byte = |i: usize| block.get(i).copied().unwrap_or(0);
    let (c0, c1) = (u16::from_le_bytes([byte(0), byte(1)]), u16::from_le_bytes([byte(2), byte(3)]));
    let (a, b) = (rgb565(c0), rgb565(c1));
    let mix = |wa: u32, wb: u32| {
        let d = wa + wb;
        let at = |i: usize| ((wa * u32::from(a[i]) + wb * u32::from(b[i]) + d / 2) / d) as u8;
        [at(0), at(1), at(2), 255]
    };
    let palette = if c0 > c1 || !one_bit_alpha {
        [[a[0], a[1], a[2], 255], [b[0], b[1], b[2], 255], mix(2, 1), mix(1, 2)]
    } else {
        [[a[0], a[1], a[2], 255], [b[0], b[1], b[2], 255], mix(1, 1), [0; 4]]
    };
    let indices = u32::from_le_bytes([byte(4), byte(5), byte(6), byte(7)]);
    std::array::from_fn(|t| palette[(indices >> (2 * t)) as usize & 3])
}

/// BC2: four explicit bits of alpha per texel.
fn bc2_alpha(block: &[u8]) -> [u8; 16] {
    let bits = u64::from_le_bytes(std::array::from_fn(|i| block.get(i).copied().unwrap_or(0)));
    std::array::from_fn(|t| ((bits >> (4 * t)) & 0xf) as u8 * 17)
}

/// BC3: two alpha endpoints and a 3-bit index per texel into eight values, or into six plus
/// 0 and 255 when `a0 <= a1`.
fn bc3_alpha(block: &[u8]) -> [u8; 16] {
    let byte = |i: usize| block.get(i).copied().unwrap_or(0);
    let (a0, a1) = (u32::from(byte(0)), u32::from(byte(1)));
    let mut palette = [a0, a1, 0, 0, 0, 0, 0, 255];
    if a0 > a1 {
        for (k, v) in palette.iter_mut().enumerate().skip(2) {
            let k = k as u32;
            *v = ((8 - k) * a0 + (k - 1) * a1 + 3) / 7;
        }
    } else {
        for (k, v) in palette.iter_mut().enumerate().take(6).skip(2) {
            let k = k as u32;
            *v = ((6 - k) * a0 + (k - 1) * a1 + 2) / 5;
        }
    }
    let bits = u64::from_le_bytes([byte(2), byte(3), byte(4), byte(5), byte(6), byte(7), 0, 0]);
    std::array::from_fn(|t| palette[((bits >> (3 * t)) & 7) as usize] as u8)
}

fn with_alpha(mut texels: [[u8; 4]; 16], alpha: [u8; 16]) -> [[u8; 4]; 16] {
    for (texel, a) in texels.iter_mut().zip(alpha) {
        texel[3] = a;
    }
    texels
}
