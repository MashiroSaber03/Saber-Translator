//! ZSoft PCX (Paintbrush) reader: strictly bounded scanline RLE, 16/256-colour
//! palettes and 8-bit three-plane RGB. Read-only: modern saves should use PNG.

use crate::SampleType;
use crate::error::CodecError;
use crate::format::Format;
use crate::image::{ChannelLayout, Image};
use crate::options::Limits;

const HEADER: usize = 128;

fn bad(reason: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(Format::Pcx, reason)
}
fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

/// PCX lacks a strong magic number. Require plausible complete header fields before
/// identifying it, rather than classifying every file starting with 0x0a as PCX.
pub(crate) fn looks_like_pcx(b: &[u8]) -> bool {
    if b.len() < HEADER || b[0] != 0x0a || !matches!(b[1], 0 | 2 | 3 | 5) || b[2] != 1 {
        return false;
    }
    let bits = b[3];
    let planes = b[65];
    let stride = u32::from(le16(b, 66));
    let (x0, y0, x1, y1) = (le16(b, 4), le16(b, 6), le16(b, 8), le16(b, 10));
    if x1 < x0 || y1 < y0 || !matches!(bits, 1 | 2 | 4 | 8) || !(1..=4).contains(&planes) {
        return false;
    }
    let width = u32::from(x1 - x0) + 1;
    let min_stride = (width * u32::from(bits)).div_ceil(8);
    stride >= min_stride && stride > 0 && (bits != 8 || matches!(planes, 1 | 3)) && (bits == 8 || u32::from(bits) * u32::from(planes) <= 4)
}

/// Read one compressed plane's row. RLE packets never cross a scanline.
fn read_row(b: &[u8], pos: &mut usize, out: &mut [u8]) -> Result<(), CodecError> {
    let mut n = 0;
    while n < out.len() {
        let code = *b.get(*pos).ok_or_else(|| bad("truncated pixel RLE"))?;
        *pos += 1;
        let (count, value) = if code >= 0xc0 {
            let count = usize::from(code & 0x3f);
            let value = *b.get(*pos).ok_or_else(|| bad("truncated RLE packet"))?;
            *pos += 1;
            (count, value)
        } else {
            (1, code)
        };
        if count == 0 || count > out.len() - n {
            return Err(bad("RLE packet crosses a scanline or has zero length"));
        }
        out[n..n + count].fill(value);
        n += count;
    }
    Ok(())
}

pub(crate) fn decode(b: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    if !looks_like_pcx(b) {
        return Err(bad("invalid PCX header or unsupported pixel layout"));
    }
    let (w, h) = (u32::from(le16(b, 8) - le16(b, 4)) + 1, u32::from(le16(b, 10) - le16(b, 6)) + 1);
    limits.check(w, h, ChannelLayout::Rgb, SampleType::U8)?;
    let (bits, planes, stride) = (usize::from(b[3]), usize::from(b[65]), usize::from(le16(b, 66)));
    let row_len = stride.checked_mul(planes).ok_or_else(|| bad("row size overflow"))?;
    let total = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(3)).ok_or_else(|| bad("pixel size overflow"))?;
    // The last 769 bytes are the optional VGA palette for 8-bit indexed images.
    let vga = bits == 8 && planes == 1 && b.len() >= HEADER + 769 && b[b.len() - 769] == 0x0c;
    if bits == 8 && planes == 1 && !vga && le16(b, 68) != 2 {
        return Err(bad("8-bit indexed PCX has no 256-colour palette"));
    }
    let image_end = if vga { b.len() - 769 } else { b.len() };
    let mut row = vec![0u8; row_len];
    let mut pixels = vec![0u8; total];
    let mut p = HEADER;
    let width = w as usize;
    for y in 0..h as usize {
        for plane in row.chunks_mut(stride) {
            read_row(&b[..image_end], &mut p, plane)?;
        }
        for x in 0..width {
            let rgb = if bits == 8 && planes == 3 {
                [row[x], row[stride + x], row[2 * stride + x]]
            } else {
                let index = if bits == 8 {
                    usize::from(row[x])
                } else {
                    let bit = x * bits;
                    let shift = 8 - bits - (bit % 8);
                    let mask = (1usize << bits) - 1;
                    (0..planes).fold(0usize, |v, plane| {
                        let sample = (usize::from(row[plane * stride + bit / 8]) >> shift) & mask;
                        v | sample << (plane * bits)
                    })
                };
                if vga {
                    let o = image_end + 1 + index * 3;
                    [b[o], b[o + 1], b[o + 2]]
                } else if bits == 8 {
                    let g = index as u8;
                    [g, g, g] // palette-info 2: gray scale
                } else {
                    let o = 16 + index * 3;
                    [b[o], b[o + 1], b[o + 2]]
                }
            };
            pixels[(y * width + x) * 3..(y * width + x + 1) * 3].copy_from_slice(&rgb);
        }
    }
    let mut image = Image::from_u8(w, h, ChannelLayout::Rgb, pixels)?;
    let (xdpi, ydpi) = (le16(b, 12), le16(b, 14));
    if xdpi > 0 && ydpi > 0 {
        image.meta.dpi = Some((f32::from(xdpi), f32::from(ydpi)));
    }
    Ok(image)
}
