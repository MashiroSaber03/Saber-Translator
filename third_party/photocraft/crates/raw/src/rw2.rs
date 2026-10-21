//! Panasonic / Leica RW2: the bit-packed stream of recent Lumix bodies
//! (PanasonicRaw RawFormat 5) and the vendor-compressed stream of the earlier
//! ones (RawFormat 3 and 4).
//!
//! The container is TIFF-like (`IIU\0`); IFD0 holds Panasonic's own tags,
//! whose ids and meanings are public (ExifTool's PanasonicRaw table):
//! SensorWidth/Height (2, 3), Sensor{Top,Left,Bottom,Right}Border (4–7),
//! CFAPattern (9), BitsPerSample (0x0A), LinearityLimit{Red,Green,Blue}
//! (0x0E–0x10), BlackLevel{Red,Green,Blue} (0x1C–0x1E),
//! WB{Red,Green,Blue}Level (0x24–0x26), RawFormat (0x2D) and RawDataOffset
//! (0x118).
//!
//! RawFormat 5 — the packing was established by observation of sample files:
//!
//! * From RawDataOffset the data is stored in pages of 0x4000 bytes, each
//!   page split at 0x1FF8 with its second part first (the logical page is
//!   bytes 0x1FF8..0x4000 followed by bytes 0..0x1FF8).
//! * The logical stream is a sequence of 16-byte blocks, each read as a
//!   little-endian 128-bit number holding 10 12-bit (or 9 14-bit) pixels from
//!   the least significant bit up, the remaining top bits unused; rows are
//!   whole blocks.
//!
//! RawFormat 3 and 4 — the vendor-compressed layout, also established by
//! observation of sample files (verified by decoding whole frames against the
//! camera's embedded JPEG and the recorded levels):
//!
//! * The payload at RawDataOffset is a fixed-rate stream: 16 bytes per 14
//!   pixels, so a row occupies `width / 14 * 16` bytes and rows are
//!   byte-aligned. Bits are consumed down from the top of each 0x4000-byte
//!   page, through the same page split as RawFormat 5 (`(i + 0x4000 - 0x2008)
//!   % 0x4000`), reading fields LSB-first from the little-endian 16-bit word
//!   the cursor points at.
//! * Pixels arrive in groups of 14 with two predictors carried over from one
//!   group to the next (even and odd columns). Every third pixel carries a
//!   2-bit scale selector `sh` (0, 1, 2 or 4). The first value of each parity
//!   is absolute (8 + 4 bits); the rest are 8-bit deltas around 0x80 scaled
//!   by `sh`, wrapping in the low `sh` bits.
//! * The decoded samples sit 15 counts above the recorded BlackLevel tags
//!   (measured on the darkest pixels of sample files), so 15 is added when
//!   they are read.
//!
//! Every other RawFormat value is not decoded.

use crate::error::{RawError, Result};
use crate::sensor::{BlackLevels, Cfa, Rect, Sensor};
use crate::tiff::{Tiff, tag};
use crate::{Limits, RawFormat, par};

const SENSOR_WIDTH: u16 = 0x0002;
const SENSOR_HEIGHT: u16 = 0x0003;
const SENSOR_TOP_BORDER: u16 = 0x0004;
const SENSOR_LEFT_BORDER: u16 = 0x0005;
const SENSOR_BOTTOM_BORDER: u16 = 0x0006;
const SENSOR_RIGHT_BORDER: u16 = 0x0007;
const CFA_PATTERN: u16 = 0x0009;
const BITS_PER_SAMPLE: u16 = 0x000A;
const LINEARITY_LIMIT: [u16; 3] = [0x000E, 0x000F, 0x0010];
const BLACK_LEVEL: [u16; 3] = [0x001C, 0x001D, 0x001E];
const WB_LEVEL: [u16; 3] = [0x0024, 0x0025, 0x0026];
const RAW_FORMAT: u16 = 0x002D;
const RAW_DATA_OFFSET: u16 = 0x0118;

/// Page size and split point of the stored stream (observed).
const PAGE: usize = 0x4000;
const SPLIT: usize = 0x1FF8;
/// Byte split of the vendor-compressed stream (observed).
const COMPRESSED_SPLIT: usize = 0x2008;

/// Physical offset (relative to RawDataOffset) of logical byte `i`.
#[inline]
pub(crate) fn physical(i: usize) -> usize {
    let page = i - i % PAGE;
    page + (i % PAGE + SPLIT) % PAGE
}

/// Pixels per 16-byte block for `bits`-bit samples.
pub(crate) fn pixels_per_block(bits: u32) -> Option<usize> {
    match bits {
        12 => Some(10),
        14 => Some(9),
        _ => None,
    }
}

/// Unpacks one 16-byte block of `bits`-bit pixels into `out`.
#[inline]
pub(crate) fn unpack_block(b: &[u8; 16], bits: u32, out: &mut [u16]) {
    let v = u128::from_le_bytes(*b);
    let mask = (1u128 << bits) - 1;
    for (i, o) in out.iter_mut().enumerate() {
        let at = bits * i as u32;
        if at + bits > 128 {
            break;
        }
        *o = ((v >> at) & mask) as u16;
    }
}

/// Reads the bit-packed RawFormat 5 stream (`src` starts at RawDataOffset).
fn decode_packed(src: &[u8], width: usize, height: usize, bits: u32) -> Result<Vec<u16>> {
    let ppb = pixels_per_block(bits).ok_or_else(|| RawError::unsupported(format!("Panasonic RW2 with {bits}-bit samples")))?;
    if !width.is_multiple_of(ppb) {
        return Err(RawError::unsupported(format!("Panasonic RW2 {width} pixels wide (not whole {ppb}-pixel blocks)")));
    }
    let row_bytes = width / ppb * 16;
    let need = row_bytes * height;
    // Whole pages are stored; tolerate a short final page (its missing bytes read as 0).
    if src.len() < need.saturating_sub(PAGE) {
        return Err(RawError::malformed("RW2 raw data is truncated"));
    }
    let half = |l: usize| -> [u8; 8] { src.get(physical(l)..).and_then(|b| b.get(..8)).and_then(|b| <[u8; 8]>::try_from(b).ok()).unwrap_or([0; 8]) };
    let mut data = vec![0u16; width * height];
    let band = par::band_rows(width);
    par::chunks_mut(&mut data, band * width, |i, chunk| {
        for (r, out) in chunk.chunks_exact_mut(width).enumerate() {
            let at = (i * band + r) * row_bytes;
            for (k, px) in out.chunks_exact_mut(ppb).enumerate() {
                // SPLIT and PAGE are multiples of 8, so each 8-byte half is contiguous.
                let l = at + k * 16;
                let mut b = [0u8; 16];
                b[..8].copy_from_slice(&half(l));
                b[8..].copy_from_slice(&half(l + 8));
                unpack_block(&b, bits, px);
            }
        }
    });
    Ok(data)
}

/// LSB-first bit cursor of the vendor-compressed stream: it walks down from
/// the top of each 0x4000-byte page (addressed through `COMPRESSED_SPLIT`)
/// and refills by advancing one page.
struct Bits<'a> {
    buf: &'a [u8],
    base: usize,
    pos: usize,
    nbits: u32,
}

impl<'a> Bits<'a> {
    fn new(buf: &'a [u8], base: usize) -> Self {
        Bits { buf, base, pos: 0, nbits: 0 }
    }

    #[inline]
    fn get(&mut self, num: u32) -> u32 {
        if num > self.nbits {
            self.nbits += PAGE as u32 * 8;
            self.pos += PAGE;
        }
        let mut at = ((self.nbits - num) >> 3) as usize ^ 0x3FF0;
        at = (at + PAGE - COMPRESSED_SPLIT) % PAGE;
        let at = self.base + at + self.pos - PAGE;
        self.nbits -= num;
        // Out-of-range reads (a page short at the end of the file) read as 0.
        let lo = u32::from(self.buf.get(at).copied().unwrap_or(0));
        let hi = u32::from(self.buf.get(at + 1).copied().unwrap_or(0));
        ((lo | (hi << 8)) >> (self.nbits & 7)) & ((1u32 << num) - 1)
    }
}

/// Decodes one row of the vendor-compressed stream.
fn compressed_row(buf: &[u8], row_bytes: usize, row: usize, out: &mut [u16]) {
    let skip = row_bytes * row;
    let mut bits = Bits::new(buf, skip / PAGE * PAGE);
    for _ in 0..(skip % PAGE) {
        bits.get(8);
    }
    let mut sh: i32 = 0;
    for group in out.as_chunks_mut::<14>().0 {
        let mut pred = [0i32; 2];
        let mut nonz = [0i32; 2];
        for (i, px) in group.iter_mut().enumerate() {
            if i % 3 == 2 {
                sh = 4 >> (3 - bits.get(2));
            }
            let p = i & 1;
            if nonz[p] != 0 {
                let j = bits.get(8) as i32;
                if j != 0 {
                    pred[p] -= 0x80 << sh;
                    if pred[p] < 0 || sh == 4 {
                        pred[p] &= !(-1 << sh);
                    }
                    pred[p] += j << sh;
                }
            } else {
                nonz[p] = bits.get(8) as i32;
                if nonz[p] != 0 || i > 11 {
                    pred[p] = (nonz[p] << 4) | bits.get(4) as i32;
                }
            }
            *px = pred[p] as u16;
        }
    }
}

/// Reads the vendor-compressed RawFormat 3/4 stream (`src` starts at
/// RawDataOffset).
fn decode_compressed(src: &[u8], width: usize, height: usize, bits: u32) -> Result<Vec<u16>> {
    if bits != 12 {
        return Err(RawError::unsupported(format!("Panasonic RW2 with {bits}-bit compressed samples")));
    }
    if !width.is_multiple_of(14) {
        return Err(RawError::unsupported(format!("Panasonic RW2 {width} pixels wide (not whole 14-pixel blocks)")));
    }
    let row_bytes = width / 14 * 16;
    let need = row_bytes * height;
    // Whole pages are stored; tolerate a short final page (its missing bytes read as 0).
    if src.len() < need.saturating_sub(PAGE) {
        return Err(RawError::malformed("RW2 raw data is truncated"));
    }
    let mut data = vec![0u16; width * height];
    let band = par::band_rows(width);
    par::chunks_mut(&mut data, band * width, |i, chunk| {
        for (r, out) in chunk.chunks_exact_mut(width).enumerate() {
            compressed_row(src, row_bytes, i * band + r, out);
        }
    });
    Ok(data)
}

pub(crate) fn decode(t: &Tiff, limits: &Limits) -> Result<Sensor> {
    let ifd = t.ifd_at(t.first_ifd, 0).ok_or_else(|| RawError::malformed("RW2 has no IFD0"))?;
    let raw_format = t.tag_uint(&ifd, RAW_FORMAT);
    let compressed = match raw_format {
        Some(5) => false,
        Some(3 | 4) => true,
        other => {
            let f = other.map(|v| v.to_string()).unwrap_or_else(|| "unknown".into());
            return Err(RawError::unsupported(format!("Panasonic RW2 raw format {f} (vendor-compressed) is not decoded yet")));
        }
    };
    let width = t.tag_uint(&ifd, SENSOR_WIDTH).ok_or_else(|| RawError::malformed("RW2 has no sensor width"))? as usize;
    let height = t.tag_uint(&ifd, SENSOR_HEIGHT).ok_or_else(|| RawError::malformed("RW2 has no sensor height"))? as usize;
    limits.check(width as u64, height as u64, 2)?;
    let bits = t.tag_uint(&ifd, BITS_PER_SAMPLE).unwrap_or(12);
    let cfa = match t.tag_uint(&ifd, CFA_PATTERN) {
        Some(1) => "RGGB",
        Some(2) => "GRBG",
        Some(3) => "GBRG",
        Some(4) => "BGGR",
        _ => return Err(RawError::unsupported("Panasonic RW2 without a CFA pattern")),
    };
    let cfa = Cfa::bayer(cfa).ok_or_else(|| RawError::malformed("bad CFA"))?;
    let offset =
        t.tag_uint(&ifd, RAW_DATA_OFFSET).or_else(|| t.tag_uint(&ifd, tag::STRIP_OFFSETS)).ok_or_else(|| RawError::malformed("RW2 has no raw data offset"))?
            as usize;
    let src = t.data.get(offset..).ok_or_else(|| RawError::malformed("RW2 raw data lies outside the file"))?;
    let data = if compressed { decode_compressed(src, width, height, bits)? } else { decode_packed(src, width, height, bits)? };

    let full = Rect::new(0, 0, width, height);
    let level = |tg: u16| t.tag_uint(&ifd, tg).map(f64::from);
    let mut warnings = Vec::new();
    // The compressed variants record a black level 15 counts below the decoded black.
    let black_offset = if compressed { 15.0 } else { 0.0 };
    let black_level = |tg: u16| t.tag_uint(&ifd, tg).map(|v| f64::from(v) + black_offset);
    let black = match BLACK_LEVEL.map(black_level) {
        [Some(r), Some(g), Some(b)] => {
            let v = cfa.phase(0, 0).map(|c| [r, g, b][usize::from(c).min(2)] as f32);
            BlackLevels { rows: 2, cols: 2, values: v.to_vec(), delta_h: Vec::new(), delta_v: Vec::new() }
        }
        _ => {
            warnings.push("RW2: black level not recorded; assumed 0".to_string());
            BlackLevels::uniform(0.0)
        }
    };
    let max = ((1u32 << bits) - 1) as f32;
    let white = LINEARITY_LIMIT.map(|tg| level(tg).map(|v| v as f32).filter(|v| *v > 0.0 && *v <= max).unwrap_or(max));
    let camera_wb = match WB_LEVEL.map(level) {
        [Some(r), Some(g), Some(b)] if g > 0.0 => Some([r / g, 1.0, b / g]).filter(|m| m.iter().all(|v| (0.25..8.0).contains(v))),
        _ => None,
    };
    let border = [SENSOR_LEFT_BORDER, SENSOR_TOP_BORDER, SENSOR_RIGHT_BORDER, SENSOR_BOTTOM_BORDER].map(|tg| t.tag_uint(&ifd, tg).map(|v| v as usize));
    let crop = match border {
        [Some(l), Some(tp), Some(r), Some(b)] if r > l && b > tp => Rect::new(l, tp, r - l, b - tp).intersect(&full),
        _ => full,
    };
    let crop = if crop.is_empty() { full } else { crop };
    Ok(Sensor {
        format: RawFormat::Rw2,
        make: t.tag_ascii(&ifd, tag::MAKE),
        model: t.tag_ascii(&ifd, tag::MODEL),
        width,
        height,
        samples: 1,
        data,
        cfa: Some(cfa),
        linearization: None,
        black,
        white,
        active: full,
        crop,
        color: Default::default(),
        camera_wb,
        orientation: crate::tiff::orientation(t.tag_uint(&ifd, tag::ORIENTATION)),
        baseline_exposure: 0.0,
        gain_maps: Vec::new(),
        tone_curve: Vec::new(),
        black_point: 0.0,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_are_split() {
        assert_eq!(physical(0), 0x1FF8);
        assert_eq!(physical(7), 0x1FFF);
        assert_eq!(physical(8), 0x2000);
        assert_eq!(physical(0x2007), 0x3FFF);
        assert_eq!(physical(0x2008), 0);
        assert_eq!(physical(0x3FFF), 0x1FF7);
        assert_eq!(physical(0x4000), 0x5FF8);
    }

    #[test]
    fn unpacks_lsb_first() {
        let mut v: u128 = 0;
        for i in 0..10u128 {
            v |= (0x100 + i) << (12 * i);
        }
        let mut out = [0u16; 10];
        unpack_block(&v.to_le_bytes(), 12, &mut out);
        assert_eq!(out, [0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109]);
        let mut out = [0u16; 12];
        unpack_block(&[0xFF; 16], 14, &mut out);
        assert_eq!(&out[..9], &[0x3FFF; 9]);
        assert_eq!(&out[9..], &[0; 3]);
    }

    #[test]
    fn compressed_bits_walk_down_pages() {
        // A fresh cursor reads the word at 0x2007 first (the top of the first
        // page under the compressed split), then walks down.
        let mut buf = vec![0u8; 0x4000];
        buf[0x2007] = 0xAC;
        buf[0x2008] = 0x03;
        buf[0x2006] = 0x11;
        buf[0x2017] = 0x55;
        // A 4-bit field starts in the high nibble of the top word.
        let mut b = Bits::new(&buf, 0);
        assert_eq!(b.get(4), 0xA);
        let mut b = Bits::new(&buf, 0);
        assert_eq!(b.get(8), 0xAC);
        assert_eq!(b.get(8), 0x11);
        // 16 bytes later the cursor wraps to the end of the next 16-byte run.
        for _ in 0..14 {
            b.get(8);
        }
        assert_eq!(b.get(8), 0x55);
    }

    #[test]
    fn compressed_group_walks_predictors() {
        // One 14-pixel group: both parities start absolute, a delta of 0x80
        // keeps their value, and the scale selector is written before the
        // delta of every third pixel. The test writer mirrors `Bits` so the
        // fields land where the decoder reads them.
        let mut buf = vec![0u8; 0x4000];
        {
            let mut w = TestWriter { buf: &mut buf, pos: 0, nbits: 0 };
            w.put(0x10, 8);
            w.put(0x5, 4);
            w.put(0x2A, 8);
            w.put(0xA, 4);
            for i in 2..14 {
                if i % 3 == 2 {
                    w.put(0, 2);
                }
                w.put(0x80, 8);
            }
        }
        let mut out = [0u16; 14];
        compressed_row(&buf, 16, 0, &mut out);
        assert_eq!(out, [0x105, 0x2AA, 0x105, 0x2AA, 0x105, 0x2AA, 0x105, 0x2AA, 0x105, 0x2AA, 0x105, 0x2AA, 0x105, 0x2AA]);
    }

    /// Mirror of [`Bits`] for unit tests: deposits fields where the reader
    /// will pick them up.
    struct TestWriter<'a> {
        buf: &'a mut [u8],
        pos: usize,
        nbits: u32,
    }

    impl TestWriter<'_> {
        fn put(&mut self, v: u32, num: u32) {
            if num > self.nbits {
                self.nbits += PAGE as u32 * 8;
                self.pos += PAGE;
            }
            let mut at = ((self.nbits - num) >> 3) as usize ^ 0x3FF0;
            at = (at + PAGE - COMPRESSED_SPLIT) % PAGE;
            let at = at + self.pos - PAGE;
            self.nbits -= num;
            let cur = u32::from(self.buf[at]) | (u32::from(self.buf[at + 1]) << 8);
            let cur = cur | ((v & ((1 << num) - 1)) << (self.nbits & 7));
            self.buf[at] = cur as u8;
            self.buf[at + 1] = (cur >> 8) as u8;
        }
    }
}
