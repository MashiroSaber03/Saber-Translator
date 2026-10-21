//! The encoder settings a lossy file was written with, read back from its own data, so a save
//! can write it again the way it came (File › Save keeps a JPEG's quality and a WebP's lossy or
//! lossless encoding).
//!
//! * **JPEG**: the quality is estimated from the quantisation tables (DQT). Most encoders scale
//!   the example tables of ITU-T T.81 Annex K (K.1 luminance, K.2 chrominance) by the
//!   Independent JPEG Group's quality curve, so the quality whose scaled tables match the file's
//!   best is the one it was saved at (exact for such tables; the closest fit for others).
//! * **WebP**: a `VP8L` image is lossless; a `VP8 ` image is lossy, and its base quantiser index
//!   (RFC 6386 §9.6) maps back to a quality on the scale our encoder uses.
//!
//! Everything here reads hostile bytes: every index is checked and a malformed file gives `None`.

use crate::codecs::vp8;

/// ITU-T T.81 Annex K.1, the example luminance quantisation table, in natural (row) order.
const ANNEX_K_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, //
    12, 12, 14, 19, 26, 58, 60, 55, //
    14, 13, 16, 24, 40, 57, 69, 56, //
    14, 17, 22, 29, 51, 87, 80, 62, //
    18, 22, 37, 56, 68, 109, 103, 77, //
    24, 35, 55, 64, 81, 104, 113, 92, //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// ITU-T T.81 Annex K.2, the example chrominance quantisation table, in natural order.
const ANNEX_K_CHROMA: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// The natural-order index of the k-th coefficient in zig-zag order (T.81 Figure A.6): DQT
/// stores its 64 values in this order.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29,
    22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// One quantisation table from a DQT segment, in natural order, with its precision.
#[derive(Debug, Clone, Copy)]
struct Table {
    values: [u16; 64],
    /// 16-bit entries (Pq = 1) rather than 8-bit.
    wide: bool,
}

/// The quality (1–100, the Independent JPEG Group's scale that Save As and Export As use) a
/// JPEG was most likely saved at, from its quantisation tables; `None` when it has no usable
/// luminance table (or isn't a JPEG).
pub fn jpeg_quality(bytes: &[u8]) -> Option<u8> {
    let tables = jpeg_tables(bytes);
    let [luma, chroma, ..] = tables;
    let luma = luma?;
    let mut best: Option<(u64, u8)> = None;
    for q in 1..=100u8 {
        let mut err = table_error(&luma, &ANNEX_K_LUMA, q);
        if let Some(c) = chroma {
            err = err.saturating_add(table_error(&c, &ANNEX_K_CHROMA, q));
        }
        if best.is_none_or(|(e, _)| err < e) {
            best = Some((err, q));
        }
    }
    best.map(|(_, q)| q)
}

/// The Independent JPEG Group's quality scaling of `base` (libjpeg's `jpeg_quality_scaling` and
/// `jpeg_add_quant_table`, as published with its source): a percentage scale of 5000 / q below
/// quality 50 and 200 − 2q from it, rounded, and kept within 1..=255 (8-bit) or 1..=32767.
fn scaled(base: u16, q: u8, wide: bool) -> u32 {
    let q = u32::from(q.clamp(1, 100));
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let v = (u32::from(base) * scale + 50) / 100;
    v.clamp(1, if wide { 32767 } else { 255 })
}

/// How far `t` is from `base` scaled to quality `q`: the summed absolute difference.
fn table_error(t: &Table, base: &[u16; 64], q: u8) -> u64 {
    t.values.iter().zip(base).map(|(&v, &b)| u64::from(u32::from(v).abs_diff(scaled(b, q, t.wide)))).sum()
}

/// The quantisation tables 0..=3 defined before the first scan (a later definition of the same
/// slot replaces an earlier one).
fn jpeg_tables(b: &[u8]) -> [Option<Table>; 4] {
    let mut tables = [None; 4];
    if b.get(..2) != Some(&[0xFF, 0xD8]) {
        return tables;
    }
    let mut i = 2usize;
    while let (Some(&0xFF), Some(&marker)) = (b.get(i), b.get(i.saturating_add(1))) {
        if marker == 0xFF {
            i = i.saturating_add(1);
            continue;
        }
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            i = i.saturating_add(2);
            continue;
        }
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        let Some(len) = b.get(i + 2..i + 4).map(|l| usize::from(u16::from_be_bytes([l[0], l[1]]))) else { break };
        let Some(seg) = len.checked_sub(2).and_then(|n| b.get(i + 4..i + 4 + n)) else { break };
        if marker == 0xDB {
            read_dqt(seg, &mut tables);
        }
        i = i.saturating_add(2).saturating_add(len);
    }
    tables
}

/// The tables of one DQT segment; a truncated or malformed table ends the segment.
fn read_dqt(mut seg: &[u8], tables: &mut [Option<Table>; 4]) {
    while let Some((&pq_tq, rest)) = seg.split_first() {
        let wide = pq_tq >> 4 == 1;
        if pq_tq >> 4 > 1 {
            return;
        }
        let slot = usize::from(pq_tq & 0x0F);
        let n = if wide { 128 } else { 64 };
        let Some(data) = rest.get(..n) else { return };
        let mut values = [0u16; 64];
        for (k, &natural) in ZIGZAG.iter().enumerate() {
            let v = if wide { data.get(2 * k..2 * k + 2).map_or(0, |p| u16::from_be_bytes([p[0], p[1]])) } else { data.get(k).map_or(0, |&p| u16::from(p)) };
            if let Some(dst) = values.get_mut(natural) {
                *dst = v;
            }
        }
        if let Some(dst) = tables.get_mut(slot) {
            *dst = Some(Table { values, wide });
        }
        seg = rest.get(n..).unwrap_or(&[]);
    }
}

/// How a WebP file's image is compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebpEncoding {
    /// VP8L.
    Lossless,
    /// VP8, with the quality (0–100, our encoder's scale) its quantiser maps back to, when the
    /// frame header could be read.
    Lossy { quality: Option<u8> },
}

/// The compression of a WebP file's (first) image, or `None` when it isn't a WebP or holds no
/// image chunk.
pub fn webp_encoding(bytes: &[u8]) -> Option<WebpEncoding> {
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WEBP") {
        return None;
    }
    find_image_chunk(bytes.get(12..)?, 0)
}

/// The first `VP8 ` or `VP8L` chunk among `chunks`, looking inside animation frames.
fn find_image_chunk(mut chunks: &[u8], depth: u8) -> Option<WebpEncoding> {
    while let Some(head) = chunks.get(..8) {
        let fourcc = head.get(..4)?;
        let size = usize::try_from(u32::from_le_bytes([head[4], head[5], head[6], head[7]])).ok()?;
        let end = 8usize.checked_add(size)?;
        // A truncated last chunk still holds its header.
        let payload = chunks.get(8..end.min(chunks.len()))?;
        match fourcc {
            b"VP8L" => return Some(WebpEncoding::Lossless),
            b"VP8 " => return Some(WebpEncoding::Lossy { quality: vp8_quality(payload) }),
            // An animation frame: a 16-byte header, then the frame's own chunks.
            b"ANMF" if depth == 0 => {
                if let Some(found) = payload.get(16..).and_then(|c| find_image_chunk(c, 1)) {
                    return Some(found);
                }
            }
            _ => {}
        }
        chunks = chunks.get(end.checked_add(size % 2)?..).unwrap_or(&[]);
    }
    None
}

/// The quality a VP8 key frame's base quantiser index maps back to on our encoder's scale (the
/// quality whose index is nearest; the middle one when several share it).
fn vp8_quality(frame: &[u8]) -> Option<u8> {
    let qi = vp8_base_quantizer(frame)?;
    let distance = |q: u8| vp8::quantizer_index(q).abs_diff(qi);
    let best = (0..=100u8).map(distance).min()?;
    let matches: Vec<u8> = (0..=100u8).filter(|&q| distance(q) == best).collect();
    let mid = matches.get(matches.len() / 2)?;
    Some(*mid)
}

/// The base quantiser index of a VP8 key frame: `y_ac_qi` from its frame header (RFC 6386
/// §9.2–9.6, §19.2), averaged over the segments when segmentation sets absolute quantisers or
/// adjusted by their mean delta otherwise.
fn vp8_base_quantizer(frame: &[u8]) -> Option<u8> {
    let tag = frame.get(..3)?;
    // Bit 0 clear: a key frame, which carries the start code and dimensions.
    if tag[0] & 1 != 0 || frame.get(3..6)? != [0x9D, 0x01, 0x2A] {
        return None;
    }
    let mut d = BoolDecoder::new(frame.get(10..)?);
    d.literal(1); // color_space
    d.literal(1); // clamping_type
    let mut segments: Option<(bool, [i32; 4])> = None;
    if d.flag() {
        let update_map = d.flag();
        if d.flag() {
            let absolute = d.flag();
            let mut q = [0i32; 4];
            for v in &mut q {
                *v = if d.flag() { d.signed(7) } else { 0 };
            }
            for _ in 0..4 {
                if d.flag() {
                    d.signed(6);
                }
            }
            segments = Some((absolute, q));
        }
        if update_map {
            for _ in 0..3 {
                if d.flag() {
                    d.literal(8);
                }
            }
        }
    }
    d.literal(1); // filter_type
    d.literal(6); // loop_filter_level
    d.literal(3); // sharpness_level
    if d.flag() && d.flag() {
        // loop_filter_adj_enable, mode_ref_lf_delta_update: 4 reference and 4 mode deltas.
        for _ in 0..8 {
            if d.flag() {
                d.signed(6);
            }
        }
    }
    d.literal(2); // log2_nbr_of_dct_partitions
    if d.exhausted() {
        return None;
    }
    let base = d.literal(7) as i32;
    let qi = match segments {
        Some((true, q)) => q.iter().sum::<i32>() / 4,
        Some((false, q)) => base + q.iter().sum::<i32>() / 4,
        None => base,
    };
    u8::try_from(qi.clamp(0, 127)).ok()
}

/// RFC 6386 §7's boolean decoder, reading zeros past the end of its data.
struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    value: u32,
    range: u32,
    bit_count: u32,
    /// Bytes read past the end of the data.
    overrun: usize,
}

impl<'a> BoolDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut d = BoolDecoder { data, pos: 0, value: 0, range: 255, bit_count: 0, overrun: 0 };
        d.value = (d.next_byte() << 8) | d.next_byte();
        d
    }

    fn next_byte(&mut self) -> u32 {
        match self.data.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                u32::from(b)
            }
            None => {
                self.overrun = self.overrun.saturating_add(1);
                0
            }
        }
    }

    /// The header was read from beyond the end of the data (a couple of bytes of look-ahead
    /// are normal).
    fn exhausted(&self) -> bool {
        self.overrun > 2
    }

    fn bool(&mut self, prob: u32) -> bool {
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let big_split = split << 8;
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.bit_count = 0;
                self.value |= self.next_byte();
            }
        }
        bit
    }

    fn flag(&mut self) -> bool {
        self.bool(128)
    }

    fn literal(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |v, _| (v << 1) | u32::from(self.flag()))
    }

    /// A magnitude of `bits` bits followed by a sign bit.
    fn signed(&mut self, bits: u32) -> i32 {
        let v = self.literal(bits) as i32;
        if self.flag() { -v } else { v }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_matches_the_published_curve() {
        // Quality 50 is the Annex K table itself; 100 is all ones; 75 halves it.
        assert_eq!(scaled(16, 50, false), 16);
        assert_eq!(scaled(121, 100, false), 1);
        assert_eq!(scaled(16, 75, false), 8);
        assert_eq!(scaled(121, 1, false), 255);
        assert_eq!(scaled(121, 1, true), 6050);
    }

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for &n in &ZIGZAG {
            seen[n] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
