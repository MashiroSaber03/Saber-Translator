//! Nikon Huffman-compressed NEF sensor data (TIFF compression 34713):
//! "lossless compressed" and "lossy compressed" (type 1 and type 2), 12 and
//! 14 bits.
//!
//! **Clean-room.** No raw-decoder source code (dcraw, LibRaw, rawspeed,
//! rawloader / rawler, darktable, RawTherapee, libopenraw, ExifTool's code)
//! was read. Sources:
//!
//! * Prose format descriptions: Laurent Clévy, "Nikon Electronic File (NEF)
//!   format" (maker-note layout and the `0x0096` linearization-table layout:
//!   version bytes, four `u16` predictor seeds at offset 2, curve size at 10,
//!   curve at 12, split row at 562); Bill Claff, "NEF Compression" (a lossy
//!   encoding curve followed by a fixed-table Huffman stage); ExifTool's Nikon
//!   tag-name documentation (`0x0093` NEFCompression, `0x0096`
//!   NEFLinearizationTable); ITU-T T.81 Annex F / H for the "difference
//!   category + additional bits" coding and canonical Huffman codes.
//! * The Huffman tables, which Nikon files do not carry, come from the
//!   black-box analysis documented in LightCraft (storytold/lightcraft#86,
//!   `crates/raw/src/vendor/nefc.rs`): recovered from CC0 raw.pixls.us samples
//!   shot in both compressed and uncompressed mode, and verified on ~45 files
//!   from ~35 bodies (D40 … D850, Df, Z 6, Z 50, 1 J1). Only the format facts
//!   and the tables were taken over; this module is written against this
//!   crate's own TIFF reader.
//!
//! Format:
//!
//! * One strip; an MSB-first bit stream without byte stuffing, continuous
//!   across rows. Per pixel a Huffman code gives the category `n`, followed by
//!   `n` additional bits (T.81: a leading 0 means a negative difference).
//! * Each pixel is predicted from the same-colour pixel two to the left; the
//!   first two pixels of a row from the first two pixels of the previous row
//!   of the same parity, starting at the seeds of maker note `0x0096`.
//! * Lossless (`0x0096` version `0x46`): the decoded values are the samples.
//!   Lossy (`0x44`): they index a curve. Type 1 (`0x44 0x10`) stores the whole
//!   curve, type 2 (`0x44 0x20` / `0x44 0x40`) stores `n` points spaced
//!   `2^bits / (n - 1)` codes apart, linearly interpolated.
//! * Values in the maker note are in the maker note's byte order.
//!
//! Not decoded (reported as unsupported; the embedded preview is used
//! instead): "lossy after split" files (non-zero split row), other `0x0096`
//! versions and bit depths other than 12 / 14.

use crate::Limits;
use crate::error::{RawError, Result};
use crate::sensor::Plane;
use crate::tiff::{Ifd, Tiff, tag};

/// TIFF Compression value of Nikon's Huffman-coded NEF data.
pub(crate) const NIKON_COMPRESSION: u32 = 34713;
/// Maker note NEFLinearizationTable.
pub(crate) const NIKON_LINEARIZATION_TABLE: u16 = 0x0096;

/// A code word that never occurred in any analysed sample.
const UNSEEN: u8 = u8::MAX;

/// Canonical Huffman table: the number of codes of each length 1..=16 and the
/// symbols (difference categories) in code order.
struct Table {
    counts: &'static [u8],
    symbols: &'static [u8],
}

const LOSSLESS_12: Table = Table { counts: &[0, 1, 4, 2, 3, 1, 2], symbols: &[5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, 11, 12] };
const LOSSLESS_14: Table = Table { counts: &[0, 1, 4, 2, 2, 3, 1, 2], symbols: &[7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14] };
const LOSSY_12: Table = Table { counts: &[0, 1, 5, 1, 1, 1, 1, 1, 1], symbols: &[5, 4, 3, 6, 2, 7, 1, 0, 8, 9, UNSEEN, 10] };
const LOSSY_14: Table = Table { counts: &[0, 1, 4, 3, 1, 1, 1, 1, 1], symbols: &[5, 6, 4, 7, 8, 3, 9, 2, 1, 0, 10, 11, 12] };

/// Longest code word in the tables above.
const MAX_LEN: u32 = 9;

/// A Huffman decoder: a lookup on the next [`MAX_LEN`] bits.
struct Huffman {
    /// `(category, code length)`; length 0 = no such code.
    lut: Vec<(u8, u8)>,
}

impl Huffman {
    fn new(t: &Table) -> Result<Huffman> {
        let mut lut = vec![(0u8, 0u8); 1 << MAX_LEN];
        let mut code = 0u32;
        let mut symbols = t.symbols.iter();
        for (i, &n) in t.counts.iter().enumerate() {
            let len = i as u32 + 1;
            if len > MAX_LEN && n > 0 {
                return Err(RawError::malformed("NEF Huffman code longer than the lookup"));
            }
            for _ in 0..n {
                let &s = symbols.next().ok_or_else(|| RawError::malformed("NEF Huffman table is short of symbols"))?;
                let first = (code << (MAX_LEN - len)) as usize;
                let span = 1usize << (MAX_LEN - len);
                if s != UNSEEN {
                    for e in lut.get_mut(first..first + span).ok_or_else(|| RawError::malformed("NEF Huffman table overflows"))? {
                        *e = (s, len as u8);
                    }
                }
                code += 1;
            }
            code <<= 1;
        }
        Ok(Huffman { lut })
    }
}

/// MSB-first bit reader without byte stuffing. Reads past the end return
/// zero bits; [`Bits::consumed`] tells how far it got.
struct Bits<'a> {
    src: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
    consumed: usize,
}

impl<'a> Bits<'a> {
    fn new(src: &'a [u8]) -> Self {
        Bits { src, pos: 0, acc: 0, n: 0, consumed: 0 }
    }

    #[inline]
    fn fill(&mut self) {
        while self.n <= 56 {
            let b = self.src.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.acc |= u64::from(b) << (56 - self.n);
            self.n += 8;
        }
    }

    #[inline]
    fn peek(&mut self, k: u32) -> u32 {
        self.fill();
        (self.acc >> (64 - k)) as u32
    }

    #[inline]
    fn skip(&mut self, k: u32) {
        self.acc <<= k;
        self.n -= k;
        self.consumed += k as usize;
    }

    /// `k` bits (k ≤ 32) as a number.
    #[inline]
    fn take(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.skip(k);
        v
    }
}

/// One coded difference, or `None` for a code word not in the table.
#[inline]
fn diff(b: &mut Bits, h: &Huffman) -> Option<i32> {
    let &(cat, len) = h.lut.get(b.peek(MAX_LEN) as usize)?;
    if len == 0 {
        return None;
    }
    b.skip(u32::from(len));
    let cat = u32::from(cat);
    if cat > 16 {
        return None;
    }
    let v = b.take(cat) as i32;
    // T.81 F.1.2.1: a leading 0 bit means a negative difference.
    Some(if cat > 0 && v < 1 << (cat - 1) { v - (1 << cat) + 1 } else { v })
}

/// How the decoded values map to samples.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Encoding {
    /// The decoded values are the samples.
    Lossless,
    /// The decoded values index this curve.
    Lossy(Vec<u16>),
}

/// Maker note `0x0096`, parsed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecodeTable {
    pub encoding: Encoding,
    /// Predictor seeds `[row parity][column]`.
    pub seeds: [[i32; 2]; 2],
}

/// Parses maker note `0x0096` for a `bits`-bit image; `le` is the maker
/// note's byte order.
pub(crate) fn parse_table(t: &[u8], le: bool, bits: u32) -> Result<DecodeTable> {
    let u16_at = |o: usize| {
        let s: [u8; 2] = t.get(o..o.checked_add(2)?)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) })
    };
    let short = || RawError::malformed(format!("NEF linearization table too short ({} bytes)", t.len()));
    if bits != 12 && bits != 14 {
        return Err(RawError::unsupported(format!("Nikon compressed NEF with {bits}-bit samples")));
    }
    let (Some(&v0), Some(&v1)) = (t.first(), t.get(1)) else { return Err(short()) };
    let mut seeds = [[0i32; 2]; 2];
    for (i, s) in seeds.iter_mut().flatten().enumerate() {
        *s = i32::from(u16_at(2 + 2 * i).ok_or_else(short)?);
    }
    let encoding = match (v0, v1) {
        (0x46, _) => Encoding::Lossless,
        (0x44, 0x10 | 0x20 | 0x40) => {
            let n = usize::from(u16_at(10).ok_or_else(short)?);
            if n < 2 {
                return Err(RawError::malformed(format!("NEF linearization curve of {n} points")));
            }
            let points: Vec<u16> = (0..n).map(|i| u16_at(12 + 2 * i)).collect::<Option<_>>().ok_or_else(short)?;
            if v1 == 0x10 {
                Encoding::Lossy(points)
            } else {
                let split = u16_at(562).unwrap_or(0);
                if split != 0 {
                    return Err(RawError::unsupported(format!("Nikon \"lossy after split\" compressed NEF (split at row {split})")));
                }
                let range = 1usize << bits;
                if n - 1 > range || !range.is_multiple_of(n - 1) {
                    return Err(RawError::unsupported(format!("Nikon compressed NEF: {n}-point curve for {bits}-bit data")));
                }
                let step = (range / (n - 1)) as u32;
                let mut curve = Vec::with_capacity(range + 1);
                for p in points.windows(2) {
                    let (a, b) = (u32::from(p.first().copied().unwrap_or(0)), u32::from(p.get(1).copied().unwrap_or(0)));
                    for k in 0..step {
                        curve.push(((a * (step - k) + b * k) / step) as u16);
                    }
                }
                curve.extend(points.last().copied());
                Encoding::Lossy(curve)
            }
        }
        _ => return Err(RawError::unsupported(format!("Nikon compressed NEF version {v0:#04x} {v1:#04x}"))),
    };
    Ok(DecodeTable { encoding, seeds })
}

/// Decodes a `w × h` Nikon Huffman-compressed strip with `bits`-bit samples.
pub(crate) fn decode(src: &[u8], w: usize, h: usize, bits: u32, table: &DecodeTable) -> Result<Vec<u16>> {
    if bits != 12 && bits != 14 {
        return Err(RawError::unsupported(format!("Nikon compressed NEF with {bits}-bit samples")));
    }
    let n = w.checked_mul(h).filter(|&n| n > 0).ok_or_else(|| RawError::malformed("NEF image is empty"))?;
    // Every code word is at least 2 bits: a strip that can't hold the image is
    // truncated (this also bounds the allocation by the file size).
    if n / 4 > src.len() {
        return Err(RawError::malformed(format!("NEF: {} bytes of compressed data for {w}x{h} pixels", src.len())));
    }
    let huff = Huffman::new(match (&table.encoding, bits) {
        (Encoding::Lossless, 12) => &LOSSLESS_12,
        (Encoding::Lossless, _) => &LOSSLESS_14,
        (Encoding::Lossy(_), 12) => &LOSSY_12,
        (Encoding::Lossy(_), _) => &LOSSY_14,
    })?;
    let (lut, max): (&[u16], i32) = match &table.encoding {
        Encoding::Lossless => (&[], (1i32 << bits) - 1),
        Encoding::Lossy(curve) => (curve, i32::try_from(curve.len()).unwrap_or(i32::MAX) - 1),
    };
    let mut out = vec![0u16; n];
    let mut stream = Bits::new(src);
    let mut vpred = table.seeds;
    let mut clipped = 0usize;
    for (y, row) in out.chunks_exact_mut(w).enumerate() {
        let seeds = &mut vpred[y & 1];
        let mut hpred = [0i32; 2];
        for (x, o) in row.iter_mut().enumerate() {
            let d = diff(&mut stream, &huff).ok_or_else(|| RawError::malformed(format!("NEF: invalid Huffman code in row {y}")))?;
            let p = if x < 2 {
                seeds[x] = seeds[x].saturating_add(d);
                hpred[x] = seeds[x];
                seeds[x]
            } else {
                hpred[x & 1] = hpred[x & 1].saturating_add(d);
                hpred[x & 1]
            };
            let v = p.clamp(0, max);
            clipped += usize::from(v != p);
            *o = if lut.is_empty() { v as u16 } else { lut.get(v as usize).copied().unwrap_or(0) };
        }
        // The camera pads the strip; reading well past its end means the data
        // is truncated or corrupt.
        if stream.consumed > src.len() * 8 + 64 {
            return Err(RawError::malformed(format!("NEF: compressed data ends in row {y} of {h}")));
        }
    }
    // A wrong table or a corrupt stream drifts out of range quickly; real
    // files stay inside (a handful of edge pixels).
    if clipped > n / 1000 + 16 {
        return Err(RawError::malformed(format!("NEF: {clipped} decoded values out of range")));
    }
    Ok(out)
}

/// Reads the Huffman-compressed CFA image of `raw`. `note` is the Nikon maker
/// note (its own TIFF and IFD 0), which holds the decode table.
pub(crate) fn read_compressed(t: &Tiff, raw: &Ifd, note: Option<&(Tiff, Ifd)>, limits: &Limits) -> Result<Plane> {
    let width = t.tag_uint(raw, tag::IMAGE_WIDTH).ok_or_else(|| RawError::malformed("raw image has no width"))? as usize;
    let height = t.tag_uint(raw, tag::IMAGE_LENGTH).ok_or_else(|| RawError::malformed("raw image has no height"))? as usize;
    let samples = t.tag_uint(raw, tag::SAMPLES_PER_PIXEL).unwrap_or(1);
    if samples != 1 {
        return Err(RawError::unsupported(format!("Nikon compressed NEF with {samples} samples per pixel")));
    }
    let bits = t.tag_uint(raw, tag::BITS_PER_SAMPLE).unwrap_or(0);
    let table = note
        .and_then(|(n, m)| Some((n.raw(m.get(NIKON_LINEARIZATION_TABLE)?)?, n.le)))
        .ok_or_else(|| RawError::unsupported("Nikon compressed NEF without a linearization table (maker note 0x96)"))?;
    let table = parse_table(table.0, table.1, bits)?;
    limits.check(width as u64, height as u64, 2)?;
    // One strip in every sample; several would be consecutive parts of the
    // same bit stream.
    let offsets = t.tag_uints(raw, tag::STRIP_OFFSETS);
    let counts = t.tag_uints(raw, tag::STRIP_BYTE_COUNTS);
    let (Some(&first), Some(&last), Some(&last_len)) = (offsets.first(), offsets.last(), counts.last()) else {
        return Err(RawError::malformed("NEF: no image data"));
    };
    let start = first as usize;
    let end = (last as usize).saturating_add(last_len as usize).min(t.data.len());
    let src = t.data.get(start..end).filter(|s| !s.is_empty()).ok_or_else(|| RawError::malformed("raw data lies outside the file"))?;
    let data = decode(src, width, height, bits, &table)?;
    Ok(Plane { width, height, samples: 1, bits, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer for the test encoder.
    #[derive(Default)]
    struct Writer {
        out: Vec<u8>,
        acc: u64,
        n: u32,
    }

    impl Writer {
        fn put(&mut self, v: u32, len: u32) {
            for i in (0..len).rev() {
                self.acc = (self.acc << 1) | u64::from((v >> i) & 1);
                self.n += 1;
                if self.n == 8 {
                    self.out.push(self.acc as u8);
                    self.acc = 0;
                    self.n = 0;
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.out.push((self.acc << (8 - self.n)) as u8);
            }
            self.out.extend_from_slice(&[0; 4]);
            self.out
        }
    }

    /// `(code, length)` per category, from the canonical table.
    fn codes(t: &Table) -> Vec<(u32, u32)> {
        let mut out = vec![(0, 0); 17];
        let mut code = 0u32;
        let mut symbols = t.symbols.iter();
        for (i, &n) in t.counts.iter().enumerate() {
            for _ in 0..n {
                let s = *symbols.next().unwrap();
                if s != UNSEEN {
                    out[s as usize] = (code, i as u32 + 1);
                }
                code += 1;
            }
            code <<= 1;
        }
        out
    }

    /// Encodes `values` (samples or curve indices) the way the camera does.
    fn encode(values: &[i32], w: usize, t: &Table, seeds: [[i32; 2]; 2]) -> Vec<u8> {
        let codes = codes(t);
        let mut wr = Writer::default();
        let mut vpred = seeds;
        for (y, row) in values.chunks(w).enumerate() {
            for (x, &v) in row.iter().enumerate() {
                let pred = if x < 2 { vpred[y & 1][x] } else { row[x - 2] };
                if x < 2 {
                    vpred[y & 1][x] = v;
                }
                let d = v - pred;
                let k = 32 - d.unsigned_abs().leading_zeros();
                let (code, len) = codes[k as usize];
                assert!(len > 0, "no code for category {k}");
                wr.put(code, len);
                wr.put(if d < 0 { (d + (1 << k) - 1) as u32 } else { d as u32 }, k);
            }
        }
        wr.finish()
    }

    /// Smooth gradient + texture + hard edges, within `0..=max`.
    fn image(w: usize, h: usize, max: i32, seed: u32) -> Vec<i32> {
        let mut s = seed;
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (s >> 24) as i32 % 9 - 4;
                let base = (x * 37 + y * 11) % (max / 2) + if (x / 7 + y / 5) % 3 == 0 { max / 3 } else { 0 };
                (base + noise).clamp(0, max)
            })
            .collect()
    }

    fn lossless(bits: u32) -> DecodeTable {
        let s = 1 << (bits - 3);
        DecodeTable { encoding: Encoding::Lossless, seeds: [[s, s], [s, s]] }
    }

    fn lossy_table(bits: u32, le: bool) -> Vec<u8> {
        let mut t = vec![0x44, 0x20];
        let put = |t: &mut Vec<u8>, v: u16| t.extend_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
        for _ in 0..4 {
            put(&mut t, 300);
        }
        put(&mut t, 257);
        let max = f64::from((1u32 << bits) - 1);
        for i in 0..257u32 {
            put(&mut t, (max * (f64::from(i) / 64.0).min(1.0).sqrt()) as u16);
        }
        t.resize(624, 0);
        t
    }

    #[test]
    fn tables_are_prefix_codes() {
        for t in [LOSSLESS_12, LOSSLESS_14, LOSSY_12, LOSSY_14] {
            assert_eq!(t.counts.iter().map(|&c| c as usize).sum::<usize>(), t.symbols.len());
            let kraft: f64 = t.counts.iter().enumerate().map(|(i, &c)| f64::from(c) / f64::from(1u32 << (i + 1))).sum();
            assert!(kraft <= 1.0);
            assert!(Huffman::new(&t).is_ok());
        }
    }

    #[test]
    fn lossless_round_trip_12_and_14() {
        for (bits, t) in [(12, LOSSLESS_12), (14, LOSSLESS_14)] {
            let (w, h) = (64, 9);
            let max = (1 << bits) - 1;
            let mut img = image(w, h, max, bits);
            img[5] = 0;
            img[7] = max;
            img[w + 9] = max;
            let table = lossless(bits);
            let out = decode(&encode(&img, w, &t, table.seeds), w, h, bits, &table).unwrap();
            assert_eq!(out, img.iter().map(|&v| v as u16).collect::<Vec<_>>(), "{bits}-bit");
        }
    }

    #[test]
    fn lossy_type2_curve_and_round_trip() {
        for (bits, t) in [(12u32, LOSSY_12), (14, LOSSY_14)] {
            for le in [false, true] {
                let table = parse_table(&lossy_table(bits, le), le, bits).unwrap();
                assert_eq!(table.seeds, [[300, 300], [300, 300]]);
                let Encoding::Lossy(curve) = &table.encoding else { panic!() };
                let step = (1usize << bits) / 256;
                assert_eq!(curve.len(), (1 << bits) + 1);
                assert_eq!(curve[64 * step], (1 << bits) - 1);
                assert!(curve.windows(2).all(|p| p[0] <= p[1]));
                let (w, h) = (48, 7);
                let img = image(w, h, 64 * step as i32 - 1, 7);
                let out = decode(&encode(&img, w, &t, table.seeds), w, h, bits, &table).unwrap();
                assert_eq!(out, img.iter().map(|&v| curve[v as usize]).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn lossy_type1_full_curve() {
        let mut t = vec![0x44, 0x10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 5];
        for v in [0u16, 100, 400, 900, 1600] {
            t.extend_from_slice(&v.to_be_bytes());
        }
        let table = parse_table(&t, false, 12).unwrap();
        assert_eq!(table.encoding, Encoding::Lossy(vec![0, 100, 400, 900, 1600]));
        let img = vec![0, 4, 1, 3, 2, 2, 4, 0];
        assert_eq!(decode(&encode(&img, 4, &LOSSY_12, table.seeds), 4, 2, 12, &table).unwrap(), vec![0, 1600, 100, 900, 400, 400, 1600, 0]);
    }

    #[test]
    fn unsupported_and_malformed_tables() {
        let unsupported = |r: Result<DecodeTable>| matches!(r, Err(RawError::Unsupported(_)));
        let malformed = |r: Result<DecodeTable>| matches!(r, Err(RawError::Malformed(_)));
        let mut split = lossy_table(12, false);
        split[562..564].copy_from_slice(&345u16.to_be_bytes());
        assert!(unsupported(parse_table(&split, false, 12)));
        assert!(unsupported(parse_table(&[0x49, 0x30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], false, 12)));
        assert!(unsupported(parse_table(&lossy_table(12, false), false, 16)));
        assert!(malformed(parse_table(&[], false, 12)));
        assert!(malformed(parse_table(&[0x46, 0x30, 8], false, 14)));
        assert!(malformed(parse_table(&lossy_table(12, false)[..100], false, 12)));
        let mut odd = lossy_table(12, false);
        odd[10..12].copy_from_slice(&300u16.to_be_bytes());
        assert!(parse_table(&odd, false, 12).is_err());
        assert!(parse_table(&[0x44, 0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], false, 12).is_err());
    }

    /// A 14-bit lossless table (version 0x46) with the given seeds.
    fn lossless_table(seed: u16, le: bool) -> Vec<u8> {
        let mut t = vec![0x46, 0x30];
        for _ in 0..4 {
            t.extend_from_slice(&if le { seed.to_le_bytes() } else { seed.to_be_bytes() });
        }
        t.resize(46, 0);
        t
    }

    #[test]
    fn decodes_through_the_nef_container() {
        use crate::testgen::{Val, nef_compressed};
        use crate::{Limits, RawFormat, identify};
        let (w, h) = (40usize, 6usize);
        // lossless 14-bit, big-endian maker note, with a BlackLevel tag
        let img = image(w, h, 16383, 11);
        let src = encode(&img, w, &LOSSLESS_14, [[2048; 2]; 2]);
        let note = vec![(0x0096, Val::Undefined(lossless_table(2048, false))), (0x003d, Val::Short(vec![600, 600, 600, 600]))];
        let b = nef_compressed(w, h, 14, src, [0, 1, 1, 2], true, note);
        assert_eq!(identify(&b), Some(RawFormat::Nef));
        let s = crate::decode(&b, &Limits::default()).unwrap();
        assert_eq!(s.data, img.iter().map(|&v| v as u16).collect::<Vec<_>>());
        assert_eq!(s.black.values, vec![600.0; 4]);
        // lossy 12-bit type 2, little-endian maker note (newer bodies); the
        // 12-bit BlackLevel is scaled from 14-bit units
        let table = lossy_table(12, true);
        let t = parse_table(&table, true, 12).unwrap();
        let Encoding::Lossy(curve) = &t.encoding else { panic!() };
        let img = image(w, h, 1023, 5);
        let src = encode(&img, w, &LOSSY_12, t.seeds);
        let note = vec![(0x0096, Val::Undefined(table.clone())), (0x003d, Val::Short(vec![600, 600, 600, 600]))];
        let s = crate::decode(&nef_compressed(w, h, 12, src.clone(), [2, 1, 1, 0], false, note), &Limits::default()).unwrap();
        assert_eq!(s.data, img.iter().map(|&v| curve[v as usize]).collect::<Vec<_>>());
        assert_eq!(s.black.values, vec![150.0; 4]);
        assert!(crate::develop_sensor(&s, &crate::DevelopOptions::default()).is_ok());
        // "lossy after split" and a missing table stay unsupported (preview fallback)
        let mut split = table;
        split[562..564].copy_from_slice(&3u16.to_le_bytes());
        let e = crate::decode(&nef_compressed(w, h, 12, src.clone(), [2, 1, 1, 0], false, vec![(0x0096, Val::Undefined(split))]), &Limits::default());
        assert!(matches!(e, Err(RawError::Unsupported(_))), "{e:?}");
        let e = crate::decode(&nef_compressed(w, h, 12, src, [2, 1, 1, 0], false, vec![(0x0001, Val::Short(vec![0]))]), &Limits::default());
        assert!(matches!(&e, Err(RawError::Unsupported(m)) if m.contains("Nikon compressed NEF")), "{e:?}");
    }

    #[test]
    fn nikon_without_black_level_tag_is_black_subtracted() {
        use crate::testgen::{Val, nef_compressed};
        let (w, h) = (32usize, 8usize);
        let img = image(w, h, 16383, 4);
        let src = encode(&img, w, &LOSSLESS_14, [[2048; 2]; 2]);
        let b = nef_compressed(w, h, 14, src, [0, 1, 1, 2], true, vec![(0x0096, Val::Undefined(lossless_table(2048, false)))]);
        let s = crate::decode(&b, &crate::Limits::default()).unwrap();
        assert_eq!(s.black.values, vec![0.0]);
        assert!(!s.warnings.iter().any(|w| w.contains("black level")), "{:?}", s.warnings);
    }

    #[test]
    fn truncated_container_never_panics() {
        use crate::testgen::{Val, nef_compressed};
        let (w, h) = (32usize, 8usize);
        let img = image(w, h, 16383, 2);
        let src = encode(&img, w, &LOSSLESS_14, [[2048; 2]; 2]);
        let b = nef_compressed(w, h, 14, src, [0, 1, 1, 2], true, vec![(0x0096, Val::Undefined(lossless_table(2048, false)))]);
        for n in 0..b.len() {
            let _ = crate::decode(&b[..n], &crate::Limits::default());
        }
    }

    #[test]
    fn truncated_and_corrupt_streams_error_never_panic() {
        let (w, h) = (64, 16);
        let table = lossless(14);
        let img = image(w, h, 16383, 3);
        let src = encode(&img, w, &LOSSLESS_14, table.seeds);
        for cut in [0, 1, 10, src.len() / 2, src.len() - 40] {
            assert!(decode(&src[..cut], w, h, 14, &table).is_err(), "cut {cut}");
        }
        let mut s = 12345u32;
        for i in 0..200 {
            let mut bad = src.clone();
            for _ in 0..1 + i % 8 {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let at = (s >> 8) as usize % bad.len();
                bad[at] ^= 1 << (s % 8);
            }
            let _ = decode(&bad, w, h, 14, &table);
            let noise: Vec<u8> = (0..src.len()).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> (i % 24)) as u8).collect();
            let _ = decode(&noise, w, h, 14, &table);
            let _ = decode(&noise, w, h, 12, &lossless(12));
        }
        let lossy = parse_table(&lossy_table(12, false), false, 12).unwrap();
        assert!(decode(&[0xff; 64], 8, 8, 12, &lossy).is_err());
        assert!(decode(&src, usize::MAX, 2, 14, &table).is_err());
        assert!(decode(&src, 0, 2, 14, &table).is_err());
    }
}
