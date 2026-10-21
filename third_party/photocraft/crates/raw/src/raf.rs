//! Fujifilm RAF.
//!
//! A RAF file starts with a big-endian header: the magic
//! `FUJIFILMCCD-RAW `, a format version, a camera id and the model name (32
//! bytes at 28), then (offset, length) pairs for the camera's JPEG preview (at
//! 84), a metadata directory (at 92) and the CFA region (at 100).
//!
//! The metadata directory is a big-endian record count followed by (tag,
//! size, data) records. The tags used here are documented in ExifTool's
//! FujiFilm RAF table: RawImageCropTopLeft (0x110: top, left),
//! RawImageCroppedSize (0x111: height, width), FujiLayout (0x130), XTransLayout (0x131), WB_GRGBLevels (0x2FF0)
//! and RawExposureBias (0x9650: a signed 16-bit fraction).
//!
//! In every body since about 2010 the CFA region is a small TIFF whose IFD0
//! points (tag 0xF000) at a Fuji IFD (ExifTool's FujiIFD table):
//! RawImageFullWidth / Height (0xF001 / 0xF002), BitsPerSample (0xF003),
//! StripOffsets / StripByteCounts (0xF007 / 0xF008, relative to the CFA
//! region), BlackLevel (0xF00A) and WB_GRBLevels (0xF00E: G, R, B). Offsets in
//! it are relative to the CFA region.
//!
//! The rest was established by observation of sample files (raw.pixls.us
//! CC0 samples of the X-Pro1, X-E2, X100, XF1, X-A5 and GFX 50S, and an
//! X-T30 III file):
//!
//! * Sample storage, told apart by the strip's byte count: 16-bit
//!   containers in the CFA TIFF's byte order (14- and 16-bit bodies: X-Trans
//!   II to V, GFX); 12-bit samples packed into a little-endian bit stream,
//!   least significant bit first (X-Trans I, X100, XF1, X-A1…); 14-bit
//!   samples packed into little-endian 32-bit words, most significant bit
//!   first (X-A3/5/7, X-T100/200, XF10). A shorter strip is Fujifilm's own
//!   lossless compression (it starts with `IS`), which has no public
//!   description and is reported as unsupported, as are the pre-2010
//!   FinePix / SuperCCD files without the CFA TIFF.
//! * XTransLayout lists the 6×6 X-Trans pattern (0 = R, 1 = G, 2 = B) in
//!   reverse order: its last byte is the colour at data (0, 0), its first
//!   the colour at (5, 5).
//! * On Bayer bodies FujiLayout lists the 2×2 pattern the same way, reversed,
//!   with the colour in the low two bits (0 = R, 1 and 3 = G, 2 = B):
//!   `0A 0B 09 08` is RGGB and `08 09 0B 0A` is BGGR at data (0, 0).
//! * The crop records address the data as stored; the preview JPEG's EXIF
//!   orientation applies to the raw image too.
//!
//! Both layouts were verified per sample by developing with every candidate
//! phase and comparing with the camera's preview.

use crate::cr2::clip_level;
use crate::error::{RawError, Result};
use crate::sensor::{BlackLevels, Cfa, Rect, Sensor};
use crate::tiff::{Tiff, tag};
use crate::{Limits, RawFormat, par};

pub(crate) const MAGIC: &[u8] = b"FUJIFILMCCD-RAW";

const CROP_TOP_LEFT: u16 = 0x0110;
const CROPPED_SIZE: u16 = 0x0111;
const FUJI_LAYOUT: u16 = 0x0130;
const XTRANS_LAYOUT: u16 = 0x0131;
const WB_GRGB_LEVELS: u16 = 0x2FF0;
const RAW_EXPOSURE_BIAS: u16 = 0x9650;

const FUJI_IFD: u16 = 0xF000;
const RAW_WIDTH: u16 = 0xF001;
const RAW_HEIGHT: u16 = 0xF002;
const RAW_BITS: u16 = 0xF003;
const STRIP_OFFSET: u16 = 0xF007;
const STRIP_BYTE_COUNT: u16 = 0xF008;
const BLACK_LEVEL: u16 = 0xF00A;
const WB_GRB_LEVELS: u16 = 0xF00E;

/// Most metadata records read.
const MAX_RECORDS: usize = 512;

fn be32(b: &[u8], at: usize) -> Option<usize> {
    let s: [u8; 4] = b.get(at..at.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_be_bytes(s) as usize)
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    let s: [u8; 2] = b.get(at..at.checked_add(2)?)?.try_into().ok()?;
    Some(u16::from_be_bytes(s))
}

/// Offset and length of the preview JPEG (header bytes 84 and 88).
pub(crate) fn preview_range(bytes: &[u8]) -> Option<(usize, usize)> {
    Some((be32(bytes, 84)?, be32(bytes, 88)?))
}

/// The metadata directory's records as (tag, data).
fn records(bytes: &[u8]) -> Vec<(u16, &[u8])> {
    let mut out = Vec::new();
    let (Some(at), Some(len)) = (be32(bytes, 92), be32(bytes, 96)) else { return out };
    let Some(dir) = at.checked_add(len).and_then(|end| bytes.get(at..end)) else { return out };
    let n = be32(dir, 0).unwrap_or(0).min(MAX_RECORDS);
    let mut p = 4usize;
    for _ in 0..n {
        let (Some(tag), Some(size)) = (be16(dir, p), be16(dir, p + 2)) else { break };
        let start = p + 4;
        let Some(data) = dir.get(start..start + usize::from(size)) else { break };
        out.push((tag, data));
        p = start + usize::from(size);
    }
    out
}

fn record<'a>(recs: &[(u16, &'a [u8])], tag: u16) -> Option<&'a [u8]> {
    recs.iter().find(|(t, _)| *t == tag).map(|(_, d)| *d)
}

/// A record holding two big-endian 16-bit values.
fn pair(recs: &[(u16, &[u8])], tag: u16) -> Option<(usize, usize)> {
    let d = record(recs, tag)?;
    Some((usize::from(be16(d, 0)?), usize::from(be16(d, 2)?)))
}

/// The CFA from XTransLayout (6×6) or FujiLayout (2×2), both stored reversed.
fn layout_cfa(recs: &[(u16, &[u8])]) -> Option<Cfa> {
    if let Some(d) = record(recs, XTRANS_LAYOUT).filter(|d| d.len() == 36) {
        let colors = d.iter().rev().map(|&c| (c <= 2).then_some(c)).collect::<Option<Vec<u8>>>()?;
        return Some(Cfa { width: 6, height: 6, colors, origin_x: 0, origin_y: 0 });
    }
    let d = record(recs, FUJI_LAYOUT).filter(|d| d.len() == 4)?;
    let colors: Vec<u8> = d.iter().rev().map(|&c| if c & 3 == 3 { 1 } else { c & 3 }).collect();
    let cfa = Cfa { width: 2, height: 2, colors, origin_x: 0, origin_y: 0 };
    cfa.is_bayer().then_some(cfa)
}

/// How the samples of the strip are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Packing {
    /// 16-bit containers in the given byte order.
    U16 { le: bool },
    /// Little-endian bit stream, least significant bit first (12-bit bodies).
    LsbFirst,
    /// Little-endian 32-bit words, most significant bit first (14-bit bodies).
    Words32MsbFirst,
}

/// Reads `bits`-bit (≤ 16) sample `i` of a packed stream (`src` holds the
/// whole stream; bytes past its end read as 0).
#[inline]
fn packed_sample(src: &[u8], i: usize, bits: u32, packing: Packing) -> u16 {
    let at = i * bits as usize;
    let (byte, shift) = (at / 8, (at % 8) as u32);
    let mask = (1u32 << bits.min(16)) - 1;
    match packing {
        Packing::LsbFirst => {
            let w = (0..4).fold(0u32, |w, k| w | u32::from(src.get(byte + k).copied().unwrap_or(0)) << (8 * k));
            ((w >> shift) & mask) as u16
        }
        Packing::Words32MsbFirst => {
            // Logical byte k is physical byte k with its position in the 32-bit word reversed.
            let phys = |k: usize| src.get((k & !3) | (3 - (k & 3))).copied().unwrap_or(0);
            let w = (0..4).fold(0u32, |w, k| (w << 8) | u32::from(phys(byte + k)));
            ((w >> (32 - shift - bits.min(16))) & mask) as u16
        }
        Packing::U16 { le } => {
            let b = [src.get(2 * i).copied().unwrap_or(0), src.get(2 * i + 1).copied().unwrap_or(0)];
            if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) }
        }
    }
}

/// Unpacks a `width × height` strip.
pub(crate) fn unpack(src: &[u8], width: usize, height: usize, bits: u32, packing: Packing) -> Vec<u16> {
    let mut data = vec![0u16; width * height];
    let band = par::band_rows(width);
    par::chunks_mut(&mut data, band * width, |b, chunk| {
        let first = b * band * width;
        for (k, v) in chunk.iter_mut().enumerate() {
            *v = packed_sample(src, first + k, bits, packing);
        }
    });
    data
}

/// Orientation, make and model from the preview JPEG's EXIF (APP1) block.
fn preview_exif(bytes: &[u8]) -> (Option<u16>, Option<String>, Option<String>) {
    let none = (None, None, None);
    let Some((off, len)) = preview_range(bytes) else { return none };
    let Some(jpeg) = off.checked_add(len).and_then(|end| bytes.get(off..end)) else { return none };
    if jpeg.get(0..4) != Some(&[0xFF, 0xD8, 0xFF, 0xE1]) || jpeg.get(6..12) != Some(b"Exif\0\0") {
        return none;
    }
    let Some(t) = jpeg.get(12..).and_then(Tiff::new) else { return none };
    let Some(ifd0) = t.ifd_at(t.first_ifd, 0) else { return none };
    let orientation = t.tag_uint(&ifd0, tag::ORIENTATION).and_then(|o| u16::try_from(o).ok()).filter(|o| (1..=8).contains(o));
    (orientation, t.tag_ascii(&ifd0, tag::MAKE), t.tag_ascii(&ifd0, tag::MODEL))
}

/// The model name in the RAF header (bytes 28..60).
fn header_model(bytes: &[u8]) -> Option<String> {
    let b = bytes.get(28..60)?;
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    let s = String::from_utf8_lossy(&b[..end]).trim().to_string();
    (!s.is_empty()).then_some(s)
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Sensor> {
    let cfa_at = be32(bytes, 100).ok_or_else(|| RawError::malformed("RAF header is truncated"))?;
    let cfa_len = be32(bytes, 104).unwrap_or(0);
    let region = bytes.get(cfa_at..).ok_or_else(|| RawError::malformed("RAF CFA region lies outside the file"))?;
    let region = if cfa_len > 0 { region.get(..cfa_len).unwrap_or(region) } else { region };
    let Some(t) = Tiff::new(region) else {
        return Err(RawError::unsupported("Fujifilm RAF of an early FinePix / SuperCCD body (no CFA TIFF) is not decoded yet"));
    };
    let ifd0 = t.ifd_at(t.first_ifd, 0).ok_or_else(|| RawError::malformed("RAF CFA TIFF has no IFD0"))?;
    let fuji_at = t.tag_uint(&ifd0, FUJI_IFD).ok_or_else(|| RawError::unsupported("RAF without a Fuji raw IFD"))?;
    let fuji = t.ifd_at(fuji_at as usize, 0).ok_or_else(|| RawError::malformed("RAF Fuji raw IFD is unreadable"))?;

    let width = t.tag_uint(&fuji, RAW_WIDTH).ok_or_else(|| RawError::malformed("RAF has no raw width"))? as usize;
    let height = t.tag_uint(&fuji, RAW_HEIGHT).ok_or_else(|| RawError::malformed("RAF has no raw height"))? as usize;
    let bits = t.tag_uint(&fuji, RAW_BITS).unwrap_or(14);
    if !(8..=16).contains(&bits) {
        return Err(RawError::unsupported(format!("RAF with {bits}-bit samples")));
    }
    limits.check(width as u64, height as u64, 2)?;
    let offset = t.tag_uint(&fuji, STRIP_OFFSET).ok_or_else(|| RawError::malformed("RAF has no strip offset"))? as usize;
    let count = t.tag_uint(&fuji, STRIP_BYTE_COUNT).unwrap_or(0) as usize;
    let src = region.get(offset..).ok_or_else(|| RawError::malformed("RAF raw data lies outside the file"))?;
    let src = if count > 0 { src.get(..count).unwrap_or(src) } else { src };
    let pixels = width * height;
    let packed = pixels.saturating_mul(bits as usize).div_ceil(8);
    let packing = if count >= pixels * 2 {
        Packing::U16 { le: t.le }
    } else if count >= packed && bits == 12 {
        Packing::LsbFirst
    } else if count >= packed && bits == 14 {
        Packing::Words32MsbFirst
    } else if count >= packed {
        return Err(RawError::unsupported(format!("Fujifilm RAF with packed {bits}-bit samples")));
    } else {
        return Err(RawError::unsupported("Fujifilm compressed RAF is not decoded (no public description of the format)"));
    };
    let need = if matches!(packing, Packing::U16 { .. }) { pixels * 2 } else { packed };
    if src.len() < need {
        return Err(RawError::malformed("RAF raw data is truncated"));
    }
    let data = unpack(src, width, height, bits, packing);

    let recs = records(bytes);
    let cfa = layout_cfa(&recs).ok_or_else(|| RawError::unsupported("Fujifilm RAF without a recognised CFA layout"))?;
    let full = Rect::new(0, 0, width, height);
    let mut warnings = Vec::new();

    let black = t.tag_floats(&fuji, BLACK_LEVEL);
    let black = if black.is_empty() || black.iter().any(|v| *v < 0.0) {
        warnings.push("RAF: black level not recorded; assumed 0".to_string());
        BlackLevels::uniform(0.0)
    } else {
        // One value per CFA position; the order is not established, and the
        // values observed differ by at most a couple of units, so use their mean.
        BlackLevels::uniform((black.iter().sum::<f64>() / black.len() as f64) as f32)
    };
    let white = clip_level(&data, bits);

    let camera_wb = match t.tag_floats(&fuji, WB_GRB_LEVELS).as_slice() {
        [g, r, b, ..] if *g > 0.0 => Some([r / g, 1.0, b / g]),
        _ => record(&recs, WB_GRGB_LEVELS).and_then(|d| {
            let v: Vec<f64> = (0..4).map_while(|k| be16(d, 2 * k).map(f64::from)).collect();
            match v.as_slice() {
                [g1, r, g2, b] if g1 + g2 > 0.0 => Some([2.0 * r / (g1 + g2), 1.0, 2.0 * b / (g1 + g2)]),
                _ => None,
            }
        }),
    }
    .filter(|m| m.iter().all(|v| (0.25..8.0).contains(v)));

    let crop = match (pair(&recs, CROP_TOP_LEFT), pair(&recs, CROPPED_SIZE)) {
        (Some((top, left)), Some((h, w))) => Rect::new(left, top, w, h).intersect(&full),
        _ => full,
    };
    let crop = if crop.is_empty() { full } else { crop };

    // RawExposureBias: how far the raw was exposed below the camera's
    // rendering (dynamic-range modes); undone like a DNG BaselineExposure.
    let baseline_exposure = record(&recs, RAW_EXPOSURE_BIAS)
        .and_then(|d| {
            let (n, den) = (be16(d, 0)? as i16, be16(d, 2)? as i16);
            (den != 0).then(|| -f64::from(n) / f64::from(den))
        })
        .filter(|v| v.is_finite() && v.abs() <= 8.0)
        .unwrap_or(0.0);

    let (orientation, make, model) = preview_exif(bytes);
    Ok(Sensor {
        format: RawFormat::Raf,
        make: make.or_else(|| Some("FUJIFILM".to_string())),
        model: model.or_else(|| header_model(bytes)),
        width,
        height,
        samples: 1,
        data,
        cfa: Some(cfa),
        linearization: None,
        black,
        white: [white; 3],
        active: full,
        crop,
        color: Default::default(),
        camera_wb,
        orientation: orientation.unwrap_or(1),
        baseline_exposure,
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
    fn lsb_first_12_bit() {
        // 0xABC, 0x123 LSB first: byte0 = 0xBC, byte1 = 0x3A, byte2 = 0x12.
        let src = [0xBC, 0x3A, 0x12];
        assert_eq!(unpack(&src, 2, 1, 12, Packing::LsbFirst), vec![0xABC, 0x123]);
    }

    #[test]
    fn words_32_msb_first_14_bit() {
        // Four 14-bit samples MSB first in the logical stream, stored as
        // little-endian 32-bit words.
        let vals: [u16; 4] = [0x3FFF, 0x0001, 0x2AAA, 0x1555];
        let mut acc: u64 = 0;
        for v in vals {
            acc = (acc << 14) | u64::from(v);
        }
        let logical: Vec<u8> = (0..7).map(|k| (acc >> (8 * (6 - k))) as u8).chain([0]).collect();
        let phys: Vec<u8> = logical.chunks(4).flat_map(|w| w.iter().rev().copied().collect::<Vec<_>>()).collect();
        assert_eq!(unpack(&phys, 4, 1, 14, Packing::Words32MsbFirst), vals.to_vec());
        // A short buffer reads as zeros rather than panicking.
        assert_eq!(unpack(&phys[..3], 4, 1, 14, Packing::Words32MsbFirst).len(), 4);
    }

    #[test]
    fn layouts_are_reversed() {
        let fuji = |v: &[u8]| layout_cfa(&[(FUJI_LAYOUT, v)]).map(|c| c.phase(0, 0));
        assert_eq!(fuji(&[0x0A, 0x0B, 0x09, 0x08]), Some([0, 1, 1, 2]));
        assert_eq!(fuji(&[0x08, 0x09, 0x0B, 0x0A]), Some([2, 1, 1, 0]));
        assert_eq!(fuji(&[0x0C, 0x0C, 0x0C, 0x0C]), None);
        let xt: Vec<u8> = (0..36).map(|i| [0, 1, 2][i % 3]).collect();
        let c = layout_cfa(&[(FUJI_LAYOUT, &[0x0C; 4]), (XTRANS_LAYOUT, &xt)]).unwrap();
        assert_eq!((c.width, c.height), (6, 6));
        assert_eq!(c.color(0, 0), xt[35]);
        assert_eq!(c.color(5, 5), xt[0]);
        assert!(layout_cfa(&[(XTRANS_LAYOUT, &[3; 36])]).is_none());
    }
}
