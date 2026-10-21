//! Sony compressed ARW ("cRAW", ARW 2): a fixed-rate lossy code of one byte
//! per pixel, used by default on most Sony bodies.
//!
//! Implemented from the public descriptions of the format (H. Dietz, "Sony
//! ARW2 Compression: Artifacts And Credible Repair", Electronic Imaging 2016;
//! the RawDigger and diglloyd write-ups of the 11 + 7-bit scheme) and from
//! observation of sample files. No decoder source code was consulted.
//!
//! * Each row is cut into runs of 32 pixels, which the Bayer pattern makes
//!   two interleaved runs of 16 same-colour pixels; the run of even columns is
//!   stored first, then the run of odd columns, 16 bytes each.
//! * A 16-byte block, read as a little-endian 128-bit number from the least
//!   significant bit, holds an 11-bit maximum, an 11-bit minimum, the 4-bit
//!   positions of the maximum and of the minimum, and fourteen 7-bit deltas
//!   for the other positions in order.
//! * The step of the deltas is the smallest power of two `s` with
//!   `128 * s > max - min`; a pixel is `min + delta * s`.
//! * The 11-bit codes go through a five-segment linear tone curve whose step
//!   doubles at each knot (1, 2, 4, 8, 16). The knots are the four values of
//!   the SonyToneCurve tag (0x7010), on a scale of eight times the code; the
//!   output is 14-bit. Observed in sample files: the 512 black level maps to
//!   code 256 and the clipping code maps to the 16383 white level.

use crate::error::{RawError, Result};
use crate::sensor::{Plane, segments};
use crate::tiff::{Ifd, Tiff, tag};
use crate::{Limits, par};

/// SonyRawFileType (ExifTool): 2 = compressed RAW.
pub(crate) const SONY_RAW_FILE_TYPE: u16 = 0x7000;
/// SonyToneCurve: the four knots of the cRAW tone curve.
pub(crate) const SONY_TONE_CURVE: u16 = 0x7010;
const MAX_14: u32 = 16383;

/// The 2048-entry code → 14-bit value table of a tone curve given by its four
/// knots (SonyToneCurve), or `None` when the knots are not increasing or
/// out of range.
pub(crate) fn tone_lut(knots: &[u32]) -> Option<Vec<u16>> {
    let [a, b, c, d] = <[u32; 4]>::try_from(knots).ok()?;
    let pts = [0, a, b, c, d, 1 << 14];
    if pts.windows(2).any(|w| w[0] > w[1]) {
        return None;
    }
    let lut = (0..2048u32)
        .map(|code| {
            // On the knots' scale a code is 8 × code; the first segment maps it
            // to 2 × code, i.e. a quarter of the scaled position.
            let x = code * 8;
            let mut y = 0u32;
            for (seg, w) in pts.windows(2).enumerate() {
                if x > w[0] {
                    y += (x.min(w[1]) - w[0]) << seg;
                }
            }
            (y / 4).min(MAX_14) as u16
        })
        .collect();
    Some(lut)
}

/// Decodes one 16-byte block into 16 codes.
#[inline]
pub(crate) fn decode_block(b: &[u8; 16]) -> [u16; 16] {
    let v = u128::from_le_bytes(*b);
    let field = |at: u32, bits: u32| ((v >> at) & ((1u128 << bits) - 1)) as u16;
    let max = field(0, 11);
    let min = field(11, 11);
    let imax = usize::from(field(22, 4));
    let imin = usize::from(field(26, 4));
    let range = max.saturating_sub(min);
    let mut sh = 0u32;
    while (128u16 << sh) <= range {
        sh += 1;
    }
    let mut out = [0u16; 16];
    let mut at = 30u32;
    for (i, o) in out.iter_mut().enumerate() {
        *o = if i == imax {
            max
        } else if i == imin {
            min
        } else if at + 7 <= 128 {
            let d = field(at, 7);
            at += 7;
            (min + (d << sh)).min(2047)
        } else {
            // Only when a damaged block names the same position twice.
            min
        };
    }
    out
}

/// `true` when the raw IFD holds cRAW data: Sony's private compression with
/// one byte per pixel.
pub(crate) fn is_craw(t: &Tiff, ifd: &Ifd) -> bool {
    if t.tag_uint(ifd, tag::COMPRESSION) != Some(32767) {
        return false;
    }
    let (w, h) = (t.tag_uint(ifd, tag::IMAGE_WIDTH).unwrap_or(0), t.tag_uint(ifd, tag::IMAGE_LENGTH).unwrap_or(0));
    let bytes: u64 = t.tag_uints(ifd, tag::STRIP_BYTE_COUNTS).iter().map(|&c| u64::from(c)).sum();
    match t.tag_uint(ifd, SONY_RAW_FILE_TYPE) {
        Some(kind) => kind == 2,
        None => bytes > 0 && bytes == u64::from(w) * u64::from(h),
    }
}

/// Reads the cRAW image of a raw IFD.
pub(crate) fn read_craw(t: &Tiff, ifd: &Ifd, limits: &Limits) -> Result<Plane> {
    let width = t.tag_uint(ifd, tag::IMAGE_WIDTH).ok_or_else(|| RawError::malformed("raw image has no width"))? as usize;
    let height = t.tag_uint(ifd, tag::IMAGE_LENGTH).ok_or_else(|| RawError::malformed("raw image has no height"))? as usize;
    if !width.is_multiple_of(32) {
        return Err(RawError::unsupported(format!("Sony compressed ARW {width} pixels wide (not a multiple of 32)")));
    }
    let lut = tone_lut(&t.tag_uints(ifd, SONY_TONE_CURVE)).ok_or_else(|| RawError::unsupported("Sony compressed ARW without a usable tone curve (0x7010)"))?;
    limits.check(width as u64, height as u64, 2)?;
    // One byte per pixel: locate every row in the strips.
    let mut rows: Vec<&[u8]> = Vec::with_capacity(height);
    for s in segments(t, ifd, width, height)? {
        let n = s.h.min(height.saturating_sub(s.y));
        for r in 0..n {
            let at = s.offset.checked_add(r * width).ok_or_else(|| RawError::malformed("raw strip offset overflows"))?;
            if (r + 1) * width > s.len {
                return Err(RawError::malformed("compressed ARW strip is shorter than its rows"));
            }
            rows.push(t.bytes(at, width).ok_or_else(|| RawError::malformed("compressed ARW data lies outside the file"))?);
        }
    }
    if rows.len() != height {
        return Err(RawError::malformed("compressed ARW strips do not cover the image"));
    }
    let mut data = vec![0u16; width * height];
    let band = par::band_rows(width);
    par::chunks_mut(&mut data, band * width, |i, chunk| {
        for (r, out) in chunk.chunks_exact_mut(width).enumerate() {
            let Some(src) = rows.get(i * band + r) else { return };
            for (group, px) in src.as_chunks::<32>().0.iter().zip(out.as_chunks_mut::<32>().0) {
                let (Ok(even), Ok(odd)) = (<&[u8; 16]>::try_from(&group[..16]), <&[u8; 16]>::try_from(&group[16..])) else { continue };
                for (k, (e, o)) in decode_block(even).into_iter().zip(decode_block(odd)).enumerate() {
                    px[2 * k] = lut.get(usize::from(e)).copied().unwrap_or(0);
                    px[2 * k + 1] = lut.get(usize::from(o)).copied().unwrap_or(0);
                }
            }
        }
    });
    Ok(Plane { width, height, samples: 1, bits: 14, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_curve_matches_observed_anchors() {
        // Knots written by an ILCE-7M3.
        let lut = tone_lut(&[8000, 10400, 12900, 14100]).unwrap();
        assert_eq!(lut[0], 0);
        assert_eq!(lut[256], 512); // black
        assert_eq!(lut[999], 1998);
        // Steps double at each knot.
        assert_eq!(lut[1001] - lut[1000], 2 * (lut[1000] - lut[999]));
        assert!(lut[2021] >= 16300, "{}", lut[2021]); // clipping code ≈ white
        assert_eq!(lut[2047], 16383);
        assert!(lut.windows(2).all(|w| w[0] <= w[1]));
        assert!(tone_lut(&[9000, 8000, 12000, 13000]).is_none());
        assert!(tone_lut(&[1, 2, 3]).is_none());
        assert!(tone_lut(&[0, 0, 0, 20000]).is_none());
    }

    #[test]
    fn hostile_blocks_do_not_panic() {
        for b in [[0xFFu8; 16], [0u8; 16], [0x55; 16]] {
            let _ = decode_block(&b);
        }
    }
}
