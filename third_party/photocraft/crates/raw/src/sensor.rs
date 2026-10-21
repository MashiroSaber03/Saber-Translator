//! Undeveloped sensor data and the shared strip/tile reader.

use crate::color::ColorInfo;
use crate::error::{RawError, Result};
use crate::tiff::{Ifd, Tiff, tag};
use crate::{Limits, RawFormat, jxl, ljpeg, par};

/// A rectangle in sensor-data pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Rect { x, y, width, height }
    }

    /// The overlap of two rectangles (may be empty).
    pub fn intersect(&self, o: &Rect) -> Rect {
        let x0 = self.x.max(o.x);
        let y0 = self.y.max(o.y);
        let x1 = (self.x.saturating_add(self.width)).min(o.x.saturating_add(o.width));
        let y1 = (self.y.saturating_add(self.height)).min(o.y.saturating_add(o.height));
        Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Colour of one CFA cell: 0 = red, 1 = green, 2 = blue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cfa {
    pub width: usize,
    pub height: usize,
    /// Row-major colours, `width * height` entries.
    pub colors: Vec<u8>,
    /// Data coordinates of the pattern's top-left cell.
    pub origin_x: usize,
    pub origin_y: usize,
}

impl Cfa {
    /// A 2×2 pattern from a string such as `"RGGB"`, anchored at (0, 0).
    pub fn bayer(pattern: &str) -> Option<Cfa> {
        let colors: Vec<u8> = pattern
            .chars()
            .map(|c| match c {
                'R' => Some(0),
                'G' => Some(1),
                'B' => Some(2),
                _ => None,
            })
            .collect::<Option<_>>()?;
        (colors.len() == 4).then_some(Cfa { width: 2, height: 2, colors, origin_x: 0, origin_y: 0 })
    }

    /// Colour at data coordinates.
    #[inline]
    pub fn color(&self, x: usize, y: usize) -> u8 {
        fn relative(v: usize, origin: usize, period: usize) -> Option<usize> {
            if period == 0 {
                return None;
            }
            let (v, origin) = (v % period, origin % period);
            Some(if v >= origin { v - origin } else { period - (origin - v) })
        }
        let Some((cx, cy)) = relative(x, self.origin_x, self.width).zip(relative(y, self.origin_y, self.height)) else {
            return 1;
        };
        cy.checked_mul(self.width).and_then(|i| i.checked_add(cx)).and_then(|i| self.colors.get(i)).copied().unwrap_or(1)
    }

    /// The 2×2 colours seen from data coordinates (`x0`, `y0`), row-major.
    pub fn phase(&self, x0: usize, y0: usize) -> [u8; 4] {
        [self.color(x0, y0), self.color(x0.saturating_add(1), y0), self.color(x0, y0.saturating_add(1)), self.color(x0.saturating_add(1), y0.saturating_add(1))]
    }

    /// `true` for a Bayer pattern: a 2×2 period holding one red, one blue and two
    /// diagonal greens.
    pub fn is_bayer(&self) -> bool {
        if self.width == 0 || self.height == 0 || self.width > 16 || self.height > 16 {
            return false;
        }
        let (ox, oy) = (self.origin_x % self.width, self.origin_y % self.height);
        let p = self.phase(ox, oy);
        let periodic = (0..self.height).all(|y| (0..self.width).all(|x| self.color(ox + x, oy + y) == p[(y % 2) * 2 + x % 2]));
        let greens_on_diagonal = (p[0] == 1 && p[3] == 1 && p[1] != 1 && p[2] != 1 && u16::from(p[1]) + u16::from(p[2]) == 2)
            || (p[1] == 1 && p[2] == 1 && p[0] != 1 && p[3] != 1 && u16::from(p[0]) + u16::from(p[3]) == 2);
        periodic && greens_on_diagonal
    }
}

/// Black levels (DNG BlackLevel / BlackLevelRepeatDim / BlackLevelDeltaH/V).
#[derive(Debug, Clone, PartialEq)]
pub struct BlackLevels {
    pub rows: usize,
    pub cols: usize,
    /// `rows * cols * samples` values, sample fastest.
    pub values: Vec<f32>,
    /// Per column of the active area.
    pub delta_h: Vec<f32>,
    /// Per row of the active area.
    pub delta_v: Vec<f32>,
}

impl BlackLevels {
    pub fn uniform(v: f32) -> Self {
        BlackLevels { rows: 1, cols: 1, values: vec![v], delta_h: Vec::new(), delta_v: Vec::new() }
    }

    /// Black level at active-area coordinates for sample `s` of `spp`.
    #[inline]
    pub fn at(&self, ax: usize, ay: usize, s: usize, spp: usize) -> f32 {
        let r = ay % self.rows.max(1);
        let c = ax % self.cols.max(1);
        let index = r.checked_mul(self.cols.max(1)).and_then(|i| i.checked_add(c)).and_then(|i| i.checked_mul(spp)).and_then(|i| i.checked_add(s));
        let base = index.and_then(|i| self.values.get(i)).or_else(|| self.values.first()).copied().unwrap_or(0.0);
        base + self.delta_h.get(ax).copied().unwrap_or(0.0) + self.delta_v.get(ay).copied().unwrap_or(0.0)
    }

    /// `true` when there is a single black value for everything.
    pub fn is_uniform(&self) -> bool {
        self.delta_h.iter().all(|v| *v == 0.0) && self.delta_v.iter().all(|v| *v == 0.0) && self.values.windows(2).all(|w| w[0] == w[1])
    }
}

/// Undeveloped sensor data with everything needed to develop it.
#[derive(Debug, Clone)]
pub struct Sensor {
    pub format: RawFormat,
    pub make: Option<String>,
    pub model: Option<String>,
    /// Data width and height in pixels.
    pub width: usize,
    pub height: usize,
    /// Samples per pixel: 1 for CFA data, 3 for LinearRaw (1 for monochrome LinearRaw).
    pub samples: usize,
    /// Row-major interleaved samples, `width * height * samples`.
    pub data: Vec<u16>,
    /// The CFA pattern (CFA data only).
    pub cfa: Option<Cfa>,
    /// Maps stored values to linear values (DNG LinearizationTable).
    pub linearization: Option<Vec<u16>>,
    pub black: BlackLevels,
    /// White (saturation) level per sample.
    pub white: [f32; 3],
    /// Sensor area holding image data (DNG ActiveArea); black-level deltas are indexed from it.
    pub active: Rect,
    /// The area to output (ActiveArea ∩ DefaultCrop).
    pub crop: Rect,
    pub color: ColorInfo,
    /// White-balance multipliers from the camera's own metadata (non-DNG), R G B.
    pub camera_wb: Option<[f64; 3]>,
    /// TIFF orientation (1–8).
    pub orientation: u16,
    /// Exposure compensation baked into the default rendering (DNG BaselineExposure), in EV.
    pub baseline_exposure: f64,
    /// DNG OpcodeList2 gain maps (lens shading), applied to the normalized data.
    pub gain_maps: Vec<crate::opcodes::GainMap>,
    /// The camera profile's default tone curve (like DNG ProfileToneCurve): increasing `[input,
    /// output]` pairs in linear ProPhoto, applied per channel after the colour conversion and
    /// exposure. Below the first point the curve is a line through 0; empty: no curve.
    pub tone_curve: Vec<[f32; 2]>,
    /// Camera Raw's adaptive black point, for profiles that copy its default rendering: this
    /// factor times the 0.1th percentile of the image's linear luminance is subtracted before the
    /// tone curve (see `tone.rs`). 0: no black point.
    pub black_point: f32,
    pub warnings: Vec<String>,
}

impl Sensor {
    /// `true` for monochrome data (one-sample LinearRaw: monochrome cameras, scanner DNGs), which
    /// develops to neutral grey with no white balance or colour calibration.
    pub fn is_monochrome(&self) -> bool {
        self.samples == 1 && self.cfa.is_none()
    }
}

/// Raw samples read from one IFD.
pub(crate) struct Plane {
    pub width: usize,
    pub height: usize,
    pub samples: usize,
    pub bits: u32,
    pub data: Vec<u16>,
}

pub(crate) struct Segment {
    pub x: usize,
    pub y: usize,
    /// Encoded width and height (tile size, or image width × rows per strip).
    pub w: usize,
    pub h: usize,
    pub offset: usize,
    pub len: usize,
}

/// How a lossless-JPEG tile's samples map onto the tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JpegLayout {
    /// The decoded sample sequence fills the tile row by row (DNG).
    Flat,
    /// A 4-component frame of half the tile's width and height: each JPEG
    /// pixel holds a 2×2 block, row-major (observed in Sony lossless ARW).
    Quads,
}

/// DNG 1.7 JPEG XL compression.
const JPEG_XL: u32 = 52546;

/// Reads the (uncompressed, lossless-JPEG or JPEG XL) image of a raw IFD.
pub(crate) fn read_plane(t: &Tiff, ifd: &Ifd, limits: &Limits, layout: JpegLayout) -> Result<Plane> {
    let width = t.tag_uint(ifd, tag::IMAGE_WIDTH).ok_or_else(|| RawError::malformed("raw image has no width"))? as usize;
    let height = t.tag_uint(ifd, tag::IMAGE_LENGTH).ok_or_else(|| RawError::malformed("raw image has no height"))? as usize;
    let samples = t.tag_uint(ifd, tag::SAMPLES_PER_PIXEL).unwrap_or(1) as usize;
    let bits = t.tag_uint(ifd, tag::BITS_PER_SAMPLE).unwrap_or(1);
    let compression = t.tag_uint(ifd, tag::COMPRESSION).unwrap_or(1);
    if !(1..=4).contains(&samples) {
        return Err(RawError::unsupported(format!("{samples} samples per pixel")));
    }
    if t.tag_uint(ifd, tag::SAMPLE_FORMAT) == Some(3) {
        return Err(RawError::unsupported("floating-point raw data"));
    }
    if !(1..=16).contains(&bits) {
        return Err(RawError::unsupported(format!("{bits}-bit raw samples")));
    }
    if samples > 1 && t.tag_uint(ifd, tag::PLANAR_CONFIGURATION) == Some(2) {
        return Err(RawError::unsupported("planar raw data"));
    }
    match compression {
        1 | 7 | 8 | 34892 | JPEG_XL => {}
        34713 => return Err(RawError::unsupported("Nikon compressed NEF")),
        32767 => return Err(RawError::unsupported("Sony compressed ARW")),
        65535 => return Err(RawError::unsupported("Pentax compressed PEF")),
        c => return Err(RawError::unsupported(format!("raw compression scheme {c}"))),
    }
    limits.check(width as u64, height as u64, samples as u64 * 2)?;
    let segments = segments(t, ifd, width, height)?;
    let le = t.le;
    let decoded: Vec<Result<Vec<u16>>> = par::map(segments.len(), |i| {
        let s = &segments[i];
        let rows = s.h.min(height - s.y);
        let src = t.bytes(s.offset, s.len).ok_or_else(|| RawError::malformed("raw data lies outside the file"))?;
        // Every branch below sizes at least one buffer from the segment's own geometry, which
        // the file declares independently of the image: a tile can be far larger than the image
        // it belongs to, so the image check above does not bound those buffers. Charge the
        // segment itself before decoding it (#2925).
        limits.check(s.w as u64, s.h as u64, samples as u64 * 2)?;
        match compression {
            1 => unpack(src, s.w, rows, samples, bits, le),
            8 => {
                // Adobe DNG compression 8: zlib streams (RFC 1950). The output is capped at the
                // tile's size in whole bytes per sample, so a bomb cannot grow past the tile;
                // `unpack` then reads it bit-packed or in 16-bit containers by its length.
                let too_big = || RawError::malformed("deflate tile is too large");
                let need =
                    s.w.checked_mul(rows).and_then(|n| n.checked_mul(samples)).and_then(|n| n.checked_mul((bits as usize).div_ceil(8))).ok_or_else(too_big)?;
                let mut out = vec![0u8; need];
                let mut d = flate2::Decompress::new(true);
                d.decompress(src, &mut out, flate2::FlushDecompress::Finish).map_err(|_| RawError::malformed("deflate tile does not decompress"))?;
                out.truncate(usize::try_from(d.total_out()).unwrap_or(need).min(need));
                let v = unpack(&out, s.w, rows, samples, bits, le)?;
                // The TIFF predictor runs after unpacking, per row, per sample plane.
                match t.tag_uint(ifd, tag::PREDICTOR).unwrap_or(1) {
                    1 => Ok(v),
                    2 => Ok(undelta(&v, samples, s.w * samples, rows, bits)),
                    3 => Err(RawError::unsupported("floating-point predictor (compression 8, predictor 3)")),
                    p => Err(RawError::malformed(format!("unknown predictor {p}"))),
                }
            }
            // A JPEG XL segment codes only the rows inside the image (strips) or the whole tile.
            JPEG_XL => {
                let coded_rows = if t.tag_uint(ifd, tag::TILE_WIDTH).is_some() { s.h } else { rows };
                let mut v = jxl::decode(src, s.w, coded_rows, samples, bits, limits)?;
                v.resize(s.w * s.h * samples, 0);
                Ok(v)
            }
            34892 => {
                // Adobe DNG compression 34892: baseline JPEG tiles (lossy, 8-bit). The tile is
                // a complete JPEG stream: a CFA part decodes to gray samples, a 3-sample
                // LinearRaw part holds YCbCr, which zune-jpeg converts to RGB. The header is
                // decoded and size-checked before any pixel allocation, so a hostile tile
                // cannot size the decoder's buffers (up to 16384x16384) with a fake SOF.
                let out = match samples {
                    1 => zune_core::colorspace::ColorSpace::Luma,
                    3 => zune_core::colorspace::ColorSpace::RGB,
                    n => return Err(RawError::unsupported(format!("lossy DNG with {n} samples per pixel (1 or 3 are supported)"))),
                };
                let options = zune_core::options::DecoderOptions::default().jpeg_set_out_colorspace(out);
                let mut dec = zune_jpeg::JpegDecoder::new_with_options(zune_core::bytestream::ZCursor::new(src), options);
                dec.decode_headers().map_err(|e| RawError::malformed(format!("lossy JPEG tile header: {e:?}")))?;
                let f = dec.info().ok_or_else(|| RawError::malformed("lossy JPEG tile has no header"))?;
                if (f.width as usize, f.height as usize) != (s.w, rows) {
                    return Err(RawError::malformed(format!("lossy JPEG tile is {}x{}, expected {}x{}", f.width, f.height, s.w, rows)));
                }
                let decoded = dec.decode().map_err(|e| RawError::malformed(format!("lossy JPEG tile: {e:?}")))?;
                let need = s.w * rows * samples;
                if decoded.len() != need {
                    return Err(RawError::malformed(format!("lossy JPEG tile holds {} samples, expected {need}", decoded.len())));
                }
                // The 8-bit samples widen to the document's 16-bit range (0xFF -> 0xFFFF).
                Ok(decoded.into_iter().map(|b| u16::from(b) * 257).collect())
            }
            // Compression 7 (lossless JPEG): the frame the entropy stream declares is capped
            // by the budget, not only by the segment, and the `Quads` scatter buffer below
            // comes out of the same budget. `max_alloc` counts bytes, a sample is two.
            _ => {
                let need = s.w * rows * samples;
                let scatter = if layout == JpegLayout::Quads && samples == 1 { s.w.saturating_mul(s.h).saturating_mul(2) as u64 } else { 0 };
                let budget = usize::try_from(limits.max_alloc.saturating_sub(scatter) / 2).unwrap_or(usize::MAX);
                let geom = s.w.saturating_mul(s.h).saturating_mul(samples).saturating_mul(4);
                let (f, v) = ljpeg::decode(src, geom.min(budget))?;
                if v.len() < need {
                    return Err(RawError::malformed("lossless JPEG tile is smaller than its tile"));
                }
                if layout == JpegLayout::Quads && samples == 1 && f.components == 4 && f.width * 2 == s.w && f.height * 2 == s.h {
                    let mut tile = vec![0u16; s.w * s.h];
                    for (i, q) in v.as_chunks::<4>().0.iter().enumerate() {
                        let at = 2 * (i / f.width) * s.w + 2 * (i % f.width);
                        tile[at] = q[0];
                        tile[at + 1] = q[1];
                        tile[at + s.w] = q[2];
                        tile[at + s.w + 1] = q[3];
                    }
                    return Ok(tile);
                }
                Ok(v)
            }
        }
    });
    let mut data = vec![0u16; width * height * samples];
    for (s, d) in segments.iter().zip(decoded) {
        let d = d?;
        let rows = s.h.min(height - s.y);
        let cols = s.w.min(width - s.x);
        for r in 0..rows {
            let src = &d[r * s.w * samples..(r * s.w + cols) * samples];
            let at = ((s.y + r) * width + s.x) * samples;
            data[at..at + cols * samples].copy_from_slice(src);
        }
    }
    Ok(Plane { width, height, samples, bits, data })
}

pub(crate) fn segments(t: &Tiff, ifd: &Ifd, width: usize, height: usize) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    if let (Some(tw), Some(th)) = (t.tag_uint(ifd, tag::TILE_WIDTH), t.tag_uint(ifd, tag::TILE_LENGTH)) {
        let (tw, th) = (tw as usize, th as usize);
        if tw == 0 || th == 0 || tw > 1 << 16 || th > 1 << 16 {
            return Err(RawError::malformed("bad tile size"));
        }
        let across = width.div_ceil(tw);
        let down = height.div_ceil(th);
        let offsets = t.tag_uints(ifd, tag::TILE_OFFSETS);
        let counts = t.tag_uints(ifd, tag::TILE_BYTE_COUNTS);
        let n = across.checked_mul(down).ok_or_else(|| RawError::malformed("too many tiles"))?;
        if offsets.len() < n || counts.len() < n {
            return Err(RawError::malformed("missing tile offsets"));
        }
        for i in 0..n {
            out.push(Segment { x: (i % across) * tw, y: (i / across) * th, w: tw, h: th, offset: offsets[i] as usize, len: counts[i] as usize });
        }
    } else {
        let rps = match t.tag_uint(ifd, tag::ROWS_PER_STRIP) {
            Some(r) if r > 0 => (r as usize).min(height),
            _ => height,
        };
        let n = height.div_ceil(rps.max(1));
        let offsets = t.tag_uints(ifd, tag::STRIP_OFFSETS);
        let mut counts = t.tag_uints(ifd, tag::STRIP_BYTE_COUNTS);
        if offsets.len() < n {
            return Err(RawError::malformed("missing strip offsets"));
        }
        if counts.len() < n {
            if n == 1 {
                counts = vec![(t.data.len().saturating_sub(offsets[0] as usize)) as u32];
            } else {
                return Err(RawError::malformed("missing strip byte counts"));
            }
        }
        for i in 0..n {
            out.push(Segment { x: 0, y: i * rps, w: width, h: rps, offset: offsets[i] as usize, len: counts[i] as usize });
        }
    }
    Ok(out)
}

/// Reverses TIFF predictor 2 (horizontal differencing) on interleaved samples: each sample
/// gains its channel's predecessor in the row, wrapping at the sample's container (8 or 16 bits).
fn undelta(v: &[u16], plane: usize, row_samples: usize, rows: usize, bits: u32) -> Vec<u16> {
    let mask: u16 = if bits <= 8 { 0xFF } else { 0xFFFF };
    let mut out = v.to_vec();
    for row in out.chunks_mut(row_samples.max(1)).take(rows) {
        for i in plane..row.len() {
            if let (Some(&prev), Some(&cur)) = (row.get(i - plane), row.get(i)) {
                row[i] = cur.wrapping_add(prev) & mask;
            }
        }
    }
    out
}

/// Unpacks uncompressed samples: 8-bit, 16-bit in the file's byte order, or
/// MSB-first bit-packed rows padded to a byte (TIFF 6.0). 9–15-bit data
/// stored in 16-bit containers (common in camera raws) is recognised by its
/// byte count.
fn unpack(src: &[u8], w: usize, rows: usize, samples: usize, bits: u32, le: bool) -> Result<Vec<u16>> {
    let n = w * rows * samples;
    let truncated = || RawError::malformed("uncompressed raw data is truncated");
    if bits == 16 || (bits > 8 && src.len() >= n * 2) {
        let b = src.get(..n * 2).ok_or_else(truncated)?;
        return Ok(b.as_chunks::<2>().0.iter().map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) }).collect());
    }
    if bits == 8 {
        return Ok(src.get(..n).ok_or_else(truncated)?.iter().map(|&v| u16::from(v)).collect());
    }
    let row_bytes = (w * samples * bits as usize).div_ceil(8);
    if src.len() < row_bytes * rows {
        return Err(truncated());
    }
    let mut out = Vec::with_capacity(n);
    for r in 0..rows {
        let row = &src[r * row_bytes..(r + 1) * row_bytes];
        let mut acc: u32 = 0;
        let mut have = 0u32;
        let mut it = row.iter();
        for _ in 0..w * samples {
            while have < bits {
                acc = (acc << 8) | u32::from(*it.next().unwrap_or(&0));
                have += 8;
            }
            have -= bits;
            out.push(((acc >> have) & ((1 << bits) - 1)) as u16);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_patterns() {
        for p in ["RGGB", "BGGR", "GRBG", "GBRG"] {
            assert!(Cfa::bayer(p).unwrap().is_bayer(), "{p}");
        }
        for p in ["RGBG", "RRGB", "GGRB"] {
            assert!(!Cfa::bayer(p).unwrap().is_bayer(), "{p}");
        }
        let mut c = Cfa::bayer("RGGB").unwrap();
        c.origin_x = 1;
        assert_eq!(c.phase(0, 0), [1, 0, 2, 1]); // GRBG
    }

    #[test]
    fn unpack_12_bit() {
        // Two 12-bit samples 0xABC, 0x123 packed MSB first.
        assert_eq!(unpack(&[0xAB, 0xC1, 0x23], 2, 1, 1, 12, true).unwrap(), vec![0xABC, 0x123]);
        // 12-bit data in 16-bit containers.
        assert_eq!(unpack(&[0xBC, 0x0A, 0x23, 0x01], 2, 1, 1, 12, true).unwrap(), vec![0xABC, 0x123]);
        assert!(unpack(&[0xAB], 2, 1, 1, 12, true).is_err());
    }
}
