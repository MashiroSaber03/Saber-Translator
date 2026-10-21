//! Channel image data codecs: Raw, RLE (PackBits), ZIP, and ZIP with prediction.
//!
//! All decoded data is *planar* bytes in the file's native big-endian sample
//! order: `planes × height × row_bytes` where `row_bytes` depends on depth
//! (depth 1 packs 8 pixels per byte, MSB first).

use std::io::{Read, Write};

use crate::error::{PsdError, Result};
use crate::header::{Version, row_bytes};
use crate::io::Reader;

/// Maximum number of decoded bytes a single decode call may produce: 8 GiB on
/// 64-bit targets, 2 GiB elsewhere (the same budget as `photocraft-codecs`'
/// default `Limits::max_alloc`).
///
/// The decoders bound their output by their input on their own (RLE expands
/// at most 64x, Raw needs every byte present), so this is a backstop against
/// decompression bombs, not the main guard. It has to fit real PSBs: a
/// 30000² RGB merged image is 2.7 GB, and a 33000² 16-bit layer channel is
/// 2.2 GB, both of which a 2 GiB cap turned into empty layers (#375).
pub const MAX_DECODED_BYTES: u64 = if cfg!(target_pointer_width = "64") { 8 << 30 } else { 2 << 30 };

/// Compression method stored before channel / image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Compression {
    /// 0: uncompressed.
    Raw,
    /// 1: PackBits RLE with a per-row byte-count table.
    Rle,
    /// 2: zlib without prediction.
    Zip,
    /// 3: zlib with horizontal delta prediction.
    ZipPrediction,
    /// Unknown method; data is preserved but cannot be decoded.
    Unknown(u16),
}

impl Compression {
    /// All four decodable methods.
    pub const ALL: [Compression; 4] = [Compression::Raw, Compression::Rle, Compression::Zip, Compression::ZipPrediction];

    /// From the stored value.
    pub fn from_u16(v: u16) -> Self {
        match v {
            0 => Compression::Raw,
            1 => Compression::Rle,
            2 => Compression::Zip,
            3 => Compression::ZipPrediction,
            v => Compression::Unknown(v),
        }
    }
    /// Stored value.
    pub fn as_u16(self) -> u16 {
        match self {
            Compression::Raw => 0,
            Compression::Rle => 1,
            Compression::Zip => 2,
            Compression::ZipPrediction => 3,
            Compression::Unknown(v) => v,
        }
    }
}

/// PackBits (Apple / TIFF) run-length coding as used by Photoshop RLE.
pub mod packbits {
    use crate::error::{PsdError, Result};

    /// Encodes one row, appending to `out`. Never emits the `-128` no-op.
    pub fn encode(src: &[u8], out: &mut Vec<u8>) {
        let n = src.len();
        let mut i = 0;
        while i < n {
            // Length of the run starting at i.
            let mut run = 1;
            while i + run < n && run < 128 && src[i + run] == src[i] {
                run += 1;
            }
            // A literal never stops before a 2-byte run, so any run of >= 2
            // seen here starts a packet boundary: a run packet is never larger.
            if run >= 2 {
                out.push((1i16 - run as i16) as i8 as u8);
                out.push(src[i]);
                i += run;
                continue;
            }
            // Literal: extend until a run of >= 3 begins or 128 bytes.
            let start = i;
            let mut len = 0;
            while i < n && len < 128 {
                if i + 2 < n && src[i] == src[i + 1] && src[i] == src[i + 2] {
                    break;
                }
                i += 1;
                len += 1;
            }
            out.push((len - 1) as u8);
            out.extend_from_slice(&src[start..start + len]);
        }
    }

    /// Returns the encoded form of `src`.
    pub fn encode_vec(src: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(src.len() + src.len() / 128 + 1);
        encode(src, &mut v);
        v
    }

    /// Decodes `src` into exactly `expected` bytes appended to `out`.
    ///
    /// Errors if the input is exhausted early or a packet would overflow the
    /// row. Trailing input after the row is complete is ignored except for
    /// `-128` no-op bytes (which are always permitted).
    pub fn decode_into(src: &[u8], expected: usize, out: &mut Vec<u8>) -> Result<()> {
        let target = out.len() + expected;
        let mut i = 0;
        while out.len() < target {
            let Some(&h) = src.get(i) else {
                return Err(PsdError::Decompress("PackBits row ended early".into()));
            };
            i += 1;
            let h = h as i8;
            if h >= 0 {
                let len = h as usize + 1;
                let Some(lit) = src.get(i..i + len) else {
                    return Err(PsdError::Decompress("PackBits literal truncated".into()));
                };
                if out.len() + len > target {
                    return Err(PsdError::Decompress("PackBits literal overflows row".into()));
                }
                out.extend_from_slice(lit);
                i += len;
            } else if h != -128 {
                let len = 1 - h as isize;
                let len = len as usize;
                let Some(&b) = src.get(i) else {
                    return Err(PsdError::Decompress("PackBits run truncated".into()));
                };
                i += 1;
                if out.len() + len > target {
                    return Err(PsdError::Decompress("PackBits run overflows row".into()));
                }
                out.extend(std::iter::repeat_n(b, len));
            }
        }
        Ok(())
    }

    /// Decodes into a new vector of exactly `expected` bytes.
    pub fn decode(src: &[u8], expected: usize) -> Result<Vec<u8>> {
        let mut v = Vec::with_capacity(expected.min(src.len().saturating_mul(64)));
        decode_into(src, expected, &mut v)?;
        Ok(v)
    }
}

/// Geometry of a set of planes to encode/decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneLayout {
    /// Number of planes (1 for a layer channel, channel count for the merged image).
    pub planes: usize,
    /// Pixels per row.
    pub width: usize,
    /// Rows per plane.
    pub height: usize,
    /// Bits per sample (1, 8, 16, 32).
    pub depth: u16,
    /// PSD vs PSB (affects RLE count width).
    pub version: Version,
}

impl PlaneLayout {
    /// Bytes per row.
    pub fn row_bytes(&self) -> usize {
        row_bytes(self.width, self.depth)
    }
    /// Total rows over all planes.
    pub fn rows(&self) -> Result<usize> {
        self.planes.checked_mul(self.height).ok_or(PsdError::LimitExceeded("row count overflow"))
    }
    /// Total decoded bytes without the [`MAX_DECODED_BYTES`] cap.
    pub fn total_bytes(&self) -> Result<u64> {
        (self.planes as u64)
            .checked_mul(self.height as u64)
            .and_then(|v| v.checked_mul(self.row_bytes() as u64))
            .ok_or(PsdError::LimitExceeded("decoded size overflow"))
    }

    /// Total decoded bytes, checked against [`MAX_DECODED_BYTES`].
    pub fn decoded_len(&self) -> Result<usize> {
        let total = self.total_bytes()?;
        if total > MAX_DECODED_BYTES {
            return Err(PsdError::LimitExceeded("decoded channel data exceeds MAX_DECODED_BYTES"));
        }
        usize::try_from(total).map_err(|_| PsdError::LimitExceeded("decoded size exceeds address space"))
    }
    fn count_size(&self) -> usize {
        if self.version.is_psb() { 4 } else { 2 }
    }
}

/// Decodes planar data (excluding the 2-byte compression field).
pub fn decode_planes(compression: Compression, data: &[u8], layout: &PlaneLayout) -> Result<Vec<u8>> {
    let total = layout.decoded_len()?;
    match compression {
        Compression::Raw => {
            if data.len() < total {
                return Err(PsdError::UnexpectedEof { offset: data.len(), needed: total - data.len() });
            }
            let mut out = Vec::new();
            out.try_reserve_exact(total).map_err(|_| PsdError::LimitExceeded("not enough memory for the decoded channel data"))?;
            out.extend_from_slice(&data[..total]);
            Ok(out)
        }
        Compression::Rle => decode_rle(data, layout, total),
        Compression::Zip => inflate_checked(data, total as u64, true),
        Compression::ZipPrediction => {
            let mut v = inflate_checked(data, total as u64, true)?;
            unpredict(&mut v, layout)?;
            Ok(v)
        }
        Compression::Unknown(c) => Err(PsdError::Unsupported(format!("compression method {c}"))),
    }
}

/// Encodes planar data (excluding the 2-byte compression field).
pub fn encode_planes(compression: Compression, decoded: &[u8], layout: &PlaneLayout) -> Result<Vec<u8>> {
    let total = layout.decoded_len()?;
    if decoded.len() != total {
        return Err(PsdError::invalid(format!("decoded buffer has {} bytes, layout requires {}", decoded.len(), total)));
    }
    match compression {
        Compression::Raw => Ok(decoded.to_vec()),
        Compression::Rle => encode_rle(decoded, layout),
        Compression::Zip => Ok(zip_compress(decoded)),
        Compression::ZipPrediction => {
            let mut v = decoded.to_vec();
            predict(&mut v, layout)?;
            Ok(zip_compress(&v))
        }
        Compression::Unknown(c) => Err(PsdError::Unsupported(format!("compression method {c}"))),
    }
}

/// Validates encoded sizes and, for ZIP, the complete stream and checksum.
///
/// A ZIP stream whose declared size is over [`MAX_DECODED_BYTES`] is not
/// decompressed at all (#2926): [`decode_planes`] refuses that layout with
/// `LimitExceeded` anyway, so inflating it here would only be work on an
/// untrusted file before the same error. The bytes are kept verbatim, like an
/// unknown compression method, so a layered file whose merged composite is
/// too large still opens.
pub(crate) fn validate_planes(compression: Compression, data: &[u8], layout: &PlaneLayout) -> Result<()> {
    let total = layout.total_bytes()?;
    match compression {
        Compression::Raw => {
            if (data.len() as u64) < total {
                return Err(PsdError::UnexpectedEof { offset: data.len(), needed: usize::try_from(total - data.len() as u64).unwrap_or(usize::MAX) });
            }
            Ok(())
        }
        Compression::Rle => {
            let rows = layout.rows()?;
            let mut r = Reader::new(data);
            r.check_count(rows as u64, layout.count_size() as u64)?;
            let mut sum: u64 = 0;
            for _ in 0..rows {
                sum += read_count(&mut r, layout.version)? as u64;
            }
            if sum > r.remaining() as u64 {
                return Err(PsdError::UnexpectedEof { offset: data.len(), needed: usize::try_from(sum - r.remaining() as u64).unwrap_or(usize::MAX) });
            }
            Ok(())
        }
        Compression::Zip | Compression::ZipPrediction if total > MAX_DECODED_BYTES => Ok(()),
        Compression::Zip | Compression::ZipPrediction => inflate_checked(data, total, false).map(|_| ()),
        Compression::Unknown(_) => Ok(()),
    }
}

// Keep only the requested pixels, but consume the trailer to verify Adler-32.
// A bounded surplus tolerates padding without unbounded decompression work.
fn inflate_checked(data: &[u8], expected: u64, keep: bool) -> Result<Vec<u8>> {
    const OOM: PsdError = PsdError::LimitExceeded("not enough memory for the decoded channel data");
    let mut decoder = flate2::Decompress::new(true);
    let mut out = Vec::new();
    if keep {
        out.try_reserve_exact(expected.min(data.len().saturating_mul(1032) as u64) as usize).map_err(|_| OOM)?;
    }
    let mut buf = vec![0; expected.saturating_add(1).clamp(8192, 1 << 20) as usize];
    loop {
        let before = (decoder.total_in(), decoder.total_out());
        let input = data.get(before.0 as usize..).ok_or_else(|| PsdError::Decompress("zlib input overflow".into()))?;
        let status = decoder.decompress(input, &mut buf, flate2::FlushDecompress::None).map_err(|e| PsdError::Decompress(e.to_string()))?;
        let got = decoder.total_out();
        if got > expected.saturating_add(1 << 20) {
            return Err(PsdError::LimitExceeded("zlib output exceeds the declared size and padding allowance"));
        }
        if keep {
            let n = (got - before.1).min(expected.saturating_sub(out.len() as u64)) as usize;
            out.try_reserve(n).map_err(|_| OOM)?;
            out.extend_from_slice(buf.get(..n).ok_or_else(|| PsdError::Decompress("zlib output overflow".into()))?);
        }
        if status == flate2::Status::StreamEnd {
            if got < expected {
                return Err(PsdError::Decompress(format!("zlib stream produced {got} of {expected} bytes")));
            }
            return Ok(out);
        }
        if before == (decoder.total_in(), got) {
            return Err(PsdError::Decompress("incomplete zlib stream".into()));
        }
    }
}

fn read_count(r: &mut Reader<'_>, v: Version) -> Result<usize> {
    Ok(if v.is_psb() { r.u32()? as usize } else { r.u16()? as usize })
}

fn decode_rle(data: &[u8], layout: &PlaneLayout, total: usize) -> Result<Vec<u8>> {
    let rows = layout.rows()?;
    let rb = layout.row_bytes();
    let mut r = Reader::new(data);
    r.check_count(rows as u64, layout.count_size() as u64)?;
    let mut counts = Vec::with_capacity(rows);
    for _ in 0..rows {
        counts.push(read_count(&mut r, layout.version)?);
    }
    // PackBits expands at most 64x (2 input bytes -> 128 output bytes).
    if (total as u64) > (r.remaining() as u64).saturating_mul(64) {
        return Err(PsdError::Decompress("RLE data too short for declared size".into()));
    }
    // Up to MAX_DECODED_BYTES: report a machine without that much memory instead of aborting.
    let mut out = Vec::new();
    out.try_reserve_exact(total).map_err(|_| PsdError::LimitExceeded("not enough memory for the decoded channel data"))?;
    for c in counts {
        let row = r.bytes(c)?;
        packbits::decode_into(row, rb, &mut out)?;
    }
    Ok(out)
}

fn encode_rle(decoded: &[u8], layout: &PlaneLayout) -> Result<Vec<u8>> {
    let rows = layout.rows()?;
    let rb = layout.row_bytes();
    let cs = layout.count_size();
    if rows == 0 {
        return Ok(Vec::new());
    }
    // Rows are independent, so they pack in parallel chunks and concatenate in order.
    let threads = if cfg!(target_arch = "wasm32") { 1 } else { std::thread::available_parallelism().map_or(1, |v| v.get()).clamp(1, 32) };
    let chunk_rows = rows.div_ceil(threads).max(1);
    struct Chunk {
        counts: Vec<u8>,
        packed: Vec<u8>,
    }
    let pack_chunk = |first: usize, count: usize| -> Result<Chunk> {
        let mut chunk = Chunk { counts: vec![0u8; count * cs], packed: Vec::new() };
        for row in first..first + count {
            let start = chunk.packed.len();
            packbits::encode(&decoded[row * rb..(row + 1) * rb], &mut chunk.packed);
            let n = chunk.packed.len() - start;
            if cs == 2 {
                let n = u16::try_from(n).map_err(|_| PsdError::LimitExceeded("RLE row exceeds 65535 bytes (use PSB)"))?;
                chunk.counts[(row - first) * 2..(row - first) * 2 + 2].copy_from_slice(&n.to_be_bytes());
            } else {
                let n = u32::try_from(n).map_err(|_| PsdError::LimitExceeded("RLE row too large"))?;
                chunk.counts[(row - first) * 4..(row - first) * 4 + 4].copy_from_slice(&n.to_be_bytes());
            }
        }
        Ok(chunk)
    };
    let chunk_list: Vec<Chunk> = if threads == 1 {
        vec![pack_chunk(0, rows)?]
    } else {
        std::thread::scope(|scope| -> Result<Vec<Chunk>> {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let first = t * chunk_rows;
                    let count = chunk_rows.min(rows.saturating_sub(first));
                    scope.spawn(move || pack_chunk(first, count))
                })
                .collect();
            // Join every worker first, then surface the first error like the sequential
            // path would instead of aborting the scope with a thread panic.
            let mut all = Vec::with_capacity(threads);
            for h in handles {
                all.push(h.join().map_err(|_| PsdError::LimitExceeded("an RLE worker panicked"))??);
            }
            Ok(all)
        })?
    };
    let packed_total: usize = chunk_list.iter().map(|c| c.packed.len()).sum();
    let mut out = vec![0u8; rows * cs + packed_total];
    let mut counts_at = 0usize;
    let mut packed_at = rows * cs;
    for chunk in &chunk_list {
        out[counts_at..counts_at + chunk.counts.len()].copy_from_slice(&chunk.counts);
        counts_at += chunk.counts.len();
        out[packed_at..packed_at + chunk.packed.len()].copy_from_slice(&chunk.packed);
        packed_at += chunk.packed.len();
    }
    Ok(out)
}

/// zlib-compresses `data` at the default level.
pub fn zip_compress(data: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    // Writing into a Vec cannot fail.
    let _ = e.write_all(data);
    e.finish().unwrap_or_default()
}

/// Decompresses a zlib stream, requiring at least `expected` output bytes and
/// returning exactly `expected`.
///
/// The output is reserved fallibly and read in chunks, never past `expected`
/// bytes, so a stream declaring a huge size reports an error on a machine
/// without that much memory instead of aborting.
pub fn zip_decompress(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    const OOM: PsdError = PsdError::LimitExceeded("not enough memory for the decoded channel data");
    const CHUNK: usize = 1 << 20;
    let mut dec = flate2::read::ZlibDecoder::new(data).take(u64::try_from(expected).unwrap_or(u64::MAX));
    let mut out = Vec::new();
    // Deflate expands at most ~1032:1, so this is the most a valid stream can need.
    out.try_reserve_exact(expected.min(data.len().saturating_mul(1032))).map_err(|_| OOM)?;
    let mut buf = vec![0u8; CHUNK.min(expected).max(1)];
    loop {
        let n = match dec.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(PsdError::Decompress(e.to_string())),
        };
        let chunk = buf.get(..n).ok_or_else(|| PsdError::Decompress("zlib reader overran its buffer".into()))?;
        out.try_reserve(n).map_err(|_| OOM)?;
        out.extend_from_slice(chunk);
    }
    if out.len() < expected {
        return Err(PsdError::Decompress(format!("zlib stream produced {} of {} bytes", out.len(), expected)));
    }
    Ok(out)
}

/// Applies horizontal delta prediction in place (encoder side).
///
/// * 1/8-bit: byte-wise delta per row.
/// * 16-bit: big-endian u16 sample delta per row.
/// * 32-bit: each row's samples are split into four byte planes (MSB plane
///   first) and the resulting row is byte-wise delta coded, as Photoshop does.
pub fn predict(buf: &mut [u8], layout: &PlaneLayout) -> Result<()> {
    let rb = layout.row_bytes();
    if rb == 0 {
        return Ok(());
    }
    let w = layout.width;
    let mut scratch = Vec::new();
    for row in buf.chunks_exact_mut(rb) {
        match layout.depth {
            1 | 8 => {
                for i in (1..row.len()).rev() {
                    row[i] = row[i].wrapping_sub(row[i - 1]);
                }
            }
            16 => {
                for i in (1..w).rev() {
                    let a = u16::from_be_bytes([row[2 * i], row[2 * i + 1]]);
                    let b = u16::from_be_bytes([row[2 * i - 2], row[2 * i - 1]]);
                    row[2 * i..2 * i + 2].copy_from_slice(&a.wrapping_sub(b).to_be_bytes());
                }
            }
            32 => {
                scratch.clear();
                scratch.resize(rb, 0);
                for i in 0..w {
                    for k in 0..4 {
                        scratch[k * w + i] = row[4 * i + k];
                    }
                }
                for i in (1..rb).rev() {
                    scratch[i] = scratch[i].wrapping_sub(scratch[i - 1]);
                }
                row.copy_from_slice(&scratch);
            }
            d => return Err(PsdError::Unsupported(format!("prediction for depth {d}"))),
        }
    }
    Ok(())
}

/// Reverses [`predict`] in place (decoder side).
pub fn unpredict(buf: &mut [u8], layout: &PlaneLayout) -> Result<()> {
    let rb = layout.row_bytes();
    if rb == 0 {
        return Ok(());
    }
    let w = layout.width;
    let mut scratch = Vec::new();
    for row in buf.chunks_exact_mut(rb) {
        match layout.depth {
            1 | 8 => {
                for i in 1..row.len() {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
            }
            16 => {
                for i in 1..w {
                    let a = u16::from_be_bytes([row[2 * i], row[2 * i + 1]]);
                    let b = u16::from_be_bytes([row[2 * i - 2], row[2 * i - 1]]);
                    row[2 * i..2 * i + 2].copy_from_slice(&a.wrapping_add(b).to_be_bytes());
                }
            }
            32 => {
                for i in 1..rb {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
                scratch.clear();
                scratch.extend_from_slice(row);
                for i in 0..w {
                    for k in 0..4 {
                        row[4 * i + k] = scratch[k * w + i];
                    }
                }
            }
            d => return Err(PsdError::Unsupported(format!("prediction for depth {d}"))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_zip_validates_trailer_size_and_padding_budget() {
        let layout = PlaneLayout { planes: 1, width: 3, height: 1, depth: 8, version: Version::Psd };
        for compression in [Compression::Zip, Compression::ZipPrediction] {
            let data = encode_planes(compression, &[1, 2, 3], &layout).unwrap();
            for cut in 0..data.len() {
                assert!(decode_planes(compression, &data[..cut], &layout).is_err(), "cut {cut}");
                assert!(validate_planes(compression, &data[..cut], &layout).is_err());
            }
            let mut corrupt = data;
            *corrupt.last_mut().unwrap() ^= 1;
            assert!(decode_planes(compression, &corrupt, &layout).is_err());
            assert!(validate_planes(compression, &corrupt, &layout).is_err());
        }
        let padded = zip_compress(&[1, 2, 3, 0]);
        assert_eq!(decode_planes(Compression::Zip, &padded, &layout).unwrap(), [1, 2, 3]);
        let bomb = zip_compress(&vec![0; (1 << 20) + 4]);
        assert!(decode_planes(Compression::Zip, &bomb, &layout).is_err());
        assert!(validate_planes(Compression::Zip, &bomb, &layout).is_err());
    }

    fn layout(planes: usize, width: usize, height: usize, depth: u16, version: Version) -> PlaneLayout {
        PlaneLayout { planes, width, height, depth, version }
    }

    fn pb_roundtrip(src: &[u8]) -> Vec<u8> {
        let enc = packbits::encode_vec(src);
        let dec = packbits::decode(&enc, src.len()).unwrap();
        assert_eq!(dec, src, "encoded: {enc:?}");
        enc
    }

    #[test]
    fn zip_absurd_declared_size_is_an_error() {
        let z = zip_compress(&[7u8; 64]);
        for expected in [1usize << 40, usize::MAX / 2, usize::MAX] {
            assert!(zip_decompress(&z, expected).is_err(), "expected {expected}");
        }
        // Through decode_planes: a tiny ZIP channel claiming a huge (but capped) plane.
        let l = layout(1, 1 << 16, 1 << 15, 8, Version::Psb);
        assert!(decode_planes(Compression::Zip, &z, &l).is_err());
        assert!(decode_planes(Compression::ZipPrediction, &z, &l).is_err());
        // Past MAX_DECODED_BYTES.
        let l = layout(4, 1 << 20, 1 << 20, 32, Version::Psb);
        assert!(decode_planes(Compression::Zip, &z, &l).is_err());
        // Garbage that is not zlib at all.
        assert!(zip_decompress(&[0xff; 16], 1 << 30).is_err());
    }

    #[test]
    fn zip_decompress_exact_and_truncated() {
        let src: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        let z = zip_compress(&src);
        assert_eq!(zip_decompress(&z, src.len()).unwrap(), src);
        // Asking for fewer bytes returns exactly that many.
        assert_eq!(zip_decompress(&z, 1000).unwrap(), &src[..1000]);
        assert!(zip_decompress(&z, src.len() + 1).is_err());
        assert!(zip_decompress(&z, 0).unwrap().is_empty());
    }

    #[test]
    fn packbits_empty() {
        assert!(pb_roundtrip(&[]).is_empty());
    }

    #[test]
    fn packbits_single() {
        assert_eq!(pb_roundtrip(&[7]), vec![0, 7]);
    }

    #[test]
    fn packbits_run_of_two() {
        assert_eq!(pb_roundtrip(&[5, 5]), vec![0xff, 5]);
    }

    #[test]
    fn packbits_run_128() {
        let src = vec![9u8; 128];
        assert_eq!(pb_roundtrip(&src), vec![0x81, 9]);
    }

    #[test]
    fn packbits_run_129() {
        let src = vec![9u8; 129];
        assert_eq!(pb_roundtrip(&src), vec![0x81, 9, 0, 9]);
    }

    #[test]
    fn packbits_run_130_and_256() {
        pb_roundtrip(&[1u8; 130]);
        let enc = pb_roundtrip(&[1u8; 256]);
        assert_eq!(enc, vec![0x81, 1, 0x81, 1]);
    }

    #[test]
    fn packbits_literal_128_and_129() {
        let src: Vec<u8> = (0..128u32).map(|i| i as u8).collect();
        let enc = pb_roundtrip(&src);
        assert_eq!(enc[0], 127);
        assert_eq!(enc.len(), 129);
        let src: Vec<u8> = (0..129u32).map(|i| i as u8).collect();
        let enc = pb_roundtrip(&src);
        assert_eq!(enc[0], 127);
        assert_eq!(enc[129], 0);
        assert_eq!(enc.len(), 131);
    }

    #[test]
    fn packbits_literal_run_boundary() {
        pb_roundtrip(&[1, 2, 3, 3, 3, 3, 4, 5]);
        pb_roundtrip(&[1, 1, 2, 2, 3, 3]);
        pb_roundtrip(&[1, 2, 2, 3]);
        pb_roundtrip(&[0, 0, 0, 1, 0, 0, 0]);
        let enc = pb_roundtrip(&[1, 2, 3, 3, 3]);
        assert_eq!(enc, vec![1, 1, 2, 0xfe, 3]);
    }

    #[test]
    fn packbits_never_emits_noop() {
        for n in 0..300usize {
            let src: Vec<u8> = (0..n).map(|i| ((i / 3) % 5) as u8).collect();
            let enc = pb_roundtrip(&src);
            // Walk the packets to find headers.
            let mut i = 0;
            while i < enc.len() {
                let h = enc[i] as i8;
                assert_ne!(h, -128);
                i += if h >= 0 { h as usize + 2 } else { 2 };
            }
        }
    }

    #[test]
    fn packbits_decode_noop_skipped() {
        assert_eq!(packbits::decode(&[0x80, 0, 42], 1).unwrap(), vec![42]);
    }

    #[test]
    fn packbits_decode_errors() {
        assert!(packbits::decode(&[], 1).is_err());
        assert!(packbits::decode(&[2, 1, 2], 3).is_err()); // literal truncated
        assert!(packbits::decode(&[0xfe], 3).is_err()); // run truncated
        assert!(packbits::decode(&[0xfd, 1], 3).is_err()); // run overflows (4 > 3)
        assert!(packbits::decode(&[3, 1, 2, 3, 4], 3).is_err()); // literal overflows
    }

    #[test]
    fn packbits_zero_expected_ignores_input() {
        assert_eq!(packbits::decode(&[0xff, 1], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn compression_values() {
        for v in 0..6 {
            assert_eq!(Compression::from_u16(v).as_u16(), v);
        }
        assert_eq!(Compression::from_u16(9), Compression::Unknown(9));
    }

    fn sample_data(l: &PlaneLayout) -> Vec<u8> {
        let n = l.decoded_len().unwrap();
        (0..n).map(|i| ((i * 7 + i / 5) % 251) as u8).collect()
    }

    #[test]
    fn all_codecs_roundtrip_all_depths() {
        for version in [Version::Psd, Version::Psb] {
            for depth in [1u16, 8, 16, 32] {
                for (w, h) in [(0, 0), (1, 1), (3, 2), (17, 5), (0, 3), (4, 0)] {
                    for planes in [1, 3] {
                        let l = layout(planes, w, h, depth, version);
                        let data = sample_data(&l);
                        for c in Compression::ALL {
                            let enc = encode_planes(c, &data, &l).unwrap();
                            validate_planes(c, &enc, &l).unwrap();
                            let dec = decode_planes(c, &enc, &l).unwrap();
                            assert_eq!(dec, data, "{c:?} {depth} {w}x{h} {version:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rle_count_width_psd_vs_psb() {
        let data = vec![0u8; 4];
        let psd = encode_planes(Compression::Rle, &data, &layout(1, 2, 2, 8, Version::Psd)).unwrap();
        let psb = encode_planes(Compression::Rle, &data, &layout(1, 2, 2, 8, Version::Psb)).unwrap();
        assert_eq!(psd, vec![0, 2, 0, 2, 0xff, 0, 0xff, 0]);
        assert_eq!(psb, vec![0, 0, 0, 2, 0, 0, 0, 2, 0xff, 0, 0xff, 0]);
        assert!(decode_planes(Compression::Rle, &psd, &layout(1, 2, 2, 8, Version::Psb)).is_err());
    }

    #[test]
    fn rle_odd_width() {
        let l = layout(1, 7, 3, 16, Version::Psd);
        let data = sample_data(&l);
        let enc = encode_planes(Compression::Rle, &data, &l).unwrap();
        assert_eq!(decode_planes(Compression::Rle, &enc, &l).unwrap(), data);
    }

    #[test]
    fn rle_truncated_fails() {
        let l = layout(1, 10, 10, 8, Version::Psd);
        let data = sample_data(&l);
        let enc = encode_planes(Compression::Rle, &data, &l).unwrap();
        for cut in 0..enc.len() {
            assert!(decode_planes(Compression::Rle, &enc[..cut], &l).is_err(), "cut {cut}");
            assert!(validate_planes(Compression::Rle, &enc[..cut], &l).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn zip_truncated_fails() {
        let l = layout(1, 10, 10, 8, Version::Psd);
        let data = sample_data(&l);
        let enc = encode_planes(Compression::Zip, &data, &l).unwrap();
        for cut in 0..enc.len() - 4 {
            assert!(decode_planes(Compression::Zip, &enc[..cut], &l).is_err(), "cut {cut}");
            assert!(validate_planes(Compression::Zip, &enc[..cut], &l).is_err(), "cut {cut}");
        }
    }

    /// #2926: a ZIP stream declaring more than `MAX_DECODED_BYTES` is not inflated at parse
    /// time. Garbage zlib data proves it: decompressing it would fail, skipping it passes, and
    /// decoding the layout still reports the size limit.
    #[test]
    fn zip_over_budget_is_not_decompressed_at_parse() {
        let l = layout(4, 300_000, 300_000, 8, Version::Psb);
        assert!(l.total_bytes().unwrap() > MAX_DECODED_BYTES);
        let garbage = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03];
        for c in [Compression::Zip, Compression::ZipPrediction] {
            validate_planes(c, &garbage, &l).unwrap();
            assert!(matches!(decode_planes(c, &garbage, &l), Err(PsdError::LimitExceeded(_))), "{c:?}");
        }
        // Within the budget the stream is still checked.
        let small = layout(1, 4, 4, 8, Version::Psd);
        assert!(validate_planes(Compression::Zip, &garbage, &small).is_err());
    }

    #[test]
    fn raw_short_fails() {
        let l = layout(2, 3, 3, 8, Version::Psd);
        assert!(decode_planes(Compression::Raw, &[0; 17], &l).is_err());
        assert!(decode_planes(Compression::Raw, &[0; 18], &l).is_ok());
    }

    #[test]
    fn unknown_compression_errors() {
        let l = layout(1, 1, 1, 8, Version::Psd);
        assert!(decode_planes(Compression::Unknown(7), &[0], &l).is_err());
        assert!(encode_planes(Compression::Unknown(7), &[0], &l).is_err());
    }

    #[test]
    fn encode_rejects_wrong_size() {
        let l = layout(1, 2, 2, 8, Version::Psd);
        assert!(encode_planes(Compression::Raw, &[0; 3], &l).is_err());
    }

    #[test]
    fn decoded_len_limits() {
        let l = layout(56, 300_000, 300_000, 32, Version::Psb);
        assert!(matches!(l.decoded_len(), Err(PsdError::LimitExceeded(_))));
        assert!(decode_planes(Compression::Rle, &[0; 16], &l).is_err());
    }

    /// Real PSBs go past 2 GiB in one decode (#375): a 30000² RGB merged image and a 33000²
    /// 16-bit layer channel. Only checks the limit, so nothing is allocated.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn decoded_len_allows_large_psb_planes() {
        assert_eq!(layout(3, 30_000, 30_000, 8, Version::Psb).decoded_len().unwrap(), 2_700_000_000);
        assert_eq!(layout(1, 33_000, 33_000, 16, Version::Psb).decoded_len().unwrap(), 2_178_000_000);
        assert!(layout(4, 40_000, 40_000, 16, Version::Psb).decoded_len().is_err());
    }

    #[test]
    fn prediction_8bit_known_vector() {
        let l = layout(1, 4, 1, 8, Version::Psd);
        let mut v = vec![10, 12, 11, 255];
        predict(&mut v, &l).unwrap();
        assert_eq!(v, vec![10, 2, 255, 244]);
        unpredict(&mut v, &l).unwrap();
        assert_eq!(v, vec![10, 12, 11, 255]);
    }

    #[test]
    fn prediction_16bit_known_vector() {
        let l = layout(1, 3, 1, 16, Version::Psd);
        let mut v = vec![0x01, 0x00, 0x01, 0x05, 0x00, 0x00];
        predict(&mut v, &l).unwrap();
        assert_eq!(v, vec![0x01, 0x00, 0x00, 0x05, 0xfe, 0xfb]);
        unpredict(&mut v, &l).unwrap();
        assert_eq!(v, vec![0x01, 0x00, 0x01, 0x05, 0x00, 0x00]);
    }

    #[test]
    fn prediction_32bit_plane_shuffle() {
        // Two samples: AABBCCDD and 11223344 → planes [AA,11][BB,22][CC,33][DD,44]
        let l = layout(1, 2, 1, 32, Version::Psd);
        let orig = vec![0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44];
        let mut v = orig.clone();
        predict(&mut v, &l).unwrap();
        let planes = [0xAAu8, 0x11, 0xBB, 0x22, 0xCC, 0x33, 0xDD, 0x44];
        let mut expect = planes;
        for i in (1..8).rev() {
            expect[i] = expect[i].wrapping_sub(expect[i - 1]);
        }
        assert_eq!(v, expect);
        unpredict(&mut v, &l).unwrap();
        assert_eq!(v, orig);
    }

    #[test]
    fn prediction_32bit_floats_roundtrip() {
        let l = layout(1, 5, 2, 32, Version::Psd);
        let mut v = Vec::new();
        for i in 0..10 {
            v.extend_from_slice(&(i as f32 * 0.37 - 1.0).to_be_bytes());
        }
        let enc = encode_planes(Compression::ZipPrediction, &v, &l).unwrap();
        assert_eq!(decode_planes(Compression::ZipPrediction, &enc, &l).unwrap(), v);
    }

    #[test]
    fn zip_bomb_is_bounded() {
        let big = zip_compress(&vec![0u8; 1 << 20]);
        let out = zip_decompress(&big, 10).unwrap();
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn zip_garbage_errors() {
        assert!(zip_decompress(&[1, 2, 3, 4], 4).is_err());
    }
}

#[cfg(test)]
mod parallel_rle_equivalence {
    use super::*;

    /// The parallel row packer's output must be byte-equal to a straightforward sequential
    /// encoding of the same plane, across depths, row counts around the chunking and data
    /// shapes (runs, noise, all-default).
    #[test]
    fn parallel_rle_matches_sequential() {
        let mut data: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            data ^= data << 13;
            data ^= data >> 7;
            data ^= data << 17;
            data
        };
        let layout_for = |rows: usize, rb: usize, version: Version| PlaneLayout { planes: 1, width: rb, height: rows, depth: 8, version };
        for (rows, rb, version) in [
            (0usize, 4usize, Version::Psd),
            (1, 4, Version::Psd),
            (7, 4, Version::Psb),
            (64, 4, Version::Psd),
            (65, 4, Version::Psb),  // chunk boundary
            (513, 4, Version::Psd), // more rows than threads on most runners
            (2049, 4, Version::Psb),
            (2, 4, Version::Psd), // fewer rows than threads
            (33, 3000, Version::Psd),
            (31, 20000, Version::Psb),
        ] {
            let pattern: Vec<u8> = match rows % 3 {
                0 => vec![7u8; rows * rb],                                   // long runs
                1 => (0..rows * rb).map(|i| (i * 31 % 251) as u8).collect(), // noise-ish
                _ => vec![0u8; rows * rb],                                   // all default
            };
            let pattern = if pattern.len() != rows * rb { vec![7u8; rows * rb] } else { pattern };
            let mut data = pattern.clone();
            if rows % 4 == 0 {
                // sprinkle runs into the noise
                for chunk in data.chunks_mut(97) {
                    if let Some(f) = chunk.first_mut() {
                        *f = 200;
                    }
                }
            }
            let _ = next();
            let layout = layout_for(rows, rb, version);
            let parallel = encode_rle(&data, &layout).expect("parallel encode");
            let sequential = encode_rle_sequential(&data, &layout).expect("sequential encode");
            assert_eq!(parallel, sequential, "rows {rows} rb {rb}");
        }
    }

    /// The pre-parallel encoder, kept as the oracle.
    fn encode_rle_sequential(decoded: &[u8], layout: &PlaneLayout) -> Result<Vec<u8>> {
        let rows = layout.rows()?;
        let rb = layout.row_bytes();
        let cs = layout.count_size();
        let mut out = vec![0u8; rows * cs];
        let mut enc = Vec::new();
        for row in 0..rows {
            let start = enc.len();
            packbits::encode(&decoded[row * rb..(row + 1) * rb], &mut enc);
            let n = enc.len() - start;
            if cs == 2 {
                let n = u16::try_from(n).map_err(|_| PsdError::LimitExceeded("RLE row exceeds 65535 bytes (use PSB)"))?;
                out[row * 2..row * 2 + 2].copy_from_slice(&n.to_be_bytes());
            } else {
                let n = u32::try_from(n).map_err(|_| PsdError::LimitExceeded("RLE row too large"))?;
                out[row * 4..row * 4 + 4].copy_from_slice(&n.to_be_bytes());
            }
        }
        out.extend_from_slice(&enc);
        Ok(out)
    }
}
