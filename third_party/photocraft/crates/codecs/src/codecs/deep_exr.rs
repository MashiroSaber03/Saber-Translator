//! Deep OpenEXR (`deepscanline` / `deeptile`): a variable-length list of samples per pixel,
//! each with its own colour, alpha and depth. The `exr` crate rejects deep headers, so the
//! chunk stream is walked directly from the bytes, per the OpenEXR file-layout specification:
//! a deep chunk carries its coordinates, the packed size of the sample-count table, the
//! packed and unpacked sizes of the sample data, then the compressed count table (cumulative
//! within each row of the block) and the compressed sample data (row by row, each row channel
//! by channel). The layout was checked against files written by the OpenEXR library. Deep data permits only NONE, RLE, ZIPS and ZIP
//! compression; we support the first three (one scan line per block, which keeps the count
//! table unambiguous) and decline ZIP with `unsupported`.
//!
//! [`decode_deep`] returns the structured samples ([`DeepImage`]); [`flatten`] composites
//! them into a flat `Image` — samples sorted by Z, front-to-back `over` with premultiplied
//! colour, un-premultiplied into straight alpha (HDR values are kept, not clamped) — because the flat document model has no
//! per-pixel depth.

use std::io::{Cursor, Read, Seek, SeekFrom};

use exr::meta::attribute::{LevelMode, SampleType as ExrSample};
use exr::meta::{BlockDescription, MetaData};
use exr::prelude::Compression;
use half::f16;

use crate::Format;
use crate::error::CodecError;
use crate::image::{ChannelLayout, DecodeWarning, DeepChannel, DeepImage, Image, SampleType};
use crate::options::Limits;

const F: Format = Format::OpenExr;

fn err(e: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, e)
}

fn unsup(reason: impl Into<String>) -> CodecError {
    CodecError::unsupported(F, reason)
}

// --- checked little-endian reads -------------------------------------------------------------
// Every length and offset comes from the file, so every read is bounds-checked and a short
// read is a malformed file, never a panic.

fn read_exact(cur: &mut Cursor<&[u8]>, n: usize, what: &str) -> Result<Vec<u8>, CodecError> {
    let mut v = vec![0u8; n];
    cur.read_exact(&mut v).map_err(|_| err(format!("truncated file while reading {what}")))?;
    Ok(v)
}

fn i32(cur: &mut Cursor<&[u8]>) -> Result<i32, CodecError> {
    let b = read_exact(cur, 4, "an i32")?;
    Ok(i32::from_le_bytes([b.first().copied().unwrap_or(0), b.get(1).copied().unwrap_or(0), b.get(2).copied().unwrap_or(0), b.get(3).copied().unwrap_or(0)]))
}

fn u64(cur: &mut Cursor<&[u8]>) -> Result<u64, CodecError> {
    let b = read_exact(cur, 8, "a u64")?;
    let a: [u8; 8] = b.try_into().unwrap_or([0; 8]);
    Ok(u64::from_le_bytes(a))
}

fn seek_to(cur: &mut Cursor<&[u8]>, pos: u64) -> Result<(), CodecError> {
    if pos > cur.get_ref().len() as u64 {
        return Err(err(format!("chunk offset {pos} points past the end of the file")));
    }
    cur.seek(SeekFrom::Start(pos)).map_err(err)?;
    Ok(())
}

// --- decompression ----------------------------------------------------------------------------

/// Undo the ZIP predictor: each byte was stored as its difference to the previous byte plus
/// 128 (so a zero difference is 0x80), modulo 256. Applies to the sample-count table and the
/// sample data alike.
fn differences_to_samples(buf: &mut [u8]) {
    if let Some(&first) = buf.first() {
        let mut prev = i16::from(first);
        for b in buf.iter_mut().skip(1) {
            let s = (prev + i16::from(*b) - 128) as u8;
            prev = i16::from(s);
            *b = s;
        }
    }
}

/// Undo the ZIP half-split: the stream is `[first half][second half]` with the second half
/// holding every second byte of the original; weave them back together.
fn interleave_byte_blocks(buf: &mut [u8]) {
    let half = buf.len().div_ceil(2);
    let mut out = vec![0u8; buf.len()];
    let mut k = 0usize;
    for i in 0..buf.len() / 2 {
        let (a, b) = (buf.get(i).copied(), buf.get(half + i).copied());
        if let (Some(a), Some(b), Some(pair)) = (a, b, out.get_mut(k..k + 2)) {
            if let Some(byte) = pair.first_mut() {
                *byte = a;
            }
            if let Some(byte) = pair.get_mut(1) {
                *byte = b;
            }
        }
        k += 2;
    }
    if buf.len() % 2 == 1 && half > 0 {
        let last = buf.get(half - 1).copied();
        if let (Some(last), Some(slot)) = (last, out.get_mut(k)) {
            *slot = last;
        }
    }
    buf.copy_from_slice(&out);
}

/// OpenEXR RLE tokens: a negative byte `n` means `-n` literal bytes follow, a non-negative
/// byte `n` means the next byte repeats `n + 1` times (as `ImfRle.cpp` and the `exr` crate).
fn rle_decode(src: &[u8], expected: usize, what: &str) -> Result<Vec<u8>, CodecError> {
    let mut out = Vec::with_capacity(expected.min(1 << 20));
    let mut i = 0;
    while i < src.len() {
        let count = src[i] as i8;
        i += 1;
        let n = if count < 0 { (-(count as i32)) as usize } else { count as usize + 1 };
        if out.len().saturating_add(n) > expected {
            return Err(err(format!("{what}: RLE data expands past {expected} bytes")));
        }
        if count < 0 {
            let end = i.saturating_add(n);
            let run = src.get(i..end.min(src.len())).ok_or_else(|| err(format!("{what}: RLE run points past the compressed data")))?;
            if run.len() != n {
                return Err(err(format!("{what}: RLE run points past the compressed data")));
            }
            out.extend_from_slice(run);
            i = end;
        } else {
            let b = src.get(i).copied().ok_or_else(|| err(format!("{what}: RLE run points past the compressed data")))?;
            i += 1;
            out.resize(out.len() + n, b);
        }
    }
    if out.len() != expected {
        return Err(err(format!("{what}: RLE data expands to {} bytes, expected {expected}", out.len())));
    }
    Ok(out)
}

/// Decompress `what` to exactly `expected` bytes. Like flat blocks, a payload that already
/// has the uncompressed length is stored raw whatever the declared compression.
fn decompress(compression: Compression, packed: &[u8], expected: usize, what: &str) -> Result<Vec<u8>, CodecError> {
    if packed.len() == expected {
        return Ok(packed.to_vec());
    }
    match compression {
        Compression::Uncompressed => Err(err(format!("{what}: expected {expected} uncompressed bytes, found {}", packed.len()))),
        Compression::RLE => {
            // Like ZIPS, RLE runs over the predicted, byte-split stream.
            let mut out = rle_decode(packed, expected, what)?;
            differences_to_samples(&mut out);
            interleave_byte_blocks(&mut out);
            Ok(out)
        }
        Compression::ZIP1 => {
            let mut out = Vec::with_capacity(expected.min(1 << 20));
            flate2::read::ZlibDecoder::new(packed)
                .take(expected as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|_| err(format!("{what}: zlib data is malformed")))?;
            if out.len() != expected {
                return Err(err(format!("{what}: zlib data expands to {} bytes, expected {expected}", out.len())));
            }
            differences_to_samples(&mut out);
            interleave_byte_blocks(&mut out);
            Ok(out)
        }
        other => Err(unsup(format!("deep data with {other:?} compression (NONE, RLE and ZIPS are supported)"))),
    }
}

// --- the deep chunk stream ---------------------------------------------------------------------

/// One parsed chunk: where its pixels go and its cumulative sample counts, plus where to
/// find the compressed sample data in the second pass.
struct DeepChunk {
    /// Global pixel indices of the block's pixels, in block order (row-major within the block).
    pixels: Vec<usize>,
    /// Pixels per row of the block (the data window width for a scan line, the clipped tile
    /// width for a tile).
    row_w: usize,
    /// Cumulative sample count table of the block: `len == pixels.len() + 1`.
    prefix: Vec<u32>,
    /// File offset of the chunk start.
    pos: u64,
    /// Size of the compressed sample data in the file.
    packed_data: u64,
    /// The decompressed sample-data size the file declares (validated against the table).
    unpacked_data: u64,
}

fn sample_bytes(t: ExrSample) -> usize {
    match t {
        ExrSample::U32 | ExrSample::F32 => 4,
        ExrSample::F16 => 2,
    }
}

fn our_sample_type(t: ExrSample) -> SampleType {
    match t {
        ExrSample::F16 => SampleType::F16,
        ExrSample::U32 | ExrSample::F32 => SampleType::F32,
    }
}

/// Read the deep samples of the first deep part. `meta` must come from an unvalidated
/// [`MetaData::read_from_buffered`] of the same bytes (deep headers fail validation).
pub(crate) fn decode_deep(meta: &MetaData, bytes: &[u8], limits: &Limits) -> Result<DeepImage, CodecError> {
    let header_index = meta.headers.iter().position(|h| h.deep).ok_or_else(|| unsup("the file has no deep part"))?;
    let header = meta.headers.get(header_index).ok_or_else(|| unsup("the file has no deep part"))?;
    let compression = header.compression;
    if !matches!(compression, Compression::Uncompressed | Compression::RLE | Compression::ZIP1) {
        return Err(unsup(format!("deep data with {compression:?} compression (NONE, RLE and ZIPS are supported)")));
    }

    let dw = header.data_window();
    let (w, h) = (dw.size.0, dw.size.1);
    if w == 0 || h == 0 {
        return Err(err("the data window is empty"));
    }
    if w > u32::MAX as usize || h > u32::MAX as usize {
        return Err(err("the data window is too large"));
    }
    let y0 = dw.position.1;
    let npixels = w.checked_mul(h).ok_or_else(|| err("the data window is too large"))?;
    limits.check_bytes(w as u32, h as u32, (header.channels.list.len().max(1) as u64) * 4)?;

    // The display window is the full stage the renderer composed onto (a render pass's data
    // window is often its crop). Open on the stage when it plausibly holds the data region:
    // sane size for the limits, and the data region inside it. Otherwise fall back to the
    // data window alone, as before.
    let display = header.shared_attributes.display_window;
    let display_size = (display.size.0.max(1) as u32, display.size.1.max(1) as u32);
    let (stage, origin) = match display_size {
        (sw, sh)
            if sw <= 1 << 15
                && sh <= 1 << 15
                && limits.check_bytes(sw, sh, 4).is_ok()
                && dw.position.0 >= display.position.0
                && dw.position.1 >= display.position.1
                && dw.position.0.saturating_add(w as i32) <= display.position.0.saturating_add(sw as i32)
                && dw.position.1.saturating_add(h as i32) <= display.position.1.saturating_add(sh as i32) =>
        {
            (Some((sw, sh)), (dw.position.0 - display.position.0, dw.position.1 - display.position.1))
        }
        _ => (None, (0, 0)),
    };

    let exr_types: Vec<ExrSample> = header.channels.list.iter().map(|c| c.sample_type).collect();
    let bytes_per_sample: usize = exr_types.iter().map(|t| sample_bytes(*t)).sum();
    let multipart = meta.requirements.has_multiple_layers;

    // The offset tables follow the headers: `chunk_count` u64 per header, in header order.
    // Our part's table starts after the tables of all earlier parts.
    let mut table_pos = header_end(bytes)?;
    for h_ in meta.headers.iter().take(header_index) {
        table_pos = table_pos.saturating_add((h_.chunk_count.min(1 << 24) as u64).saturating_mul(8));
    }
    let mut cur = Cursor::new(bytes);
    seek_to(&mut cur, table_pos)?;
    let mut offsets = Vec::with_capacity(header.chunk_count.min(1 << 20));
    for _ in 0..header.chunk_count {
        offsets.push(u64(&mut cur)?);
    }
    let tables_end = cur.position();

    // Block geometry.
    let (tiles_x, tile_w, tile_h, slots) = match &header.blocks {
        BlockDescription::ScanLines => (0usize, w, 1usize, h), // one line per block for NONE/RLE/ZIPS
        BlockDescription::Tiles(t) => {
            if t.level_mode != LevelMode::Singular {
                return Err(unsup("deep mip-map or rip-map tiles (single-level tiles are supported)"));
            }
            let (tw, th) = (t.tile_size.0.max(1), t.tile_size.1.max(1));
            let tx = w.saturating_add(tw - 1) / tw;
            let ty = h.saturating_add(th - 1) / th;
            (tx, tw, th, tx.checked_mul(ty).ok_or_else(|| err("tile count overflows"))?)
        }
    };

    let mut counts = vec![0u32; npixels];
    let mut seen = vec![false; slots];
    let mut chunks: Vec<DeepChunk> = Vec::with_capacity(offsets.len());

    for &off in &offsets {
        if off < tables_end || off >= bytes.len() as u64 {
            return Err(err(format!("chunk offset {off} is outside the chunk area of the file")));
        }
        seek_to(&mut cur, off)?;
        if multipart {
            let part = i32(&mut cur)?;
            if part < 0 || part as usize != header_index {
                return Err(err(format!("chunk declares part {part}, expected {header_index}")));
            }
        }
        // Block coordinates → the block's global pixel range and its slot.
        let (block_pixels, slot, row_w): (Vec<usize>, usize, usize) = match &header.blocks {
            BlockDescription::ScanLines => {
                let y = i32(&mut cur)?;
                let row = match y.checked_sub(y0) {
                    Some(d) if d >= 0 && (d as usize) < h => d as usize,
                    _ => return Err(err(format!("scan line block y={y} is outside the data window"))),
                };
                ((0..w).map(|x| row * w + x).collect(), row, w)
            }
            BlockDescription::Tiles(_) => {
                let (tx, ty, lx, ly) = (i32(&mut cur)?, i32(&mut cur)?, i32(&mut cur)?, i32(&mut cur)?);
                if (lx, ly) != (0, 0) {
                    return Err(err(format!("deep tile at level ({lx},{ly}); only level 0 exists in a single-level file")));
                }
                if tx < 0 || ty < 0 || (tx as usize) >= tiles_x || (ty as usize) * tile_h >= h {
                    return Err(err(format!("tile ({tx},{ty}) is outside the data window")));
                }
                let (bx, by) = ((tx as usize) * tile_w, (ty as usize) * tile_h);
                let rows = (by + tile_h).min(h);
                let cols = (bx + tile_w).min(w);
                let mut px = Vec::with_capacity((rows - by) * (cols - bx));
                for yy in by..rows {
                    for xx in bx..cols {
                        px.push(yy.saturating_mul(w).saturating_add(xx));
                    }
                }
                let slot = (ty as usize).saturating_mul(tiles_x) + tx as usize;
                (px, slot, cols - bx)
            }
        };
        if seen.get(slot).copied().unwrap_or(true) {
            return Err(err(format!("two chunks cover block {slot}")));
        }
        if let Some(s) = seen.get_mut(slot) {
            *s = true;
        }

        let packed_table = u64(&mut cur)?;
        let packed_data = u64(&mut cur)?;
        let unpacked_data = u64(&mut cur)?;
        let remaining = (bytes.len() as u64).saturating_sub(cur.position());
        if packed_table > remaining || packed_data > remaining.saturating_sub(packed_table) {
            return Err(err("deep chunk sizes point past the end of the file"));
        }
        let table_bytes = read_exact(&mut cur, packed_table as usize, "the sample-count table")?;
        let expected_table = block_pixels.len().checked_mul(4).ok_or_else(|| err("count table overflows"))?;
        // An edge tile's table may still be stored at the full tile size (its pixels first, in
        // rows of the clipped width, then unused entries), as the OpenEXR library writes it.
        let full_table = tile_w.saturating_mul(tile_h).saturating_mul(4).max(expected_table);
        let raw = decompress(compression, &table_bytes, full_table, "the sample-count table")
            .or_else(|_| decompress(compression, &table_bytes, expected_table, "the sample-count table"))?;

        // The table is cumulative within each row of the block (a tile restarts at every row);
        // `prefix` is cumulative over the whole block, and each pixel's own count is the difference.
        let mut prefix = Vec::with_capacity(block_pixels.len() + 1);
        prefix.push(0u32);
        let (mut row_base, mut prev) = (0u32, 0u32);
        for (i, p) in block_pixels.iter().enumerate() {
            let b: [u8; 4] = raw.get(i * 4..i * 4 + 4).and_then(|s| s.try_into().ok()).ok_or_else(|| err("the count table is short"))?;
            let v = u32::from_le_bytes(b);
            if row_w > 0 && i % row_w == 0 {
                row_base = row_base.checked_add(prev).ok_or_else(|| err("the block's sample count overflows"))?;
                prev = 0;
            }
            if v < prev {
                return Err(err("the sample-count table decreases (malformed cumulative counts)"));
            }
            if let Some(c) = counts.get_mut(*p) {
                *c = v - prev;
            }
            prev = v;
            prefix.push(row_base.checked_add(v).ok_or_else(|| err("the block's sample count overflows"))?);
        }
        let total = prefix.last().copied().unwrap_or(0) as u64;
        let expected_data = total.checked_mul(bytes_per_sample as u64).ok_or_else(|| err("the sample data size overflows"))?;
        if expected_data != unpacked_data {
            return Err(err(format!("the chunk declares {unpacked_data} bytes of sample data, the count table implies {expected_data}")));
        }
        chunks.push(DeepChunk { pixels: block_pixels, row_w, prefix, pos: off, packed_data, unpacked_data });
    }
    if seen.iter().any(|s| !s) {
        let missing = seen.iter().filter(|s| !**s).count();
        return Err(err(format!("{missing} of {slots} chunks are missing from the offset table")));
    }

    // Global cumulative counts in raster order.
    let mut deep_counts = Vec::with_capacity(npixels + 1);
    deep_counts.push(0u64);
    let mut acc = 0u64;
    for &c in &counts {
        acc = acc.checked_add(c as u64).ok_or_else(|| err("the total sample count overflows"))?;
        deep_counts.push(acc);
    }
    let total_samples = acc;
    let sample_bytes_total = total_samples.checked_mul(exr_types.len() as u64 * 4).ok_or_else(|| err("the sample buffer size overflows"))?;
    if sample_bytes_total > limits.max_alloc {
        return Err(CodecError::LimitExceeded(format!("deep samples need {sample_bytes_total} bytes, exceeding max_alloc {}", limits.max_alloc)));
    }

    let mut channels: Vec<DeepChannel> = header
        .channels
        .list
        .iter()
        .map(|c| DeepChannel { name: c.name.to_string(), sample: our_sample_type(c.sample_type), samples: vec![0f32; total_samples as usize] })
        .collect();

    // Second pass: the compressed sample data, scattered per pixel (a block's channel data
    // is contiguous in the file but the pixels' samples interleave with other blocks').
    for chunk in &chunks {
        seek_to(&mut cur, chunk.pos)?;
        if multipart {
            let _part = i32(&mut cur)?;
        }
        match &header.blocks {
            BlockDescription::ScanLines => {
                let _y = i32(&mut cur)?;
            }
            BlockDescription::Tiles(_) => {
                let _coords = (i32(&mut cur)?, i32(&mut cur)?, i32(&mut cur)?, i32(&mut cur)?);
            }
        }
        let packed_table = u64(&mut cur)?;
        let _packed_data = u64(&mut cur)?;
        let _unpacked_data = u64(&mut cur)?;
        // The count table sits between the sizes and the sample data; skip its bytes.
        read_exact(&mut cur, packed_table as usize, "the sample-count table")?;
        let data_bytes = read_exact(&mut cur, chunk.packed_data as usize, "the deep sample data")?;
        let raw = decompress(compression, &data_bytes, chunk.unpacked_data as usize, "the deep sample data")?;
        // Row by row (a scan line block is one row), and within a row channel by channel, as
        // flat tiles are laid out: each pixel's samples are contiguous within its channel.
        let mut in_off = 0usize;
        let row_w = chunk.row_w.max(1);
        for (r, row) in chunk.pixels.chunks(row_w).enumerate() {
            let i0 = r * row_w;
            let row_from = chunk.prefix.get(i0).copied().unwrap_or(0) as usize;
            let row_to = chunk.prefix.get(i0 + row.len()).copied().unwrap_or(0) as usize;
            let row_total = row_to.saturating_sub(row_from);
            for (ci, t) in exr_types.iter().enumerate() {
                let sb = sample_bytes(*t);
                let ch_bytes = row_total.checked_mul(sb).ok_or_else(|| err("channel size overflows"))?;
                let end = in_off.checked_add(ch_bytes).ok_or_else(|| err("channel size overflows"))?;
                let src = raw.get(in_off..end).ok_or_else(|| err("the sample data is short"))?;
                in_off = end;
                let out = channels.get_mut(ci).ok_or_else(|| err("channel index out of range"))?;
                for (j, p) in row.iter().enumerate() {
                    let from = (chunk.prefix.get(i0 + j).copied().unwrap_or(0) as usize).saturating_sub(row_from);
                    let n = (chunk.prefix.get(i0 + j + 1).copied().unwrap_or(0) as usize).saturating_sub(row_from).saturating_sub(from);
                    let to = deep_counts.get(*p).copied().unwrap_or(0) as usize;
                    for k in 0..n {
                        let base = (from + k) * sb;
                        let v = match t {
                            ExrSample::F16 => src.get(base..base + 2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()),
                            ExrSample::F32 => src.get(base..base + 4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                            ExrSample::U32 => src.get(base..base + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32),
                        };
                        let (Some(v), Some(slot)) = (v, out.samples.get_mut(to + k)) else {
                            return Err(err("the sample data is short"));
                        };
                        *slot = v;
                    }
                }
            }
        }
    }

    Ok(DeepImage { width: w as u32, height: h as u32, stage, origin, channels, counts: deep_counts })
}

/// Where the headers end and the offset tables begin. [`MetaData::read_from_buffered`]
/// consumes exactly the version field and the headers, so re-reading with a fresh cursor
/// and taking its end position yields it.
fn header_end(bytes: &[u8]) -> Result<u64, CodecError> {
    let mut cur = Cursor::new(bytes);
    MetaData::read_from_buffered(&mut cur, false).map_err(err)?;
    Ok(cur.position())
}

/// Composite the deep samples into a flat image: per pixel, samples sorted by Z (stable; the
/// file is assumed messy), front-to-back `over` with premultiplied colour, un-premultiplied
/// into straight alpha at the end. Without any alpha channel the front-most sample wins,
/// opaque. Z, ZBack and other deep channels influence the flat result only through the
/// ordering; volume samples (with ZBack) are approximated as point samples.
pub(crate) fn flatten(deep: &DeepImage) -> Result<Image, CodecError> {
    let npx = deep.width as usize * deep.height as usize;
    let has = |n: &str| deep.channel(n).is_some();
    let rgb = has("R") && has("G") && has("B");
    let gray = !rgb && (has("Y") || deep.channels.len() == 1);
    let layout = if rgb || gray {
        if has("A") || has("AR") || has("AG") || has("AB") {
            if rgb { ChannelLayout::Rgba } else { ChannelLayout::GrayA }
        } else if rgb {
            ChannelLayout::Rgb
        } else {
            ChannelLayout::Gray
        }
    } else {
        return Err(CodecError::unsupported(F, "no R/G/B or Y channels in the deep part"));
    };
    let nc = layout.channels();
    let samples = |n: &str| -> &[f32] { deep.channel(n).map(|c| c.samples.as_slice()).unwrap_or(&[]) };
    let (r, g, b) = (samples("R"), samples("G"), samples("B"));
    let (ar, ag, ab) = (samples("AR"), samples("AG"), samples("AB"));
    let (a, z) = (samples("A"), samples("Z"));
    let y = samples("Y");
    let lum = if y.is_empty() { deep.channels.first().map(|c| c.samples.as_slice()).unwrap_or(&[]) } else { y };

    let mut out = vec![0f32; npx * nc];
    let mut order: Vec<u32> = Vec::new();
    let mut max_samples = 0u32;
    for p in 0..npx {
        let range = deep.sample_range(p);
        let (s0, n) = (range.start as usize, (range.end - range.start) as usize);
        max_samples = max_samples.max(u32::try_from(n).unwrap_or(u32::MAX));
        if n == 0 {
            continue; // no samples: transparent black
        }
        order.clear();
        order.extend(0u32..n as u32);
        if !z.is_empty() {
            // Stable sort by Z; a NaN depth (invalid) sorts last, deterministically.
            order.sort_by(|&i, &j| {
                let zi = z.get(s0 + i as usize).copied().unwrap_or(f32::NAN);
                let zj = z.get(s0 + j as usize).copied().unwrap_or(f32::NAN);
                zi.total_cmp(&zj)
            });
        }
        let at = |ch: &[f32], i: u32| ch.get(s0 + i as usize).copied().unwrap_or(0.0);
        // Per-colour-channel alpha: AR/AG/AB when present, else A, else opaque.
        let alpha = |i: u32| -> [f32; 3] {
            let s = at(a, i);
            [
                if !a.is_empty() && ar.is_empty() {
                    s
                } else if !ar.is_empty() {
                    at(ar, i)
                } else {
                    1.0
                },
                if !a.is_empty() && ag.is_empty() {
                    s
                } else if !ag.is_empty() {
                    at(ag, i)
                } else {
                    1.0
                },
                if !a.is_empty() && ab.is_empty() {
                    s
                } else if !ab.is_empty() {
                    at(ab, i)
                } else {
                    1.0
                },
            ]
        };
        let mut cacc = [0f32; 3]; // premultiplied colour
        let mut aacc = [0f32; 3]; // accumulated alpha
        for &i in &order {
            let ai = alpha(i);
            let src = if rgb { [at(r, i), at(g, i), at(b, i)] } else { [at(lum, i); 3] };
            let mut covered = true;
            for k in 0..3 {
                let t = (1.0 - aacc[k]).max(0.0);
                cacc[k] += src[k] * t;
                aacc[k] += t * ai[k];
                covered &= aacc[k] >= 1.0;
            }
            if covered {
                break; // everything behind is hidden
            }
        }
        let a_out = if a.is_empty() && ar.is_empty() && ag.is_empty() && ab.is_empty() { 1.0 } else { (aacc[0] + aacc[1] + aacc[2]) / 3.0 };
        let a_clamped = a_out.clamp(0.0, 1.0);
        // Un-premultiply into straight alpha (the deep colours are premultiplied).
        let straight = |c: f32| if a_clamped > 1e-6 { c / a_clamped } else { 0.0 };
        let base = p * nc;
        // Written by layout: RGB(A), gray(+A) — a fixed RGBA quadruple would put a luminance
        // copy into a GrayA image's alpha slot.
        // A fixed array plus a length: this runs once per pixel, so no allocation.
        let (vals, len) = match (rgb, nc) {
            (true, 4) => ([straight(cacc[0]), straight(cacc[1]), straight(cacc[2]), a_clamped], 4),
            (true, _) => ([straight(cacc[0]), straight(cacc[1]), straight(cacc[2]), 0.0], 3),
            (false, 2) => ([straight(cacc[0]), a_clamped, 0.0, 0.0], 2),
            (false, _) => ([straight(cacc[0]), 0.0, 0.0, 0.0], 1),
        };
        for (k, v) in vals.into_iter().take(len).enumerate() {
            if let Some(slot) = out.get_mut(base + k) {
                *slot = v;
            }
        }
    }
    // Open on the full display-window stage when the file declares one: the renderer's data
    // window is its crop, and the composition lives on the whole stage. The composed region
    // pads to the stage at its offset, transparent elsewhere (limits validated the size).
    let mut img = match deep.stage {
        Some((sw, sh)) => {
            let mut padded = vec![0f32; sw as usize * sh as usize * nc];
            let row_len = deep.width as usize * nc;
            let dx = deep.origin.0.max(0) as usize;
            for row in 0..deep.height as usize {
                let src = row * row_len;
                let dst = (row + deep.origin.1.max(0) as usize) * sw as usize + dx;
                if let (Some(d), Some(r)) = (padded.get_mut(dst * nc..dst * nc + row_len), out.get(src..src + row_len)) {
                    d.copy_from_slice(r);
                }
            }
            Image::from_f32(sw, sh, layout, &padded)?
        }
        None => Image::from_f32(deep.width, deep.height, layout, &out)?,
    };
    if let Some((sw, sh)) = deep.stage {
        img.warnings.push(DecodeWarning::DisplayWindowStage { stage: (sw, sh), origin: deep.origin });
    }
    img.warnings.push(DecodeWarning::DeepFlattened { format: F, max_samples_per_pixel: max_samples });
    Ok(img)
}
