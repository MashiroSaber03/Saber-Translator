//! PNG via the `png` crate directly: 8/16-bit, Adam7 read (and write),
//! iCCP, eXIf, tEXt/zTXt/iTXt (XMP in `XML:com.adobe.xmp`), pHYs.

use std::io::{Cursor, Write};

use crate::Format;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, DecodeWarning, Image, Metadata, SampleType};
use crate::options::{EncodeOptions, Limits, PngCompression, alloc_zeroed};

const F: Format = Format::Png;
pub(crate) const XMP_KEYWORD: &str = "XML:com.adobe.xmp";
const METERS_PER_INCH: f32 = 0.0254;

fn err(e: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, e)
}

/// Pixels per inch for `pHYs` pixels per metre. A whole number of ppi is stored rounded to whole
/// pixels per metre (300 ppi = 11811), so reading it back gives 299.9994: the whole number that
/// rounds to the stored value is the resolution that was meant, as Photoshop reads it.
pub(crate) fn dpi_of_ppm(ppm: u32) -> f32 {
    let dpi = f64::from(ppm) * f64::from(METERS_PER_INCH);
    let whole = dpi.round();
    if (whole / f64::from(METERS_PER_INCH)).round() == f64::from(ppm) { whole as f32 } else { dpi as f32 }
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    let mut input = Cursor::new(bytes);
    let mut decoder = png::Decoder::new_with_limits(&mut input, png::Limits { bytes: limits.alloc_usize() });
    decoder.set_transformations(png::Transformations::EXPAND);
    // png's lazy text get_text() inflates without consulting its allocation limit.
    // Read text separately below, without retaining compressed copies in the decoder.
    decoder.set_ignore_text_chunk(true);
    {
        let info = decoder.read_header_info().map_err(map_png_err)?;
        let (w, h) = info.size();
        // Worst case output: 4 channels x 16 bit.
        limits.check_bytes(w, h, 8)?;
        // zlib never beats 1032:1; refuse a file too small for the declared image before the
        // output buffer exists. Measured on the stored samples (before palette/bit expansion).
        let stored = u64::from(w).saturating_mul(u64::from(h)).saturating_mul(info.bits_per_pixel() as u64).div_ceil(8);
        Limits::check_plausible(stored, bytes.len(), 1100)?;
    }
    let mut reader = decoder.read_info().map_err(map_png_err)?;
    // APNG: the IDAT image is the first frame, or an extra default image when no fcTL precedes it.
    let images = reader.info().animation_control.map_or(1, |a| u64::from(a.num_frames) + u64::from(reader.info().frame_control.is_none()));
    let (color, depth) = reader.output_color_type();
    let layout = match color {
        png::ColorType::Grayscale => ChannelLayout::Gray,
        png::ColorType::GrayscaleAlpha => ChannelLayout::GrayA,
        png::ColorType::Rgb => ChannelLayout::Rgb,
        png::ColorType::Rgba => ChannelLayout::Rgba,
        png::ColorType::Indexed => return Err(err("palette was not expanded")),
    };
    let sample = match depth {
        png::BitDepth::Eight => SampleType::U8,
        png::BitDepth::Sixteen => SampleType::U16,
        d => return Err(err(format!("unexpected output bit depth {d:?}"))),
    };
    let size = reader.output_buffer_size().ok_or_else(|| err("image too large"))?;
    let mut buf = alloc_zeroed::<u8>(size)?;
    let out = reader.next_frame(&mut buf).map_err(map_png_err)?;
    buf.truncate(out.buffer_size());
    // Collect ancillary chunks after IDAT. Tolerate malformed tails once pixels
    // are complete, but never swallow an allocation-limit failure.
    if let Err(e @ png::DecodingError::LimitsExceeded) = reader.finish() {
        return Err(map_png_err(e));
    }
    if sample == SampleType::U16 {
        for c in buf.as_chunks_mut::<2>().0 {
            let v = u16::from_be_bytes([c[0], c[1]]);
            c.copy_from_slice(&v.to_ne_bytes());
        }
    }
    let info = reader.info();
    let mut img = Image::from_raw(out.width, out.height, layout, sample, buf)?;
    img.icc = info.icc_profile.as_ref().map(|c| c.to_vec());
    let mut meta = Metadata { exif: info.exif_metadata.as_ref().map(|c| c.to_vec()), ..Default::default() };
    if let Some(d) = info.pixel_dims
        && d.unit == png::Unit::Meter
        && d.xppu > 0
        && d.yppu > 0
    {
        meta.dpi = Some((dpi_of_ppm(d.xppu), dpi_of_ppm(d.yppu)));
    }
    drop(reader);
    let end = usize::try_from(input.position()).map_err(|_| err("invalid PNG reader position"))?;
    let consumed = bytes.get(..end).ok_or_else(|| err("invalid PNG reader position"))?;
    read_text_metadata(consumed, limits.alloc_usize(), &mut meta)?;
    img.meta = meta;
    if images > 1 {
        img.warnings.push(DecodeWarning::MoreFrames { total: u32::try_from(images).ok() });
    }
    Ok(img)
}

/// The colour table of a palette PNG (`PLTE`) and the one entry its `tRNS` makes transparent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngPalette {
    pub colors: Vec<[u8; 3]>,
    pub transparent: Option<u8>,
}

/// The colour table of a palette PNG (colour type 3), read from the headers without decoding
/// pixels. `None` for other colour types, and for a `tRNS` with partial alpha or several
/// transparent entries: an Indexed Color document holds one transparent entry.
pub fn palette(bytes: &[u8]) -> Result<Option<PngPalette>, CodecError> {
    let mut decoder = png::Decoder::new_with_limits(Cursor::new(bytes), png::Limits::default());
    decoder.set_ignore_text_chunk(true);
    let reader = decoder.read_info().map_err(map_png_err)?;
    let info = reader.info();
    if info.color_type != png::ColorType::Indexed {
        return Ok(None);
    }
    let Some(plte) = info.palette.as_deref() else { return Ok(None) };
    let colors = plte.as_chunks::<3>().0.to_vec();
    let mut transparent = None;
    for (i, &a) in info.trns.as_deref().unwrap_or_default().iter().enumerate() {
        match a {
            255 => {}
            0 if transparent.is_none() && i < colors.len() => transparent = u8::try_from(i).ok(),
            _ => return Ok(None),
        }
    }
    Ok(Some(PngPalette { colors, transparent }))
}

fn map_png_err(e: png::DecodingError) -> CodecError {
    match e {
        png::DecodingError::LimitsExceeded => CodecError::LimitExceeded("PNG decoder memory limit".into()),
        e => err(e),
    }
}

fn text_limit() -> CodecError {
    CodecError::LimitExceeded("PNG decoded text metadata exceeds max_alloc".into())
}

fn split_nul(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let at = bytes.iter().position(|&b| b == 0)?;
    Some((bytes.get(..at)?, bytes.get(at.checked_add(1)?..)?))
}

fn latin1_len(bytes: &[u8]) -> Option<usize> {
    bytes.len().checked_add(bytes.iter().filter(|&&b| b >= 128).count())
}

/// Read only CRC-valid text chunks, including those after IDAT, with one shared
/// budget for decoded UTF-8 keywords and values (including overwritten XMP).
/// Invalid ancillary text stays optional, as it was with png's text decoder.
fn read_text_metadata(bytes: &[u8], mut remaining: usize, meta: &mut Metadata) -> Result<(), CodecError> {
    let Some(mut rest) = bytes.get(8..) else { return Ok(()) };
    let (mut plain, mut compressed, mut international) = (Vec::new(), Vec::new(), Vec::new());
    while let Some((length, tail)) = rest.split_first_chunk::<4>() {
        let Some((kind, tail)) = tail.split_first_chunk::<4>() else { break };
        let length = u32::from_be_bytes(*length);
        // PNG chunk lengths are at most 2^31-1. Broken tails remain tolerated.
        if length > i32::MAX as u32 {
            break;
        }
        let Ok(length) = usize::try_from(length) else { break };
        let Some(data) = tail.get(..length) else { break };
        let Some((checksum, next)) = tail.get(length..).and_then(|t| t.split_first_chunk::<4>()) else { break };
        rest = next;
        if kind == b"IEND" {
            break;
        }
        if !matches!(kind, b"tEXt" | b"zTXt" | b"iTXt") {
            continue;
        }
        let mut crc = flate2::Crc::new();
        crc.update(kind);
        crc.update(data);
        if crc.sum() != u32::from_be_bytes(*checksum) {
            continue;
        }
        let Some((keyword, value)) = split_nul(data) else { continue };
        if keyword.is_empty() || keyword.len() > 79 {
            continue;
        }
        let (value, deflated, latin1) = match kind {
            b"tEXt" => (value, false, true),
            b"zTXt" => {
                let Some((&0, value)) = value.split_first() else { continue };
                (value, true, true)
            }
            _ => {
                let Some(([flag, method], value)) = value.split_first_chunk::<2>() else { continue };
                if *flag > 1 || (*flag == 1 && *method != 0) {
                    continue;
                }
                let Some((language, value)) = split_nul(value) else { continue };
                let Some((translated, value)) = split_nul(value) else { continue };
                if !language.is_ascii() || std::str::from_utf8(translated).is_err() {
                    continue;
                }
                (value, *flag == 1, false)
            }
        };
        let keyword_len = latin1_len(keyword).ok_or_else(text_limit)?;
        let limit = remaining.saturating_sub(keyword_len);
        let Some(text) = bounded_text(value, deflated, latin1, limit)? else { continue };
        remaining = remaining.checked_sub(keyword_len).and_then(|left| left.checked_sub(text.len())).ok_or_else(text_limit)?;
        let keyword: String = keyword.iter().map(|&b| char::from(b)).collect();
        match kind {
            // Some writers put XMP in Latin-1 text chunks instead of the usual iTXt.
            // Treat the reserved keyword consistently, after the same budget checks.
            _ if keyword == XMP_KEYWORD => meta.xmp = Some(text),
            b"tEXt" => plain.push((keyword, text)),
            b"zTXt" => compressed.push((keyword, text)),
            _ => international.push((keyword, text)),
        }
    }
    // Preserve the previous decoder's grouping of text entries by chunk type.
    meta.text.extend(plain);
    meta.text.extend(compressed);
    meta.text.extend(international);
    Ok(())
}

/// No input-sized allocation until its decoded size fits. Inflation uses a
/// fixed scratch buffer and checks each batch before growing the output.
fn bounded_text(input: &[u8], compressed: bool, latin1: bool, limit: usize) -> Result<Option<String>, CodecError> {
    let mut output = Vec::new();
    let mut append = |bytes: &[u8]| -> Result<(), CodecError> {
        let added = if latin1 { latin1_len(bytes).ok_or_else(text_limit)? } else { bytes.len() };
        if output.len().checked_add(added).is_none_or(|len| len > limit) {
            return Err(text_limit());
        }
        output.try_reserve(added).map_err(|_| text_limit())?;
        if latin1 {
            // Every Latin-1 code point fits in at most two UTF-8 bytes.
            let mut utf8 = [0; 2];
            for &b in bytes {
                output.extend_from_slice(char::from(b).encode_utf8(&mut utf8).as_bytes());
            }
        } else {
            output.extend_from_slice(bytes);
        }
        Ok(())
    };
    if compressed {
        let mut decoder = flate2::Decompress::new(true);
        let mut scratch = [0; 8192];
        loop {
            let start_in = decoder.total_in();
            let start_out = decoder.total_out();
            let Some(tail) = usize::try_from(start_in).ok().and_then(|at| input.get(at..)) else { return Ok(None) };
            let Ok(status) = decoder.decompress(tail, &mut scratch, flate2::FlushDecompress::None) else { return Ok(None) };
            let Some(batch) = usize::try_from(decoder.total_out() - start_out).ok().and_then(|len| scratch.get(..len)) else { return Ok(None) };
            append(batch)?;
            if status == flate2::Status::StreamEnd {
                break;
            }
            if start_in == decoder.total_in() && start_out == decoder.total_out() {
                return Ok(None); // Truncated stream: never keep partial text.
            }
        }
    } else {
        if !latin1 && std::str::from_utf8(input).is_err() {
            return Ok(None);
        }
        append(input)?;
    }
    Ok(String::from_utf8(output).ok())
}

fn is_latin1_keyword(k: &str) -> bool {
    !k.is_empty() && k.len() <= 79 && k.chars().all(|c| (' '..='~').contains(&c) || ('\u{a1}'..='\u{ff}').contains(&c))
}

pub(crate) fn encode(src: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let img = src.converted(plan.layout, plan.sample);
    let (w, h) = img.dimensions();
    let color = match img.layout() {
        ChannelLayout::Gray => png::ColorType::Grayscale,
        ChannelLayout::GrayA => png::ColorType::GrayscaleAlpha,
        ChannelLayout::Rgb => png::ColorType::Rgb,
        ChannelLayout::Rgba => png::ColorType::Rgba,
        l => return Err(CodecError::encode(F, format!("unsupported layout {l:?}"))),
    };
    let depth = match img.sample_type() {
        SampleType::U8 => png::BitDepth::Eight,
        SampleType::U16 => png::BitDepth::Sixteen,
        s => return Err(CodecError::encode(F, format!("unsupported sample {s:?}"))),
    };
    // Big-endian sample bytes.
    let mut data = std::borrow::Cow::Borrowed(img.data());
    if depth == png::BitDepth::Sixteen {
        for c in data.to_mut().as_chunks_mut::<2>().0 {
            let v = u16::from_ne_bytes([c[0], c[1]]);
            c.copy_from_slice(&v.to_be_bytes());
        }
    }

    let mut info = png::Info::with_size(w, h);
    info.color_type = color;
    info.bit_depth = depth;
    info.interlaced = opts.png_interlaced;
    if opts.embed_icc
        && let Some(icc) = &img.icc
    {
        info.icc_profile = Some(icc.clone().into());
    }
    if opts.embed_metadata {
        if let Some(exif) = &img.meta.exif {
            // The pixels are written as they are shown: never let a viewer rotate them again.
            info.exif_metadata = Some(crate::resolution::export_exif(exif, img.meta.dpi).into_owned().into());
        }
        if let Some((x, y)) = img.meta.dpi
            && x > 0.0
            && y > 0.0
        {
            info.pixel_dims =
                Some(png::PixelDimensions { xppu: (x / METERS_PER_INCH).round() as u32, yppu: (y / METERS_PER_INCH).round() as u32, unit: png::Unit::Meter });
        }
    }

    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::with_info(&mut out, info).map_err(|e| CodecError::encode(F, e))?;
        encoder.set_compression(match opts.png_compression {
            PngCompression::None => png::Compression::NoCompression,
            PngCompression::Fast => png::Compression::Fast,
            PngCompression::Default => png::Compression::Balanced,
            PngCompression::Best => png::Compression::High,
        });
        if opts.embed_metadata {
            for (k, v) in &img.meta.text {
                let latin1_text = v.chars().all(|c| (c as u32) < 256);
                let res = if is_latin1_keyword(k) && latin1_text {
                    encoder.add_text_chunk(k.clone(), v.clone())
                } else if is_latin1_keyword(k) {
                    encoder.add_itxt_chunk(k.clone(), v.clone())
                } else {
                    continue;
                };
                res.map_err(|e| CodecError::encode(F, e))?;
            }
            if let Some(xmp) = &img.meta.xmp {
                encoder
                    .add_itxt_chunk(XMP_KEYWORD.into(), crate::resolution::export_xmp(xmp, img.meta.dpi).into_owned())
                    .map_err(|e| CodecError::encode(F, e))?;
            }
        }
        let mut writer = encoder.write_header().map_err(|e| CodecError::encode(F, e))?;
        if opts.png_interlaced {
            let bpp = img.layout().channels() * img.sample_type().bytes();
            let idat = adam7_idat(&data, w as usize, h as usize, bpp, opts.png_compression)?;
            writer.write_chunk(png::chunk::IDAT, &idat).map_err(|e| CodecError::encode(F, e))?;
        } else if let Some(idat) = parallel_idat(&data, w as usize, h as usize, img.layout().channels() * img.sample_type().bytes(), opts.png_compression) {
            writer.write_chunk(png::chunk::IDAT, &idat).map_err(|e| CodecError::encode(F, e))?;
        } else {
            writer.write_image_data(&data).map_err(|e| CodecError::encode(F, e))?;
        }
        writer.finish().map_err(|e| CodecError::encode(F, e))?;
    }
    Ok(out)
}

/// Build the zlib stream for an Adam7-interlaced image (filter type 0).
fn adam7_idat(data: &[u8], w: usize, h: usize, bpp: usize, level: PngCompression) -> Result<Vec<u8>, CodecError> {
    // (x0, y0, dx, dy) per pass.
    const PASSES: [(usize, usize, usize, usize); 7] = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];
    let mut raw = Vec::with_capacity(data.len() + h * 7);
    for (x0, y0, dx, dy) in PASSES {
        if x0 >= w || y0 >= h {
            continue;
        }
        let mut y = y0;
        while y < h {
            raw.push(0u8);
            let mut x = x0;
            while x < w {
                let o = (y * w + x) * bpp;
                raw.extend_from_slice(&data[o..o + bpp]);
                x += dx;
            }
            y += dy;
        }
    }
    let lvl = match level {
        PngCompression::None => flate2::Compression::none(),
        PngCompression::Fast => flate2::Compression::fast(),
        PngCompression::Default => flate2::Compression::default(),
        PngCompression::Best => flate2::Compression::best(),
    };
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), lvl);
    z.write_all(&raw).map_err(|e| CodecError::encode(F, e))?;
    z.finish().map_err(|e| CodecError::encode(F, e))
}

/// Images at least this large (bytes) are filtered and deflated on all cores.
const PARALLEL_MIN_BYTES: usize = 4 << 20;

/// The zlib stream of a non-interlaced image, built in parallel (`None` below
/// [`PARALLEL_MIN_BYTES`], on wasm, or without compression): rows get the adaptive filter
/// (minimum sum of absolute differences over the five PNG filters, as libpng and the `png`
/// crate do), then bands of rows are deflated independently at the requested level, each
/// ending on a sync flush so the raw deflate streams concatenate into one (as `pigz` does).
/// A 36 MP RGB export drops from ~9 s to well under a second on 12 cores; the output is a few
/// percent larger (no dictionary across bands).
fn parallel_idat(data: &[u8], w: usize, h: usize, bpp: usize, level: PngCompression) -> Option<Vec<u8>> {
    if cfg!(target_arch = "wasm32") || data.len() < PARALLEL_MIN_BYTES || level == PngCompression::None || w == 0 || h == 0 {
        return None;
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 32);
    banded_idat(data, w, h, bpp, level, threads)
}

/// Filtered bytes per deflate band. The bands depend only on the image, never on the number of
/// threads, so the same image and options give the same bytes whatever the machine's core count.
const BAND_BYTES: usize = 512 << 10;

/// [`parallel_idat`] on `threads` worker threads.
fn banded_idat(data: &[u8], w: usize, h: usize, bpp: usize, level: PngCompression, threads: usize) -> Option<Vec<u8>> {
    let stride = w * bpp;
    let band_rows = (BAND_BYTES / stride.saturating_add(1)).max(1);
    let bands: Vec<(usize, usize)> = (0..h).step_by(band_rows).map(|y| (y, (y + band_rows).min(h))).collect();
    let lvl = match level {
        PngCompression::Fast => flate2::Compression::fast(),
        PngCompression::Best => flate2::Compression::best(),
        _ => flate2::Compression::default(),
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Band = (usize, Vec<u8>, u32, usize);
    let mut parts: Vec<Band> = std::thread::scope(|sc| {
        let workers: Vec<_> = (0..threads.min(bands.len()).max(1))
            .map(|_| {
                sc.spawn(|| {
                    let mut out: Vec<Band> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(y0, y1)) = bands.get(i) else { break };
                        let mut filtered = Vec::with_capacity((y1 - y0) * (stride + 1));
                        let mut scratch = vec![0u8; stride];
                        for y in y0..y1 {
                            let row = &data[y * stride..(y + 1) * stride];
                            let prev = (y > 0).then(|| &data[(y - 1) * stride..y * stride]);
                            filter_row(row, prev, bpp, &mut scratch, &mut filtered);
                        }
                        let adler = adler32(1, &filtered);
                        let Some(z) = deflate_band(&filtered, lvl, y1 == h) else { return Vec::new() };
                        out.push((i, z, adler, filtered.len()));
                    }
                    out
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
    });
    if parts.len() != bands.len() {
        return None;
    }
    parts.sort_by_key(|p| p.0);
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.1.len()).sum::<usize>() + 6);
    // zlib header: deflate, 32 K window; FLEVEL only advertises the effort.
    out.extend_from_slice(match level {
        PngCompression::Fast => &[0x78, 0x01],
        PngCompression::Best => &[0x78, 0xDA],
        _ => &[0x78, 0x9C],
    });
    let mut adler = 1u32;
    for (_, z, a, len) in &parts {
        out.extend_from_slice(z);
        adler = adler32_combine(adler, *a, *len);
    }
    out.extend_from_slice(&adler.to_be_bytes());
    Some(out)
}

/// Raw deflate of one band: ends on a sync flush, or finishes the stream for the `last` band.
/// The output buffer is sized past deflate's worst case so that one call takes the whole band
/// and completes the flush: when a sync flush fills the buffer, miniz_oxide (flate2's Rust
/// backend) can return before compressing the last lookahead bytes, and the next call only
/// drains the pending output, so a resumed flush silently drops data. `None` (the caller then
/// uses the serial encoder) if the single call did not finish.
fn deflate_band(filtered: &[u8], lvl: flate2::Compression, last: bool) -> Option<Vec<u8>> {
    let mut c = flate2::Compress::new(lvl, false);
    // Stored blocks cost 5 bytes per 64 KiB, the flush an empty stored block: 1/8 is ample.
    let mut z = Vec::with_capacity(filtered.len() + filtered.len() / 8 + 1024);
    let flush = if last { flate2::FlushCompress::Finish } else { flate2::FlushCompress::Sync };
    let status = c.compress_vec(filtered, &mut z, flush).ok()?;
    let done = if last { status == flate2::Status::StreamEnd } else { c.total_in() as usize == filtered.len() && z.len() < z.capacity() };
    done.then_some(z)
}

/// One scanline with the adaptive filter choice, appended to `out` (type byte + data).
fn filter_row(row: &[u8], prev: Option<&[u8]>, bpp: usize, scratch: &mut [u8], out: &mut Vec<u8>) {
    let n = row.len();
    let up = |i: usize| prev.map_or(0, |p| p[i]);
    let left = |i: usize| if i >= bpp { row[i - bpp] } else { 0 };
    let ul = |i: usize| if i >= bpp { prev.map_or(0, |p| p[i - bpp]) } else { 0 };
    let paeth = |a: u8, b: u8, c: u8| {
        let p = i16::from(a) + i16::from(b) - i16::from(c);
        let (pa, pb, pc) = ((p - i16::from(a)).abs(), (p - i16::from(b)).abs(), (p - i16::from(c)).abs());
        if pa <= pb && pa <= pc {
            a
        } else if pb <= pc {
            b
        } else {
            c
        }
    };
    let cost = |v: &[u8]| v.iter().map(|&b| u64::from((b as i8).unsigned_abs())).sum::<u64>();
    let mut best = (cost(row), 0u8);
    for kind in 1u8..=4 {
        for i in 0..n {
            let x = row[i];
            scratch[i] = match kind {
                1 => x.wrapping_sub(left(i)),
                2 => x.wrapping_sub(up(i)),
                3 => x.wrapping_sub(((u16::from(left(i)) + u16::from(up(i))) / 2) as u8),
                _ => x.wrapping_sub(paeth(left(i), up(i), ul(i))),
            };
        }
        let c = cost(&scratch[..n]);
        if c < best.0 {
            best = (c, kind);
        }
    }
    out.push(best.1);
    if best.1 == 0 {
        out.extend_from_slice(row);
        return;
    }
    for i in 0..n {
        let x = row[i];
        scratch[i] = match best.1 {
            1 => x.wrapping_sub(left(i)),
            2 => x.wrapping_sub(up(i)),
            3 => x.wrapping_sub(((u16::from(left(i)) + u16::from(up(i))) / 2) as u8),
            _ => x.wrapping_sub(paeth(left(i), up(i), ul(i))),
        };
    }
    out.extend_from_slice(&scratch[..n]);
}

const ADLER_MOD: u32 = 65_521;

fn adler32(start: u32, data: &[u8]) -> u32 {
    let (mut a, mut b) = (start & 0xffff, start >> 16);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= ADLER_MOD;
        b %= ADLER_MOD;
    }
    (b << 16) | a
}

/// Adler-32 of A ‖ B from adler(A), adler(B) and |B| (zlib's `adler32_combine`).
fn adler32_combine(a1: u32, a2: u32, len2: usize) -> u32 {
    let rem = (len2 % ADLER_MOD as usize) as u64;
    let m = u64::from(ADLER_MOD);
    let (s1a, s2a) = (u64::from(a1 & 0xffff), u64::from(a1 >> 16));
    let (s1b, s2b) = (u64::from(a2 & 0xffff), u64::from(a2 >> 16));
    let s1 = (s1a + s1b + m - 1) % m;
    let s2 = (s2a + s2b + (rem * s1a) % m + m - rem) % m;
    ((s2 as u32) << 16) | s1 as u32
}

/// Writes an 8-bit palette PNG (PNG-8): `indices` row-major, `palette` RGB entries (≤ 256),
/// optional fully transparent entry (`tRNS`) and optional resolution in dpi (`pHYs`, written when
/// both axes are positive and finite). Used for Indexed Color documents.
pub fn encode_indexed(
    width: u32,
    height: u32,
    indices: &[u8],
    palette: &[[u8; 3]],
    transparent: Option<u8>,
    dpi: Option<(f32, f32)>,
) -> Result<Vec<u8>, CodecError> {
    if width == 0 || height == 0 || indices.len() != width as usize * height as usize {
        return Err(CodecError::InvalidImage("index data does not match the size".into()));
    }
    if palette.is_empty() || palette.len() > 256 || indices.iter().any(|&i| usize::from(i) >= palette.len()) {
        return Err(CodecError::InvalidImage("palette must have 1..=256 entries covering every index".into()));
    }
    let mut info = png::Info::with_size(width, height);
    info.color_type = png::ColorType::Indexed;
    info.bit_depth = png::BitDepth::Eight;
    info.palette = Some(palette.iter().flatten().copied().collect::<Vec<u8>>().into());
    if let Some(t) = transparent.filter(|&t| usize::from(t) < palette.len()) {
        let mut trns = vec![255u8; usize::from(t) + 1];
        trns[usize::from(t)] = 0;
        info.trns = Some(trns.into());
    }
    if let Some((x, y)) = dpi
        && x.is_finite()
        && y.is_finite()
        && x > 0.0
        && y > 0.0
    {
        info.pixel_dims =
            Some(png::PixelDimensions { xppu: (x / METERS_PER_INCH).round() as u32, yppu: (y / METERS_PER_INCH).round() as u32, unit: png::Unit::Meter });
    }
    let mut out = Vec::new();
    {
        let encoder = png::Encoder::with_info(&mut out, info).map_err(|e| CodecError::encode(F, e))?;
        let mut writer = encoder.write_header().map_err(|e| CodecError::encode(F, e))?;
        writer.write_image_data(indices).map_err(|e| CodecError::encode(F, e))?;
        writer.finish().map_err(|e| CodecError::encode(F, e))?;
    }
    Ok(out)
}

#[cfg(test)]
mod indexed_tests {
    use super::*;

    #[test]
    fn png8_roundtrips_through_the_decoder() {
        let pal = [[255, 0, 0], [0, 0, 255], [0, 0, 0]];
        let idx = [0u8, 1, 2, 1, 0, 2];
        let bytes = encode_indexed(3, 2, &idx, &pal, Some(2), None).unwrap();
        let img = decode(&bytes, &Limits::default()).unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        let px = img.convert(ChannelLayout::Rgba, SampleType::U8);
        assert_eq!(&px.data()[..8], &[255, 0, 0, 255, 0, 0, 255, 255]);
        assert_eq!(px.data()[11], 0, "transparent entry");
        assert!(encode_indexed(3, 2, &[9; 6], &pal, None, None).is_err());
        assert!(encode_indexed(3, 3, &idx, &pal, None, None).is_err());
    }

    fn png(color: png::ColorType, trns: Option<&[u8]>) -> Vec<u8> {
        let mut out = Vec::new();
        let mut e = png::Encoder::new(&mut out, 2, 1);
        e.set_color(color);
        e.set_depth(png::BitDepth::Eight);
        e.set_palette(vec![1, 2, 3, 4, 5, 6]);
        if let Some(t) = trns {
            e.set_trns(t.to_vec());
        }
        let data = if color == png::ColorType::Rgb { vec![0; 6] } else { vec![0, 1] };
        e.write_header().unwrap().write_image_data(&data).unwrap();
        out
    }

    #[test]
    fn palette_reads_the_table_and_its_one_transparent_entry() {
        let bytes = encode_indexed(2, 1, &[0, 1], &[[1, 2, 3], [4, 5, 6]], Some(1), None).unwrap();
        let colors = vec![[1, 2, 3], [4, 5, 6]];
        assert_eq!(palette(&bytes).unwrap(), Some(PngPalette { colors: colors.clone(), transparent: Some(1) }));
        assert_eq!(palette(&png(png::ColorType::Indexed, None)).unwrap(), Some(PngPalette { colors, transparent: None }));
        // A truecolour PNG's suggested palette is not a colour table.
        assert_eq!(palette(&png(png::ColorType::Rgb, None)).unwrap(), None);
        // Partial alpha or two transparent entries don't fit one transparent index.
        assert_eq!(palette(&png(png::ColorType::Indexed, Some(&[255, 128]))).unwrap(), None);
        assert_eq!(palette(&png(png::ColorType::Indexed, Some(&[0, 0]))).unwrap(), None);
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    #[test]
    fn adler_combine_matches_one_pass() {
        let a: Vec<u8> = (0..100_000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        let b: Vec<u8> = (0..70_001u32).map(|i| (i * 31) as u8).collect();
        let whole: Vec<u8> = a.iter().chain(&b).copied().collect();
        assert_eq!(adler32_combine(adler32(1, &a), adler32(1, &b), b.len()), adler32(1, &whole));
    }

    #[test]
    fn bands_that_end_just_past_a_block_keep_all_their_data() {
        // Deflate output ~0.6 of the input, cut just past one of miniz_oxide's block
        // boundaries: a sync flush that fills the output buffer there used to lose the
        // last lookahead bytes, corrupting every band after it.
        for (k, len) in [(20u32, 136_916usize), (20, 345_044), (24, 195_796), (28, 189_142)] {
            let mut x = 0x9E37_79B9u32 ^ k;
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    ((x >> 8) % k) as u8
                })
                .collect();
            for last in [false, true] {
                let z = deflate_band(&data, flate2::Compression::default(), last).unwrap();
                let mut d = flate2::Decompress::new(false);
                let mut back = Vec::with_capacity(len + 1);
                d.decompress_vec(&z, &mut back, flate2::FlushDecompress::Sync).unwrap();
                assert!(back == data, "k {k} len {len} last {last}: {} of {len} bytes", back.len());
            }
        }
    }

    /// The bytes depend on the image, not on the machine: one, two, seven and 32 worker threads
    /// (and the machine's own count, through `encode`) give the same stream.
    #[test]
    fn the_parallel_encoder_gives_the_same_bytes_on_any_thread_count() {
        let (w, h, bpp) = (1201usize, 1013usize, 4usize);
        let px: Vec<u8> =
            (0..w * h * bpp).map(|i| ((i / 5) as u32).wrapping_mul(2_654_435_761).rotate_left(i as u32 % 11) as u8 / 5 + (i % 89) as u8).collect();
        assert!(px.len() >= PARALLEL_MIN_BYTES && px.len() > 2 * BAND_BYTES);
        for level in [PngCompression::Fast, PngCompression::Default, PngCompression::Best] {
            let one = banded_idat(&px, w, h, bpp, level, 1).unwrap();
            for threads in [2, 7, 32] {
                assert!(banded_idat(&px, w, h, bpp, level, threads).unwrap() == one, "{level:?}: {threads} threads differ from one");
            }
            assert!(parallel_idat(&px, w, h, bpp, level).unwrap() == one, "{level:?}: this machine's thread count differs from one");
            let img = Image::from_raw(w as u32, h as u32, ChannelLayout::Rgba, SampleType::U8, px.clone()).unwrap();
            let opts = EncodeOptions { png_compression: level, ..Default::default() };
            let png = crate::encode(&img, Format::Png, &opts).unwrap();
            assert!(png.windows(one.len()).any(|c| c == one.as_slice()), "{level:?}: the file carries the stream");
            assert_eq!(decode(&png, &Limits::default()).unwrap().data(), img.data());
        }
    }

    #[test]
    fn large_images_roundtrip_through_the_parallel_encoder() {
        for (layout, sample, ch, bytes) in [(ChannelLayout::Rgb, SampleType::U8, 3usize, 1usize), (ChannelLayout::Rgba, SampleType::U16, 4, 2)] {
            let (w, h) = (1531u32, 977u32);
            let n = w as usize * h as usize * ch * bytes;
            let px: Vec<u8> = (0..n).map(|i| ((i / 3) as u32).wrapping_mul(2_654_435_761).rotate_left(i as u32 % 7) as u8 / 3 + (i % 97) as u8).collect();
            let img = Image::from_raw(w, h, layout, sample, px.clone()).unwrap();
            assert!(n >= PARALLEL_MIN_BYTES || sample == SampleType::U8);
            for level in [PngCompression::Fast, PngCompression::Default] {
                let opts = EncodeOptions { png_compression: level, ..Default::default() };
                let bytes = crate::encode(&img, Format::Png, &opts).unwrap();
                let back = decode(&bytes, &Limits::default()).unwrap();
                assert_eq!(back.dimensions(), (w, h));
                assert_eq!(back.convert(layout, sample).data(), img.data(), "{layout:?} {sample:?} {level:?}");
                // The oracle decoder agrees.
                let o = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).unwrap();
                assert_eq!((o.width(), o.height()), (w, h));
            }
        }
    }
}
