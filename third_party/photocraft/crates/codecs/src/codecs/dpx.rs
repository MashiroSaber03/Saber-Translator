//! DPX (SMPTE 268M Digital Picture Exchange), implemented here from the published layout: the
//! 1664-byte generic header (file and image information, image source), the 384-byte industry
//! header (film and television), then the image data of each element.
//!
//! Reading:
//!
//! * Both byte orders: `SDPX` (big-endian) and `XPDS` (little-endian).
//! * The first image element, descriptor 6 (luma), 50 (RGB) or 51 (RGBA), unsigned, without
//!   run-length encoding, at 8, 10, 12 or 16 bits. 10-bit samples are packed into 32-bit words
//!   (packing 0), or three to a word with the padding bits at the bottom (1, method A) or the
//!   top (2, method B); 12-bit samples are packed (0) or one to a 16-bit word in its top bits
//!   (1, method A). Lines start on a 32-bit word boundary.
//! * 8- and 16-bit code values become U8 and U16 samples unchanged; 10- and 12-bit ones become U16
//!   as `round(c × 65535 / (2^b − 1))`, so they come back exactly when written at their depth.
//! * No colour transform: the transfer characteristic and colorimetric specification are reported
//!   ([`DecodeWarning::DpxEncoding`]), so a log frame is labelled log, not shown as display-referred.
//! * The image orientation is applied on import (the caller turns the image upright).
//! * Header fields a pipeline relies on travel as text metadata under the [`KEYS`] names, and a
//!   DPX export writes them back.
//!
//! Writing: one element, big-endian, unencoded, descriptor 6, 50 or 51; 8 and 16 bits with
//! packing 0, 10 and 12 bits with packing 1 (method A). The bit depth follows the image
//! (8 → 8, 16 → 16) unless [`EncodeOptions::dpx_bits`] picks one.

use crate::Format;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, DecodeWarning, Image, SampleType};
use crate::options::{DpxBits, EncodeOptions, Limits};

const F: Format = Format::Dpx;
/// Generic plus industry header: where our image data starts.
const HEADER_LEN: usize = 2048;
const GENERIC_LEN: u32 = 1664;
const INDUSTRY_LEN: u32 = 384;
/// "Undefined" for 32-bit fields (also the bit pattern of an undefined float).
const UNDEF32: u32 = u32::MAX;
const UNDEF8: u8 = u8::MAX;
/// Offset of the first image element and the size of one.
const ELEMENT0: usize = 780;
const ELEMENT_LEN: usize = 72;
const MAX_ELEMENTS: u16 = 8;

// Header field offsets (generic file header, image information, film and TV headers).
const CREATION_TIME: (usize, usize) = (136, 24);
const CREATOR: (usize, usize) = (160, 100);
const PROJECT: (usize, usize) = (260, 200);
const COPYRIGHT: (usize, usize) = (460, 200);
const ORIENTATION: usize = 768;
const FRAME_POSITION: usize = 1712;
const FRAME_RATE: usize = 1724;
const TIME_CODE: usize = 1920;

/// Text metadata keys of the header fields kept through a DPX → document → DPX round trip.
pub(crate) const KEY_CREATOR: &str = "DPX:Creator";
pub(crate) const KEY_PROJECT: &str = "DPX:Project";
pub(crate) const KEY_COPYRIGHT: &str = "DPX:Copyright";
pub(crate) const KEY_CREATION_TIME: &str = "DPX:CreationTime";
pub(crate) const KEY_TRANSFER: &str = "DPX:Transfer";
pub(crate) const KEY_COLORIMETRIC: &str = "DPX:Colorimetric";
pub(crate) const KEY_FRAME_POSITION: &str = "DPX:FramePosition";
pub(crate) const KEY_FRAME_RATE: &str = "DPX:FrameRate";
pub(crate) const KEY_TIME_CODE: &str = "DPX:TimeCode";
pub(crate) const KEYS: [&str; 9] =
    [KEY_CREATOR, KEY_PROJECT, KEY_COPYRIGHT, KEY_CREATION_TIME, KEY_TRANSFER, KEY_COLORIMETRIC, KEY_FRAME_POSITION, KEY_FRAME_RATE, KEY_TIME_CODE];

fn err(msg: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, msg)
}

fn unsupported(msg: impl Into<String>) -> CodecError {
    CodecError::unsupported(F, msg)
}

/// The file's byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Big,
    Little,
}

impl Order {
    fn bytes<const N: usize>(b: &[u8], at: usize) -> Option<[u8; N]> {
        b.get(at..at.checked_add(N)?)?.try_into().ok()
    }

    fn u16(self, b: &[u8], at: usize) -> Option<u16> {
        let s = Self::bytes::<2>(b, at)?;
        Some(if self == Order::Big { u16::from_be_bytes(s) } else { u16::from_le_bytes(s) })
    }

    fn u32(self, b: &[u8], at: usize) -> Option<u32> {
        let s = Self::bytes::<4>(b, at)?;
        Some(if self == Order::Big { u32::from_be_bytes(s) } else { u32::from_le_bytes(s) })
    }

    /// A 32-bit field that may be undefined (all ones).
    fn defined_u32(self, b: &[u8], at: usize) -> Option<u32> {
        self.u32(b, at).filter(|&v| v != UNDEF32)
    }

    fn f32(self, b: &[u8], at: usize) -> Option<f32> {
        self.defined_u32(b, at).map(f32::from_bits).filter(|v| v.is_finite())
    }
}

/// The byte order from the magic number, `None` when it is not DPX.
fn order(b: &[u8]) -> Option<Order> {
    match b.get(..4)? {
        b"SDPX" => Some(Order::Big),
        b"XPDS" => Some(Order::Little),
        _ => None,
    }
}

/// An ASCII header string: up to the first NUL, printable characters only, trimmed. `None` when
/// empty or undefined (all ones).
fn string(b: &[u8], (at, len): (usize, usize)) -> Option<String> {
    let raw = b.get(at..at.checked_add(len)?)?;
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    let s: String = raw.get(..end)?.iter().filter(|c| (32..=126).contains(*c)).map(|&c| c as char).collect();
    Some(s.trim().to_string()).filter(|s| !s.is_empty())
}

/// The descriptive fields of image element 0 that decoding needs.
#[derive(Debug, Clone, Copy)]
struct Element {
    signed: bool,
    descriptor: u8,
    transfer: u8,
    colorimetric: u8,
    bits: u8,
    packing: u16,
    encoding: u16,
    data_offset: usize,
    eol_padding: usize,
}

struct Header {
    order: Order,
    width: u32,
    height: u32,
    elements: u16,
    element: Element,
}

fn parse(b: &[u8]) -> Result<Header, CodecError> {
    let o = order(b).ok_or_else(|| err("missing SDPX/XPDS magic number"))?;
    let missing = |what: &str| err(format!("the header ends before {what}"));
    let image_offset = o.u32(b, 4).ok_or_else(|| missing("the image data offset"))?;
    o.u16(b, ORIENTATION).ok_or_else(|| missing("the orientation"))?;
    let elements = o.u16(b, ORIENTATION + 2).ok_or_else(|| missing("the element count"))?;
    if !(1..=MAX_ELEMENTS).contains(&elements) {
        return Err(err(format!("invalid number of image elements {elements}")));
    }
    let width = o.u32(b, 772).ok_or_else(|| missing("the line length"))?;
    let height = o.u32(b, 776).ok_or_else(|| missing("the line count"))?;
    let e = ELEMENT0;
    let field = |at: usize| o.u32(b, e + at).ok_or_else(|| missing("image element 1"));
    let byte = |at: usize| b.get(e + at).copied().ok_or_else(|| missing("image element 1"));
    let half = |at: usize| o.u16(b, e + at).ok_or_else(|| missing("image element 1"));
    let data_offset = match field(28)? {
        0 | UNDEF32 => image_offset,
        v => v,
    };
    let eol_padding = match field(32)? {
        UNDEF32 => 0,
        v => v,
    };
    let element = Element {
        signed: field(0)? == 1,
        descriptor: byte(20)?,
        transfer: byte(21)?,
        colorimetric: byte(22)?,
        bits: byte(23)?,
        packing: half(24)?,
        encoding: half(26)?,
        data_offset: usize::try_from(data_offset).map_err(|_| err("image data offset out of range"))?,
        eol_padding: usize::try_from(eol_padding).map_err(|_| err("line padding out of range"))?,
    };
    Ok(Header { order: o, width, height, elements, element })
}

/// The EXIF orientation (1–8) that turns the stored lines upright: the DPX orientation codes 0–7
/// in SMPTE 268M order (left-to-right top-to-bottom, mirrored, flipped, rotated, transposed…).
/// Anything else, undefined included, is taken as the default (1).
pub(crate) fn orientation(b: &[u8]) -> u16 {
    let Some(o) = order(b).and_then(|o| o.u16(b, ORIENTATION)) else { return 1 };
    match o {
        1 => 2,
        2 => 4,
        3 => 3,
        4 => 5,
        5 => 6,
        6 => 8,
        7 => 7,
        _ => 1,
    }
}

/// Bytes in one stored line of `samples` samples, padded to a 32-bit word.
fn line_bytes(samples: usize, bits: u8, packing: u16) -> Option<usize> {
    let words = match (bits, packing) {
        // Three samples to a word, or one to a 16-bit half word.
        (10, 1 | 2) => samples.div_ceil(3),
        (12, 1 | 2) => samples.div_ceil(2),
        _ => samples.checked_mul(usize::from(bits))?.div_ceil(32),
    };
    words.checked_mul(4)
}

/// Code values of one stored line.
fn unpack_line(line: &[u8], o: Order, bits: u8, packing: u16, out: &mut [u16]) -> Result<(), CodecError> {
    let short = || err("truncated image data");
    match (bits, packing) {
        (8, _) => {
            let src = line.get(..out.len()).ok_or_else(short)?;
            for (v, &c) in out.iter_mut().zip(src) {
                *v = u16::from(c);
            }
        }
        (16, _) => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = o.u16(line, i * 2).ok_or_else(short)?;
            }
        }
        (10, 1 | 2) => {
            let base = if packing == 1 { 22 } else { 20 };
            for (i, v) in out.iter_mut().enumerate() {
                let word = o.u32(line, i / 3 * 4).ok_or_else(short)?;
                *v = ((word >> (base - 10 * (i % 3) as u32)) & 0x3FF) as u16;
            }
        }
        (12, 1 | 2) => {
            let shift = if packing == 1 { 4 } else { 0 };
            for (i, v) in out.iter_mut().enumerate() {
                *v = (o.u16(line, i * 2).ok_or_else(short)? >> shift) & 0xFFF;
            }
        }
        // Packing 0: one bit stream through the line's 32-bit words, high bits first.
        _ => {
            let mask = (1u64 << bits) - 1;
            let (mut acc, mut have, mut word) = (0u64, 0u32, 0usize);
            for v in out.iter_mut() {
                while have < u32::from(bits) {
                    acc = (acc << 32) | u64::from(o.u32(line, word * 4).ok_or_else(short)?);
                    word += 1;
                    have += 32;
                }
                have -= u32::from(bits);
                *v = ((acc >> have) & mask) as u16;
            }
        }
    }
    Ok(())
}

/// A 10- or 12-bit code value as a 16-bit sample: `round(c × 65535 / max)`.
fn widen(c: u16, max: u64) -> u16 {
    ((u64::from(c) * 2 * 65535 + max) / (2 * max)).min(65535) as u16
}

/// The inverse of [`widen`]: `round(v × max / 65535)`.
fn narrow(v: u16, max: u64) -> u16 {
    ((u64::from(v) * 2 * max + 65535) / (2 * 65535)) as u16
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    let h = parse(bytes)?;
    let e = h.element;
    let layout = match e.descriptor {
        6 => ChannelLayout::Gray,
        50 => ChannelLayout::Rgb,
        51 => ChannelLayout::Rgba,
        d => return Err(unsupported(format!("descriptor {d} is not supported (only 6 luma, 50 RGB and 51 RGBA)"))),
    };
    if e.signed {
        return Err(unsupported("data sign 1 (signed samples) is not supported"));
    }
    if e.encoding != 0 {
        return Err(unsupported(format!("encoding {} (run-length encoded data) is not supported", e.encoding)));
    }
    let packing_ok = match e.bits {
        8 | 16 => e.packing == 0,
        10 => e.packing <= 2,
        12 => e.packing <= 1,
        b => return Err(unsupported(format!("bit size {b} is not supported (only 8, 10, 12 and 16)"))),
    };
    if !packing_ok {
        return Err(unsupported(format!("packing {} is not supported at {} bits", e.packing, e.bits)));
    }
    let sample = if e.bits == 8 { SampleType::U8 } else { SampleType::U16 };
    limits.check(h.width, h.height, layout, sample)?;

    let nc = layout.channels();
    let per_line = (h.width as usize).checked_mul(nc).ok_or_else(|| CodecError::LimitExceeded("line too long".into()))?;
    let height = h.height as usize;
    let data = bytes.get(e.data_offset..).ok_or_else(|| err("the image data offset is past the end of the file"))?;
    let too_large = || CodecError::LimitExceeded("image too large".into());
    let mut line_len = line_bytes(per_line, e.bits, e.packing).ok_or_else(too_large)?;
    let mut stride = line_len.checked_add(e.eol_padding).ok_or_else(too_large)?;
    // Lines run `stride` apart; the last one needs no padding after it.
    let span = |stride: usize, line_len: usize| stride.checked_mul(height - 1)?.checked_add(line_len);
    // Some writers pack 8- and 16-bit lines without padding them to a word; take that layout
    // when the file is too short for padded lines but holds the unpadded ones.
    let tight = per_line * usize::from(e.bits / 8);
    if matches!(e.bits, 8 | 16) && e.eol_padding == 0 && line_len != tight && span(stride, line_len).is_none_or(|need| data.len() < need) {
        (line_len, stride) = (tight, tight);
    }
    if span(stride, line_len).ok_or_else(too_large)? > data.len() {
        return Err(err("truncated image data"));
    }

    let mut codes = vec![0u16; per_line.checked_mul(height).ok_or_else(too_large)?];
    for (y, row) in codes.chunks_exact_mut(per_line.max(1)).enumerate() {
        let start = y * stride;
        let line = data.get(start..).ok_or_else(|| err("truncated image data"))?;
        unpack_line(line, h.order, e.bits, e.packing, row)?;
    }

    let mut img = match e.bits {
        8 => Image::from_u8(h.width, h.height, layout, codes.iter().map(|&c| c as u8).collect())?,
        16 => Image::from_u16(h.width, h.height, layout, &codes)?,
        b => {
            let max = (1u64 << b) - 1;
            let wide: Vec<u16> = codes.iter().map(|&c| widen(c, max)).collect();
            Image::from_u16(h.width, h.height, layout, &wide)?
        }
    };
    if h.elements > 1 {
        img.warnings.push(DecodeWarning::MoreElements { total: u32::from(h.elements) });
    }
    if e.transfer != UNDEF8 || e.colorimetric != UNDEF8 {
        img.warnings.push(DecodeWarning::DpxEncoding { transfer: e.transfer, colorimetric: e.colorimetric });
    }
    img.meta.text = header_text(bytes, h.order, e);
    Ok(img)
}

/// The kept header fields as text metadata, in [`KEYS`] order; undefined ones are left out.
fn header_text(b: &[u8], o: Order, e: Element) -> Vec<(String, String)> {
    let mut text = Vec::new();
    let mut put = |k: &str, v: Option<String>| {
        if let Some(v) = v {
            text.push((k.to_string(), v));
        }
    };
    put(KEY_CREATOR, string(b, CREATOR));
    put(KEY_PROJECT, string(b, PROJECT));
    put(KEY_COPYRIGHT, string(b, COPYRIGHT));
    put(KEY_CREATION_TIME, string(b, CREATION_TIME));
    put(KEY_TRANSFER, (e.transfer != UNDEF8).then(|| e.transfer.to_string()));
    put(KEY_COLORIMETRIC, (e.colorimetric != UNDEF8).then(|| e.colorimetric.to_string()));
    // The industry header is there when the generic header says so and the file holds it.
    let industry = o.u32(b, 28).is_some_and(|l| l != 0 && l != UNDEF32) && b.len() >= HEADER_LEN;
    if industry {
        put(KEY_FRAME_POSITION, o.defined_u32(b, FRAME_POSITION).map(|v| v.to_string()));
        put(KEY_FRAME_RATE, o.f32(b, FRAME_RATE).map(|v| v.to_string()));
        put(KEY_TIME_CODE, o.defined_u32(b, TIME_CODE).map(format_time_code));
    }
    text
}

/// A SMPTE time code (BCD `HHMMSSFF`) as `HH:MM:SS:FF`, or `0x…` when it is not valid BCD.
fn format_time_code(v: u32) -> String {
    let d: [u32; 8] = std::array::from_fn(|i| (v >> ((7 - i) * 4)) & 0xF);
    if d.iter().all(|&d| d <= 9) { format!("{}{}:{}{}:{}{}:{}{}", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]) } else { format!("{v:#010x}") }
}

/// The inverse of [`format_time_code`]; `None` for anything else.
fn parse_time_code(s: &str) -> Option<u32> {
    if let Some(hex) = s.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).ok();
    }
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 4 || parts.iter().any(|p| p.len() != 2 || !p.bytes().all(|c| c.is_ascii_digit())) {
        return None;
    }
    parts.iter().flat_map(|p| p.bytes()).try_fold(0u32, |acc, c| Some((acc << 4) | u32::from(c - b'0')))
}

/// Bits written for `sample` when the options pick none: 8 for U8, 16 otherwise.
pub(crate) fn bits_for(opt: Option<DpxBits>, sample: SampleType) -> u8 {
    match opt {
        Some(b) => b.bits(),
        None if sample == SampleType::U8 => 8,
        None => 16,
    }
}

/// The plan's sample type for the chosen bits: U8 for 8 bits, U16 for 10, 12 and 16.
pub(crate) fn plan_sample(opt: Option<DpxBits>, src: SampleType) -> SampleType {
    match opt {
        Some(DpxBits::Bits8) => SampleType::U8,
        Some(_) => SampleType::U16,
        None if src == SampleType::U8 => SampleType::U8,
        None => SampleType::U16,
    }
}

struct Writer {
    h: Vec<u8>,
}

impl Writer {
    fn u8(&mut self, at: usize, v: u8) {
        if let Some(b) = self.h.get_mut(at) {
            *b = v;
        }
    }

    fn u16(&mut self, at: usize, v: u16) {
        if let Some(s) = self.h.get_mut(at..at + 2) {
            s.copy_from_slice(&v.to_be_bytes());
        }
    }

    fn u32(&mut self, at: usize, v: u32) {
        if let Some(s) = self.h.get_mut(at..at + 4) {
            s.copy_from_slice(&v.to_be_bytes());
        }
    }

    /// Marks a run of numeric fields undefined (all ones).
    fn undefined(&mut self, from: usize, to: usize) {
        if let Some(s) = self.h.get_mut(from..to) {
            s.fill(0xFF);
        }
    }

    /// An ASCII string field, cut to leave room for a terminating NUL.
    fn string(&mut self, (at, len): (usize, usize), v: &str) {
        let ascii: Vec<u8> = v.chars().map(|c| if (' '..='~').contains(&c) { c as u8 } else { b'?' }).take(len.saturating_sub(1)).collect();
        if let Some(s) = self.h.get_mut(at..at + ascii.len()) {
            s.copy_from_slice(&ascii);
        }
    }
}

fn text<'a>(img: &'a Image, key: &str) -> Option<&'a str> {
    img.meta.text.iter().find(|(k, _)| k == key).map(|(_, v)| v.trim()).filter(|v| !v.is_empty())
}

/// Appends one line's code values, padded to a 32-bit word.
fn pack_line(out: &mut Vec<u8>, codes: &[u16], bits: u8) {
    let start = out.len();
    match bits {
        8 => out.extend(codes.iter().map(|&c| c as u8)),
        10 => {
            for chunk in codes.chunks(3) {
                let word = chunk.iter().enumerate().fold(0u32, |w, (i, &c)| w | (u32::from(c & 0x3FF) << (22 - 10 * i as u32)));
                out.extend_from_slice(&word.to_be_bytes());
            }
        }
        12 => out.extend(codes.iter().flat_map(|&c| ((c & 0xFFF) << 4).to_be_bytes())),
        _ => out.extend(codes.iter().flat_map(|&c| c.to_be_bytes())),
    }
    let written = out.len() - start;
    out.resize(start + written.div_ceil(4) * 4, 0);
}

pub(crate) fn encode(src: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let img = src.converted(plan.layout, plan.sample);
    let (w, h) = img.dimensions();
    let (descriptor, layout) = match img.layout() {
        ChannelLayout::Gray => (6u8, ChannelLayout::Gray),
        ChannelLayout::Rgb => (50, ChannelLayout::Rgb),
        ChannelLayout::Rgba => (51, ChannelLayout::Rgba),
        l => return Err(CodecError::encode(F, format!("DPX cannot store {l:?}"))),
    };
    let bits = bits_for(opts.dpx_bits, img.sample_type());
    let codes: Vec<u16> = match (img.sample_type(), bits) {
        (SampleType::U8, 8) => img.data().iter().map(|&v| u16::from(v)).collect(),
        (SampleType::U16, 16) => img.to_u16_samples().unwrap_or_default(),
        (SampleType::U16, 10 | 12) => {
            let max = (1u64 << bits) - 1;
            img.to_u16_samples().unwrap_or_default().into_iter().map(|v| narrow(v, max)).collect()
        }
        (s, b) => return Err(CodecError::encode(F, format!("cannot write {s:?} samples at {b} bits"))),
    };
    let per_line = w as usize * layout.channels();
    if codes.len() != per_line * h as usize {
        return Err(CodecError::encode(F, "the image has the wrong number of samples"));
    }
    let data_len = line_bytes(per_line, bits, if matches!(bits, 10 | 12) { 1 } else { 0 })
        .and_then(|l| l.checked_mul(h as usize))
        .ok_or_else(|| CodecError::encode(F, "image too large"))?;
    let total = u32::try_from(HEADER_LEN + data_len).map_err(|_| CodecError::encode(F, "image too large for a DPX file (4 GiB)"))?;

    let mut wr = Writer { h: vec![0; HEADER_LEN] };
    // Generic file header.
    wr.h[..4].copy_from_slice(b"SDPX");
    wr.u32(4, HEADER_LEN as u32);
    wr.string((8, 8), "V2.0");
    wr.u32(16, total);
    wr.u32(20, 1);
    wr.u32(24, GENERIC_LEN);
    wr.u32(28, INDUSTRY_LEN);
    wr.u32(32, 0);
    let keep = |key| if opts.embed_metadata { text(&img, key) } else { None };
    wr.string(CREATION_TIME, keep(KEY_CREATION_TIME).unwrap_or(""));
    wr.string(CREATOR, keep(KEY_CREATOR).unwrap_or("PhotoCraft"));
    wr.string(PROJECT, keep(KEY_PROJECT).unwrap_or(""));
    wr.string(COPYRIGHT, keep(KEY_COPYRIGHT).unwrap_or(""));
    wr.u32(660, UNDEF32);
    // Image information: default orientation, one element.
    wr.u16(ORIENTATION, 0);
    wr.u16(ORIENTATION + 2, 1);
    wr.u32(772, w);
    wr.u32(776, h);
    let e = ELEMENT0;
    wr.u32(e, 0);
    wr.u32(e + 4, 0);
    wr.u32(e + 8, UNDEF32);
    wr.u32(e + 12, ((1u32 << bits) - 1).min(65535));
    wr.u32(e + 16, UNDEF32);
    wr.u8(e + 20, descriptor);
    // No claim about the encoding unless the document came with one.
    let code = |key| keep(key).and_then(|v| v.parse::<u8>().ok()).unwrap_or(UNDEF8);
    wr.u8(e + 21, code(KEY_TRANSFER));
    wr.u8(e + 22, code(KEY_COLORIMETRIC));
    wr.u8(e + 23, bits);
    wr.u16(e + 24, if matches!(bits, 10 | 12) { 1 } else { 0 });
    wr.u16(e + 26, 0);
    wr.u32(e + 28, HEADER_LEN as u32);
    wr.u32(e + 32, 0);
    wr.u32(e + 36, 0);
    // The seven unused elements, the image source header's numbers and the industry header's
    // numbers are undefined; their strings stay empty.
    wr.undefined(e + ELEMENT_LEN, e + ELEMENT_LEN * usize::from(MAX_ELEMENTS));
    wr.undefined(1408, 1432);
    wr.undefined(1620, 1644);
    wr.undefined(FRAME_POSITION, 1732);
    wr.undefined(TIME_CODE, 1972);
    if let Some(v) = keep(KEY_FRAME_POSITION).and_then(|v| v.parse::<u32>().ok()) {
        wr.u32(FRAME_POSITION, v);
    }
    if let Some(v) = keep(KEY_FRAME_RATE).and_then(|v| v.parse::<f32>().ok()).filter(|v| v.is_finite()) {
        wr.u32(FRAME_RATE, v.to_bits());
    }
    if let Some(v) = keep(KEY_TIME_CODE).and_then(parse_time_code) {
        wr.u32(TIME_CODE, v);
    }

    let mut out = wr.h;
    out.try_reserve_exact(data_len).map_err(|_| CodecError::encode(F, "out of memory"))?;
    for row in codes.chunks_exact(per_line.max(1)) {
        pack_line(&mut out, row, bits);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal header for `w`×`h` with element 0 set; image data at 2048.
    fn header(o: Order, w: u32, h: u32, descriptor: u8, bits: u8, packing: u16) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_LEN];
        let put32 = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&if o == Order::Big { v.to_be_bytes() } else { v.to_le_bytes() });
        let put16 = |b: &mut Vec<u8>, at: usize, v: u16| b[at..at + 2].copy_from_slice(&if o == Order::Big { v.to_be_bytes() } else { v.to_le_bytes() });
        b[..4].copy_from_slice(if o == Order::Big { b"SDPX" } else { b"XPDS" });
        put32(&mut b, 4, HEADER_LEN as u32);
        put32(&mut b, 28, INDUSTRY_LEN);
        put16(&mut b, ORIENTATION, 0);
        put16(&mut b, ORIENTATION + 2, 1);
        put32(&mut b, 772, w);
        put32(&mut b, 776, h);
        b[ELEMENT0 + 20] = descriptor;
        b[ELEMENT0 + 21] = UNDEF8;
        b[ELEMENT0 + 22] = UNDEF8;
        b[ELEMENT0 + 23] = bits;
        put16(&mut b, ELEMENT0 + 24, packing);
        put32(&mut b, ELEMENT0 + 28, HEADER_LEN as u32);
        b[FRAME_POSITION..1732].fill(0xFF);
        b[TIME_CODE..1972].fill(0xFF);
        b
    }

    fn words(o: Order, ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| if o == Order::Big { w.to_be_bytes() } else { w.to_le_bytes() }).collect()
    }

    #[test]
    fn ten_bit_method_a_and_b_unpack_three_to_a_word() {
        for o in [Order::Big, Order::Little] {
            for (packing, base) in [(1u16, 22u32), (2, 20)] {
                let mut b = header(o, 1, 1, 50, 10, packing);
                b.extend(words(o, &[(1023 << base) | (512 << (base - 10)) | (1 << (base - 20))]));
                let img = decode(&b, &Limits::default()).unwrap();
                assert_eq!(img.to_u16_samples().unwrap(), vec![65535, widen(512, 1023), widen(1, 1023)], "{o:?} packing {packing}");
            }
        }
    }

    #[test]
    fn ten_bit_packing_0_is_one_bit_stream_per_line() {
        // Four luma samples: 40 bits over two words, the line padded to the second word.
        let samples = [1u64, 2, 1022, 1023];
        let stream = samples.iter().fold(0u64, |acc, &s| (acc << 10) | s) << 24;
        for o in [Order::Big, Order::Little] {
            let mut b = header(o, 4, 2, 6, 10, 0);
            let line = words(o, &[(stream >> 32) as u32, stream as u32]);
            b.extend(&line);
            b.extend(&line);
            let img = decode(&b, &Limits::default()).unwrap();
            let want: Vec<u16> = samples.iter().map(|&s| widen(s as u16, 1023)).collect();
            assert_eq!(img.to_u16_samples().unwrap(), [want.clone(), want].concat(), "{o:?}");
        }
    }

    #[test]
    fn twelve_bit_method_a_keeps_the_top_bits() {
        let mut b = header(Order::Big, 2, 1, 6, 12, 1);
        b.extend_from_slice(&[0xFF, 0xF0, 0x00, 0x10]);
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.to_u16_samples().unwrap(), vec![65535, widen(1, 4095)]);
    }

    #[test]
    fn widen_and_narrow_round_trip_every_code() {
        for bits in [10u8, 12] {
            let max = (1u64 << bits) - 1;
            for c in 0..=max as u16 {
                assert_eq!(narrow(widen(c, max), max), c, "{bits}-bit {c}");
            }
        }
        assert_eq!(widen(512, 1023), ((512.0f64 * 65535.0 / 1023.0).round()) as u16);
    }

    #[test]
    fn orientation_codes_map_to_exif() {
        let mut b = header(Order::Big, 1, 1, 6, 8, 0);
        for (dpx, exif) in [(0u16, 1u16), (1, 2), (2, 4), (3, 3), (4, 5), (5, 6), (6, 8), (7, 7), (0xFFFF, 1)] {
            b[ORIENTATION..ORIENTATION + 2].copy_from_slice(&dpx.to_be_bytes());
            assert_eq!(orientation(&b), exif, "DPX {dpx}");
        }
    }

    #[test]
    fn time_codes_round_trip() {
        assert_eq!(format_time_code(0x0102_0304), "01:02:03:04");
        assert_eq!(parse_time_code("01:02:03:04"), Some(0x0102_0304));
        assert_eq!(parse_time_code(&format_time_code(0xABCD_0000)), Some(0xABCD_0000));
        assert_eq!(parse_time_code("1:2:3:4"), None);
    }

    #[test]
    fn eight_bit_lines_without_word_padding_still_open() {
        // 1 × 2 RGB: 3 bytes a line, 6 in all; padded lines would need 8.
        let mut b = header(Order::Big, 1, 2, 50, 8, 0);
        b.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.data(), &[1, 2, 3, 4, 5, 6]);
    }
}
