//! FITS (Flexible Image Transport System, FITS Standard 4.0), implemented here: the image in the
//! primary HDU, or the first `IMAGE` extension when the primary HDU holds no data.
//!
//! Reading:
//!
//! * `BITPIX` 8, 16, 32, 64, -32 and -64, with `BSCALE`/`BZERO` and integer `BLANK`.
//! * `NAXIS3 = 3` opens as RGB (the planes are R, G, B); any other plane count opens the first
//!   plane as gray with [`DecodeWarning::MorePages`].
//! * Rows are stored bottom-to-top, as the standard's convention has it (Siril, DS9, Astropy),
//!   unless `ROWORDER = 'TOP-DOWN'`.
//! * Physical values that fit (unsigned 8-bit, or integers in 0..=65535 such as a camera's
//!   `BITPIX = 16`, `BZERO = 32768`) are kept exactly as U8/U16. Everything else becomes F32:
//!   values up to 2 are taken as already normalised (Siril, PixInsight), larger ones are scaled by
//!   65535 when they fit in it (DeepSkyStacker, most stackers) and by the maximum otherwise. NaN
//!   and blank pixels read as 0.
//! * The other header cards are kept as text: `KEY` → value (strings unquoted), `COMMENT` and
//!   `HISTORY` by their text, `HIERARCH` keywords by their long name. Long strings continued
//!   on `CONTINUE` cards are joined.
//!
//! Writing: gray or RGB, U8 (`BITPIX = 8`), U16 (`BITPIX = 16`, `BZERO = 32768`) or F32
//! (`BITPIX = -32`, 0..1), bottom-to-top with `ROWORDER = 'BOTTOM-UP'`, plus the text cards
//! (strings longer than one card on `CONTINUE` cards).
//! Tile-compressed files (fpack `ZIMAGE` tables) are recognised and refused.

use crate::Format;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, DecodeWarning, Image, SampleType};
use crate::options::{EncodeOptions, Limits};

const F: Format = Format::Fits;
const BLOCK: usize = 2880;
const CARD: usize = 80;
/// Most HDUs walked looking for an image; FITS files from cameras and stackers hold a handful.
const MAX_HDUS: usize = 1000;

/// Keywords that describe the data layout. They are written from the image, never copied from
/// (or kept as) text metadata: after decoding they no longer describe the pixels.
const STRUCTURAL: &[&str] = &[
    "SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BZERO", "BSCALE", "BLANK", "END", "XTENSION", "PCOUNT", "GCOUNT", "ROWORDER", "CHECKSUM", "DATASUM", "DATAMIN",
    "DATAMAX", "ZIMAGE",
];

fn err(msg: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, msg)
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    /// A quoted string, unescaped, trailing blanks removed.
    Str(String),
    /// Anything else (number, logical, complex) as written.
    Raw(String),
}

#[derive(Debug, Clone)]
struct Card {
    key: String,
    /// `None` for commentary cards (`COMMENT`, `HISTORY`, blank keyword).
    value: Option<Value>,
    /// The text of a commentary card.
    text: String,
    /// A `HIERARCH` keyword (`key` holds its long name).
    hierarch: bool,
}

fn parse_value(field: &str) -> Value {
    let t = field.trim_start();
    if let Some(rest) = t.strip_prefix('\'') {
        let mut s = String::new();
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\'' {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    s.push('\'');
                } else {
                    break;
                }
            } else {
                s.push(c);
            }
        }
        Value::Str(s.trim_end().to_string())
    } else {
        Value::Raw(t.split('/').next().unwrap_or("").trim().to_string())
    }
}

fn parse_card(raw: &[u8]) -> Card {
    // Header bytes are ASCII 32–126 by the standard; anything else is replaced, never trusted.
    let s: String = raw.iter().map(|&b| if (32..=126).contains(&b) { b as char } else { '?' }).collect();
    let key_field = s.get(..8).unwrap_or(&s).trim_end().to_string();
    if key_field == "HIERARCH" {
        let rest = s.get(9..).unwrap_or("");
        if let Some((k, v)) = rest.split_once('=') {
            return Card { key: k.trim().to_string(), value: Some(parse_value(v)), text: String::new(), hierarch: true };
        }
    }
    if s.get(8..10) == Some("= ") {
        let v = parse_value(s.get(10..).unwrap_or(""));
        return Card { key: key_field, value: Some(v), text: String::new(), hierarch: false };
    }
    Card { key: key_field, value: None, text: s.get(8..).unwrap_or("").trim_end().to_string(), hierarch: false }
}

struct Header {
    cards: Vec<Card>,
    /// Offset of the data unit.
    data_start: usize,
}

impl Header {
    fn get(&self, key: &str) -> Option<&Value> {
        self.cards.iter().find(|c| !c.hierarch && c.key == key).and_then(|c| c.value.as_ref())
    }

    fn int(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            Value::Raw(r) => r.parse::<i64>().ok().or_else(|| self.float(key).filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15).map(|f| f as i64)),
            Value::Str(_) => None,
        }
    }

    fn float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            // Fortran-style `D` exponents are allowed in FITS.
            Value::Raw(r) => r.replace(['D', 'd'], "E").parse::<f64>().ok().filter(|f| f.is_finite()),
            Value::Str(_) => None,
        }
    }

    fn logical(&self, key: &str) -> bool {
        matches!(self.get(key), Some(Value::Raw(r)) if r == "T")
    }

    fn string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            Value::Str(s) => Some(s.trim()),
            Value::Raw(_) => None,
        }
    }
}

fn parse_header(b: &[u8], start: usize) -> Result<Header, CodecError> {
    let mut cards = Vec::new();
    let mut pos = start;
    loop {
        let raw = b.get(pos..pos.checked_add(CARD).ok_or_else(|| err("header too long"))?).ok_or_else(|| err("header ends before END"))?;
        pos += CARD;
        if raw.starts_with(b"END") && raw.get(3..).is_some_and(|r| r.iter().all(|&c| c == b' ')) {
            break;
        }
        let card = parse_card(raw);
        // Long strings (FITS 4.0 section 4.2.1.2): a string ending in '&' goes on in the next
        // `CONTINUE` record. An orphaned `CONTINUE` stays a card of its own (commentary).
        if let Some(Card { value: Some(Value::Str(s)), .. }) = cards.last_mut()
            && s.ends_with('&')
            && card.key == "CONTINUE"
            && card.text.starts_with("  ")
            && let Value::Str(more) = parse_value(&card.text)
        {
            s.pop();
            s.push_str(&more);
            continue;
        }
        cards.push(card);
    }
    let data_start = pos.div_ceil(BLOCK).checked_mul(BLOCK).ok_or_else(|| err("header too long"))?;
    Ok(Header { cards, data_start })
}

/// The axis lengths (`NAXIS1`, `NAXIS2`, …).
fn axes(h: &Header) -> Result<Vec<u64>, CodecError> {
    let n = h.int("NAXIS").ok_or_else(|| err("missing NAXIS"))?;
    if !(0..=999).contains(&n) {
        return Err(err(format!("invalid NAXIS {n}")));
    }
    (1..=n).map(|i| h.int(&format!("NAXIS{i}")).and_then(|v| u64::try_from(v).ok()).ok_or_else(|| err(format!("missing or negative NAXIS{i}")))).collect()
}

/// Bytes in the data unit (before padding to a block).
fn data_len(h: &Header, bitpix: i64) -> Result<u64, CodecError> {
    let ax = axes(h)?;
    if ax.is_empty() {
        return Ok(0);
    }
    let pcount = u64::try_from(h.int("PCOUNT").unwrap_or(0)).map_err(|_| err("negative PCOUNT"))?;
    let gcount = u64::try_from(h.int("GCOUNT").unwrap_or(1)).map_err(|_| err("negative GCOUNT"))?;
    let elems = ax.iter().try_fold(1u64, |a, &n| a.checked_mul(n)).ok_or_else(|| err("data size overflows"))?;
    elems
        .checked_add(pcount)
        .and_then(|v| v.checked_mul(gcount))
        .and_then(|v| v.checked_mul(bitpix.unsigned_abs() / 8))
        .ok_or_else(|| err("data size overflows"))
}

/// Finds the image HDU: the primary one when it has at least two axes, else the first `IMAGE`
/// extension. Returns its header and the primary header (for its descriptive cards).
fn find_image(b: &[u8]) -> Result<(Header, Option<Header>), CodecError> {
    let primary = parse_header(b, 0)?;
    if primary.get("SIMPLE").is_none() {
        return Err(err("missing SIMPLE"));
    }
    if axes(&primary)?.len() >= 2 {
        return Ok((primary, None));
    }
    let mut compressed = false;
    let mut cur_start = primary.data_start;
    let mut cur_len = data_len(&primary, primary.int("BITPIX").unwrap_or(8))?;
    for _ in 0..MAX_HDUS {
        let next = usize::try_from(cur_len.div_ceil(BLOCK as u64).saturating_mul(BLOCK as u64))
            .ok()
            .and_then(|l| cur_start.checked_add(l))
            .ok_or_else(|| err("data size overflows"))?;
        if next >= b.len() {
            break;
        }
        let h = parse_header(b, next)?;
        let kind = h.string("XTENSION").unwrap_or("").to_string();
        if kind == "IMAGE" && axes(&h)?.len() >= 2 {
            return Ok((h, Some(primary)));
        }
        if kind == "BINTABLE" && h.logical("ZIMAGE") {
            compressed = true;
        }
        cur_len = data_len(&h, h.int("BITPIX").unwrap_or(8))?;
        cur_start = h.data_start;
    }
    if compressed {
        return Err(CodecError::unsupported(F, "tile-compressed FITS (fpack) is not supported; decompress it with funpack first"));
    }
    Err(err("the file holds no image"))
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    let (h, primary) = find_image(bytes)?;
    let bitpix = h.int("BITPIX").ok_or_else(|| err("missing BITPIX"))?;
    if !matches!(bitpix, 8 | 16 | 32 | 64 | -32 | -64) {
        return Err(err(format!("invalid BITPIX {bitpix}")));
    }
    let ax = axes(&h)?;
    let (Some(&w), Some(&hgt)) = (ax.first(), ax.get(1)) else { return Err(err("the image has fewer than two axes")) };
    let w = u32::try_from(w).map_err(|_| CodecError::LimitExceeded(format!("width {w}")))?;
    let hgt = u32::try_from(hgt).map_err(|_| CodecError::LimitExceeded(format!("height {hgt}")))?;
    let planes = ax.iter().skip(2).try_fold(1u64, |a, &n| a.checked_mul(n)).ok_or_else(|| err("too many planes"))?;
    let (layout, nplanes) = if ax.get(2) == Some(&3) && planes == 3 { (ChannelLayout::Rgb, 3usize) } else { (ChannelLayout::Gray, 1) };

    let bscale = h.float("BSCALE").unwrap_or(1.0);
    let bzero = h.float("BZERO").unwrap_or(0.0);
    let blank = if bitpix > 0 { h.int("BLANK") } else { None };
    let integral = bscale == 1.0 && bzero.fract() == 0.0;
    let candidate = match bitpix {
        8 if integral && bzero == 0.0 => SampleType::U8,
        8 | 16 if integral => SampleType::U16,
        _ => SampleType::F32,
    };
    limits.check(w, hgt, layout, candidate)?;

    let bps = (bitpix.unsigned_abs() / 8) as usize;
    let plane_len = (w as usize).checked_mul(hgt as usize).ok_or_else(|| CodecError::LimitExceeded(format!("{w}x{hgt} does not fit in memory")))?;
    let n = plane_len.checked_mul(nplanes).ok_or_else(|| CodecError::LimitExceeded("image does not fit in memory".into()))?;
    let need = n.checked_mul(bps).and_then(|l| l.checked_add(h.data_start)).ok_or_else(|| err("data size overflows"))?;
    let data = bytes.get(h.data_start..need).ok_or_else(|| err("truncated data"))?;

    // Physical value of sample `i` (planar order); `None` for a blank or NaN pixel.
    let physical = |i: usize| -> Option<f64> {
        let s = data.get(i * bps..(i + 1) * bps)?;
        let raw = match bitpix {
            8 => s.first().map(|&v| v as i64)?,
            16 => i16::from_be_bytes([s[0], s[1]]) as i64,
            32 => i32::from_be_bytes([s[0], s[1], s[2], s[3]]) as i64,
            64 => i64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
            -32 => return Some(f32::from_be_bytes([s[0], s[1], s[2], s[3]]) as f64 * bscale + bzero).filter(|v| v.is_finite()),
            _ => return Some(f64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) * bscale + bzero).filter(|v| v.is_finite()),
        };
        if blank == Some(raw) {
            return None;
        }
        Some(raw as f64 * bscale + bzero)
    };

    let top_down = h.string("ROWORDER").is_some_and(|r| r.eq_ignore_ascii_case("TOP-DOWN"));
    let (wu, hu) = (w as usize, hgt as usize);
    // Planar, bottom-up (by default) → interleaved, top-down.
    let dest = |i: usize| -> usize {
        let (p, rem) = (i / plane_len, i % plane_len);
        let (y, x) = (rem / wu, rem % wu);
        let y = if top_down { y } else { hu - 1 - y };
        (y * wu + x) * nplanes + p
    };

    let mut img = match candidate {
        SampleType::U8 => {
            let mut out = vec![0u8; n];
            for i in 0..n {
                out[dest(i)] = physical(i).unwrap_or(0.0) as u8;
            }
            Image::from_u8(w, hgt, layout, out)?
        }
        _ => {
            let mut out16 = if candidate == SampleType::U16 { Some(vec![0u16; n]) } else { None };
            if let Some(out) = out16.as_mut() {
                for i in 0..n {
                    let v = physical(i).unwrap_or(0.0);
                    if !(0.0..=65535.0).contains(&v) {
                        // Signed data with negative values: fall back to float.
                        out16 = None;
                        break;
                    }
                    out[dest(i)] = v as u16;
                }
            }
            match out16 {
                Some(out) => Image::from_u16(w, hgt, layout, &out)?,
                None => {
                    let max = (0..n).filter_map(physical).fold(0.0f64, f64::max);
                    let white = if max <= 2.0 {
                        1.0
                    } else if max <= 65535.0 {
                        65535.0
                    } else {
                        max
                    };
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[dest(i)] = (physical(i).unwrap_or(0.0) / white) as f32;
                    }
                    Image::from_f32(w, hgt, layout, &out)?
                }
            }
        }
    };

    if planes > nplanes as u64 {
        img.warnings.push(DecodeWarning::MorePages { total: u32::try_from(planes).ok() });
    }
    let mut text = Vec::new();
    for c in primary.iter().flat_map(|p| p.cards.iter()).chain(h.cards.iter()) {
        if !c.hierarch && (STRUCTURAL.contains(&c.key.as_str()) || is_naxis(&c.key)) {
            continue;
        }
        let entry = match &c.value {
            Some(Value::Str(s) | Value::Raw(s)) => (c.key.clone(), s.clone()),
            None if c.key.is_empty() && c.text.trim().is_empty() => continue,
            None => (if c.key.is_empty() { "COMMENT".to_string() } else { c.key.clone() }, c.text.trim().to_string()),
        };
        text.push(entry);
    }
    img.meta.text = text;
    Ok(img)
}

fn is_naxis(key: &str) -> bool {
    key.strip_prefix("NAXIS").is_some_and(|d| d.chars().all(|c| c.is_ascii_digit()))
}

/// A standard keyword: 1–8 characters from `A-Z 0-9 _ -`.
fn standard_key(key: &str) -> bool {
    (1..=8).contains(&key.len()) && key.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

fn ascii(s: &str) -> String {
    s.chars().map(|c| if (' '..='~').contains(&c) { c } else { '?' }).collect()
}

/// A FITS literal (number or logical) that can be written unquoted.
fn literal(v: &str) -> bool {
    v == "T" || v == "F" || (v.replace(['D', 'd'], "E").parse::<f64>().is_ok_and(|f| f.is_finite()) && v.bytes().all(|b| b"0123456789+-.EeDd".contains(&b)))
}

fn quoted(v: &str, room: usize) -> String {
    let mut s = String::from("'");
    for c in ascii(v).chars() {
        let add = if c == '\'' { 2 } else { 1 };
        // Keep room for the closing quote.
        if s.len() + add + 1 > room {
            break;
        }
        s.push(c);
        if c == '\'' {
            s.push('\'');
        }
    }
    // Strings are at least eight characters inside the quotes.
    while s.len() < 9 {
        s.push(' ');
    }
    s.push('\'');
    s
}

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn card(&mut self, text: &str) {
        let mut c: Vec<u8> = ascii(text).into_bytes();
        c.truncate(CARD);
        c.resize(CARD, b' ');
        self.out.extend_from_slice(&c);
    }

    fn value(&mut self, key: &str, v: &str) {
        if literal(v) {
            self.card(&format!("{key:<8}= {v:>20}"));
            return;
        }
        // A string too long for one card goes on in `CONTINUE` cards, every piece but the last
        // ending in '&' (FITS 4.0 section 4.2.1.2). A doubled quote is never split, and trailing
        // blanks (not significant) are dropped so none are kept before an '&'.
        let escaped = ascii(v).trim_end().replace('\'', "''");
        let mut rest = escaped.as_str();
        let mut head = format!("{key:<8}= ");
        while rest.len() > CARD - 12 {
            let cut = CARD - 13 - rest.get(..CARD - 13).unwrap_or("").bytes().rev().take_while(|&b| b == b'\'').count() % 2;
            let (piece, tail) = rest.split_at_checked(cut).unwrap_or((rest, ""));
            self.card(&format!("{head}'{piece}&'"));
            head = "CONTINUE  ".to_string();
            rest = tail;
        }
        self.card(&format!("{head}'{rest:<8}'"));
    }

    fn commentary(&mut self, key: &str, text: &str) {
        let chars: Vec<char> = ascii(text).chars().collect();
        if chars.is_empty() {
            self.card(key);
        }
        for chunk in chars.chunks(CARD - 8) {
            self.card(&format!("{key:<8}{}", chunk.iter().collect::<String>()));
        }
    }

    fn text(&mut self, key: &str, v: &str) {
        let upper = key.to_ascii_uppercase();
        if matches!(upper.as_str(), "COMMENT" | "HISTORY") {
            self.commentary(&upper, v);
        } else if standard_key(key) {
            // `CONTINUE` only ever carries a string's continuation, written by `value`.
            if !STRUCTURAL.contains(&key) && !is_naxis(key) && key != "CONTINUE" {
                self.value(key, v);
            }
        } else {
            // Long or mixed-case names (`Description`, `Software`): the HIERARCH convention,
            // when the card fits, else a comment.
            let name = ascii(key);
            let head = format!("HIERARCH {name} = ");
            let fits_key = !name.contains('=') && !name.trim().is_empty();
            let body = if literal(v) { v.to_string() } else { quoted(v, CARD.saturating_sub(head.len())) };
            if fits_key && head.len() + body.len() <= CARD && (literal(v) || body.ends_with('\'')) && quoted_whole(v, &body) {
                self.card(&format!("{head}{body}"));
            } else {
                self.commentary("COMMENT", &format!("{key} = {v}"));
            }
        }
    }
}

/// `body` holds all of `v` (it was not cut short to fit the card).
fn quoted_whole(v: &str, body: &str) -> bool {
    literal(v) || body.trim_matches('\'').trim_end().replace("''", "'") == ascii(v).trim_end()
}

pub(crate) fn encode(src: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let img = src.converted(plan.layout, plan.sample);
    let (w, h) = img.dimensions();
    let layout = img.layout();
    let nc = match layout {
        ChannelLayout::Gray => 1,
        ChannelLayout::Rgb => 3,
        l => return Err(CodecError::encode(F, format!("FITS cannot store {l:?}"))),
    };
    let bitpix = match img.sample_type() {
        SampleType::U8 => 8,
        SampleType::U16 => 16,
        SampleType::F32 => -32,
        s => return Err(CodecError::encode(F, format!("unsupported sample {s:?}"))),
    };
    let mut wr = Writer { out: Vec::new() };
    wr.value("SIMPLE", "T");
    wr.value("BITPIX", &bitpix.to_string());
    wr.value("NAXIS", if nc == 3 { "3" } else { "2" });
    wr.value("NAXIS1", &w.to_string());
    wr.value("NAXIS2", &h.to_string());
    if nc == 3 {
        wr.value("NAXIS3", "3");
    }
    if bitpix == 16 {
        wr.value("BZERO", "32768");
        wr.value("BSCALE", "1");
    }
    wr.value("ROWORDER", "BOTTOM-UP");
    if opts.embed_metadata {
        for (k, v) in &img.meta.text {
            wr.text(k, v);
        }
    }
    wr.card("END");
    let mut out = wr.out;
    out.resize(out.len().div_ceil(BLOCK) * BLOCK, b' ');

    // Planar, bottom row first.
    let (wu, hu) = (w as usize, h as usize);
    let idx = |p: usize, y: usize, x: usize| ((hu - 1 - y) * wu + x) * nc + p;
    match bitpix {
        8 => {
            let d = img.data();
            for p in 0..nc {
                for y in 0..hu {
                    for x in 0..wu {
                        out.push(d[idx(p, y, x)]);
                    }
                }
            }
        }
        16 => {
            let d = img.to_u16_samples().unwrap_or_default();
            for p in 0..nc {
                for y in 0..hu {
                    for x in 0..wu {
                        out.extend_from_slice(&((d[idx(p, y, x)] as i32 - 32768) as i16).to_be_bytes());
                    }
                }
            }
        }
        _ => {
            let d = img.to_f32_samples().unwrap_or_default();
            for p in 0..nc {
                for y in 0..hu {
                    for x in 0..wu {
                        out.extend_from_slice(&d[idx(p, y, x)].to_be_bytes());
                    }
                }
            }
        }
    }
    out.resize(out.len().div_ceil(BLOCK) * BLOCK, 0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(cards: &[&str]) -> Vec<u8> {
        let mut w = Writer { out: Vec::new() };
        for c in cards {
            w.card(c);
        }
        w.card("END");
        let mut out = w.out;
        out.resize(out.len().div_ceil(BLOCK) * BLOCK, b' ');
        out
    }

    #[test]
    fn continue_strings_decode() {
        let mut b = header(&[
            "SIMPLE  = T",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 1",
            "NAXIS2  = 1",
            "OBJECT  = 'aaa&'",
            "CONTINUE  'bbb&' / a comment",
            "CONTINUE  '&'",
            "CONTINUE  ''",
            "OBSERVER= 'R&'",
            "COMMENT hi",
            "CONTINUE  'orphan'",
        ]);
        b.push(0);
        let img = decode(&b, &Limits::default()).unwrap();
        let text = [("OBJECT", "aaabbb"), ("OBSERVER", "R&"), ("COMMENT", "hi"), ("CONTINUE", "'orphan'")];
        assert_eq!(img.meta.text, text.map(|(k, v)| (k.to_string(), v.to_string())));
    }

    #[test]
    fn continue_strings_round_trip() {
        for value in ["a".repeat(100), format!("{}' {}", "a".repeat(65), "b".repeat(100)), "'".repeat(100), "a".repeat(68)] {
            let mut img = Image::from_u8(1, 1, ChannelLayout::Gray, vec![0]).unwrap();
            img.meta.text = vec![("OBJECT".into(), value.clone()), ("CONTINUE".into(), "orphan".into())];
            let b = crate::encode(&img, F, &EncodeOptions::default()).unwrap();
            assert!(!b.chunks(CARD).any(|c| c.starts_with(b"CONTINUE=")));
            let decoded = decode(&b, &Limits::default()).unwrap();
            assert_eq!(decoded.meta.text, vec![("OBJECT".into(), value)]);
        }
    }

    #[test]
    fn camera_u16_with_bzero_reads_exactly_and_bottom_up() {
        let mut b = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    2",
            "NAXIS2  =                    2",
            "BZERO   =                32768",
            "OBJECT  = 'M 31    '",
            "EXPTIME =                300.0 / seconds",
        ]);
        for v in [0u16, 1000, 40000, 65535] {
            b.extend_from_slice(&((v as i32 - 32768) as i16).to_be_bytes());
        }
        b.resize(b.len().div_ceil(BLOCK) * BLOCK, 0);
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.sample_type(), SampleType::U16);
        // The first stored row is the bottom one.
        assert_eq!(img.to_u16_samples().unwrap(), vec![40000, 65535, 0, 1000]);
        assert!(img.meta.text.contains(&("OBJECT".into(), "M 31".into())));
        assert!(img.meta.text.contains(&("EXPTIME".into(), "300.0".into())));
        assert!(!img.meta.text.iter().any(|(k, _)| k == "BZERO" || k == "NAXIS1"));
    }

    #[test]
    fn roworder_top_down_is_honoured() {
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 2", "NAXIS1  = 1", "NAXIS2  = 2", "ROWORDER= 'TOP-DOWN'"]);
        b.extend_from_slice(&[10, 20]);
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.data(), &[10, 20]);
    }

    #[test]
    fn stacked_float_in_adu_is_scaled_by_65535() {
        let mut b = header(&["SIMPLE  = T", "BITPIX  = -32", "NAXIS   = 2", "NAXIS1  = 2", "NAXIS2  = 1"]);
        for v in [65535.0f32, f32::NAN] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.to_f32_samples().unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn signed_negative_data_falls_back_to_float() {
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 16", "NAXIS   = 2", "NAXIS1  = 2", "NAXIS2  = 1"]);
        for v in [-100i16, 1000] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.sample_type(), SampleType::F32);
    }

    #[test]
    fn image_extension_after_empty_primary() {
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 0", "EXTEND  = T", "TELESCOP= 'Seestar S50'"]);
        b.extend(header(&["XTENSION= 'IMAGE   '", "BITPIX  = 8", "NAXIS   = 2", "NAXIS1  = 1", "NAXIS2  = 1", "PCOUNT  = 0", "GCOUNT  = 1"]));
        b.push(42);
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.data(), &[42]);
        assert!(img.meta.text.contains(&("TELESCOP".into(), "Seestar S50".into())));
    }

    #[test]
    fn fpack_is_refused_as_unsupported() {
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 0"]);
        b.extend(header(&["XTENSION= 'BINTABLE'", "BITPIX  = 8", "NAXIS   = 2", "NAXIS1  = 0", "NAXIS2  = 0", "ZIMAGE  = T"]));
        b.extend(header(&["XTENSION= 'BINTABLE'", "BITPIX  = 8", "NAXIS   = 1", "NAXIS1  = 0"]));
        assert!(matches!(decode(&b, &Limits::default()), Err(CodecError::Unsupported { .. })));
    }

    /// Hostile headers: absurd or negative axes, bad BITPIX, overflowing sizes, missing END,
    /// a long chain of empty HDUs and declared data past the end all error, never panic or
    /// allocate the declared size.
    #[test]
    fn hostile_headers_error_without_panicking() {
        let base = ["SIMPLE  = T", "BITPIX  = 16"];
        let cases: &[&[&str]] = &[
            &["NAXIS   = 2", "NAXIS1  = 4294967296", "NAXIS2  = 1"],
            &["NAXIS   = 2", "NAXIS1  = -5", "NAXIS2  = 1"],
            &["NAXIS   = 2", "NAXIS1  = 2000000", "NAXIS2  = 2000000"],
            &["NAXIS   = 3", "NAXIS1  = 65536", "NAXIS2  = 65536", "NAXIS3  = 9223372036854775807"],
            &["NAXIS   = 1000"],
            &["NAXIS   = -1"],
            &["NAXIS   = 2", "NAXIS1  = 0", "NAXIS2  = 0"],
            &["NAXIS   = 2", "NAXIS1  = 1.5D300", "NAXIS2  = 1"],
            &["NAXIS   = 2"],
            &["NAXIS   = 2", "NAXIS1  = 100", "NAXIS2  = 100"],
        ];
        for extra in cases {
            let cards: Vec<&str> = base.iter().chain(extra.iter()).copied().collect();
            let b = header(&cards);
            assert!(decode(&b, &Limits::default()).is_err(), "{extra:?}");
        }
        for bitpix in ["0", "7", "-16", "128", "'X'"] {
            let card = format!("BITPIX  = {bitpix}");
            let b = header(&["SIMPLE  = T", &card, "NAXIS   = 2", "NAXIS1  = 1", "NAXIS2  = 1"]);
            assert!(decode(&b, &Limits::default()).is_err(), "BITPIX {bitpix}");
        }
        // No END card: the header runs off the end of the file.
        let mut w = Writer { out: Vec::new() };
        for c in ["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 2"] {
            w.card(c);
        }
        assert!(decode(&w.out, &Limits::default()).is_err());
        // An empty primary followed by many empty extensions: bounded walk, then "no image".
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 0"]);
        for _ in 0..(MAX_HDUS + 5) {
            b.extend(header(&["XTENSION= 'IMAGE   '", "BITPIX  = 8", "NAXIS   = 0"]));
        }
        assert!(decode(&b, &Limits::default()).is_err());
        // An extension whose declared data size overflows.
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 0"]);
        b.extend(header(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 64",
            "NAXIS   = 2",
            "NAXIS1  = 9223372036854775807",
            "NAXIS2  = 4",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
        ]));
        assert!(decode(&b, &Limits::default()).is_err());
    }

    #[test]
    fn extra_planes_warn() {
        let mut b = header(&["SIMPLE  = T", "BITPIX  = 8", "NAXIS   = 3", "NAXIS1  = 1", "NAXIS2  = 1", "NAXIS3  = 4"]);
        b.extend_from_slice(&[1, 2, 3, 4]);
        let img = decode(&b, &Limits::default()).unwrap();
        assert_eq!(img.layout(), ChannelLayout::Gray);
        assert_eq!(img.warnings, vec![DecodeWarning::MorePages { total: Some(4) }]);
    }

    #[test]
    fn strings_with_quotes_and_hierarch_round_trip() {
        let mut w = Writer { out: Vec::new() };
        w.text("OBSERVER", "O'Brien");
        w.text("Software", "photocraft 1.0");
        w.text("HISTORY", "stacked");
        let cards: Vec<Card> = w.out.chunks(CARD).map(parse_card).collect();
        assert_eq!(cards[0].value, Some(Value::Str("O'Brien".into())));
        assert!(cards[1].hierarch && cards[1].key == "Software");
        assert_eq!(cards[1].value, Some(Value::Str("photocraft 1.0".into())));
        assert_eq!((cards[2].key.as_str(), cards[2].text.as_str()), ("HISTORY", "stacked"));
    }
}
