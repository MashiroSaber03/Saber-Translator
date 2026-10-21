//! X PixMap reader for XPM2 plain-text icons and XPM3 C-source icon arrays.
//! Supports transparency (None), RGB hex at 4/8/16 bits per channel and
//! common X11 colour names. Does not execute C code.

use std::collections::HashMap;

mod x11_colors;

use crate::SampleType;
use crate::error::CodecError;
use crate::format::Format;
use crate::image::{ChannelLayout, Image};
use crate::options::Limits;

const MAX_SOURCE: usize = 256 << 20;
const MAX_COLOURS: usize = 65_536;
const MAX_CPP: usize = 8;

fn bad(msg: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(Format::Xpm, msg)
}

/// Extract C string literals without treating quoted text or // in a string as a comment.
/// XPM uses one literal for the header, each palette entry and each row.
fn c_strings(bytes: &[u8]) -> Result<Vec<Vec<u8>>, CodecError> {
    let mut lines = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"/*") {
            let tail = &bytes[i + 2..];
            let Some(end) = tail.windows(2).position(|p| p == b"*/") else { return Err(bad("unterminated C comment")) };
            i += end + 4;
        } else if bytes[i..].starts_with(b"//") {
            i += bytes[i..].iter().position(|&c| c == b'\n').unwrap_or(bytes.len() - i);
        } else if bytes[i] == b'"' {
            i += 1;
            let mut s = Vec::new();
            let mut closed = false;
            while i < bytes.len() {
                let c = bytes[i];
                i += 1;
                match c {
                    b'"' => {
                        closed = true;
                        break;
                    }
                    b'\\' => {
                        let next = *bytes.get(i).ok_or_else(|| bad("truncated C escape"))?;
                        i += 1;
                        s.push(match next {
                            b'n' => b'\n',
                            b'r' => b'\r',
                            b't' => b'\t',
                            b'0' => 0,
                            _ => next, // quoted backslash or quote
                        });
                    }
                    _ => s.push(c),
                }
            }
            if !closed {
                return Err(bad("unterminated XPM string"));
            }
            lines.push(s);
        } else {
            i += 1;
        }
    }
    Ok(lines)
}

/// Extract XPM2's plain-text lines without trimming pixel rows: leading and
/// trailing spaces can be palette keys, and physical CRLF is not image data.
fn xpm2_lines(bytes: &[u8]) -> Result<Vec<Vec<u8>>, CodecError> {
    let mut input = bytes.split(|&b| b == b'\n');
    let header = input.next().ok_or_else(|| bad("missing XPM2 marker"))?;
    if header.strip_suffix(b"\r").unwrap_or(header) != b"! XPM2" {
        return Err(bad("not an XPM2 image"));
    }
    let mut lines = Vec::new();
    for raw in input {
        let line = raw.strip_suffix(b"\r").unwrap_or(raw);
        // The header may be preceded by comments and blank lines. After the
        // header, a leading '!' is allowed as a palette key or pixel value.
        if lines.is_empty() && (line.is_empty() || line.starts_with(b"!")) {
            continue;
        }
        lines.push(line.to_vec());
    }
    Ok(lines)
}

fn channel(hex: &str) -> Option<u8> {
    let v = u32::from_str_radix(hex, 16).ok()?;
    let max = (1u32 << (hex.len() * 4)) - 1;
    Some(((v * 255 + max / 2) / max) as u8)
}

fn named(name: &str) -> Option<[u8; 3]> {
    let s = name.to_ascii_lowercase().replace([' ', '-', '_'], "");
    // XPM imports X11 names, whose RGB values differ from CSS for names
    // such as gray, maroon and purple. Keep these exact rather than guessing.
    if let Ok(index) = x11_colors::X11_COLORS.binary_search_by_key(&s.as_str(), |(name, _)| *name) {
        return Some(x11_colors::X11_COLORS[index].1);
    }
    // Retain useful web color aliases not present in the X11 database.
    Some(match s.as_str() {
        "silver" => [192, 192, 192],
        "olive" => [128, 128, 0],
        "teal" => [0, 128, 128],
        "fuchsia" => [255, 0, 255],
        "navyblue" => [0, 0, 128],
        _ => return None,
    })
}

fn color(name: &str) -> Option<[u8; 4]> {
    if name.eq_ignore_ascii_case("none") {
        return Some([0, 0, 0, 0]);
    }
    let rgb = if let Some(hex) = name.strip_prefix('#') {
        // UTF-8 byte lengths can match a supported hex width while slicing would panic.
        // `from_str_radix` would also take a leading `+`: accept hex digits only.
        if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let digits = match hex.len() {
            3 => 1,
            6 => 2,
            12 => 4,
            _ => return None,
        };
        let rgb = (0..3).map(|i| channel(&hex[i * digits..(i + 1) * digits])).collect::<Option<Vec<_>>>()?;
        [rgb[0], rgb[1], rgb[2]]
    } else {
        named(name)?
    };
    Some([rgb[0], rgb[1], rgb[2], 255])
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    if bytes.len() > MAX_SOURCE {
        return Err(CodecError::LimitExceeded("XPM source exceeds 256 MiB".into()));
    }
    let lines = if bytes.starts_with(b"/* XPM */") {
        c_strings(bytes)?
    } else if bytes.starts_with(b"! XPM2\n") || bytes.starts_with(b"! XPM2\r\n") {
        xpm2_lines(bytes)?
    } else {
        return Err(bad("not an XPM2/XPM3 image"));
    };
    decode_lines(&lines, limits)
}

fn decode_lines(lines: &[Vec<u8>], limits: &Limits) -> Result<Image, CodecError> {
    let header = lines.first().ok_or_else(|| bad("missing image header"))?;
    let parts = std::str::from_utf8(header).map_err(|_| bad("non-UTF-8 XPM header"))?;
    let mut parts = parts.split_ascii_whitespace();
    let number = |key: &str, s: Option<&str>| -> Result<usize, CodecError> {
        s.ok_or_else(|| bad(format!("missing {key}")))?.parse::<usize>().map_err(|_| bad(format!("invalid {key}")))
    };
    let width = number("width", parts.next())?;
    let height = number("height", parts.next())?;
    let colours = number("palette size", parts.next())?;
    let cpp = number("chars per pixel", parts.next())?;
    if width == 0 || height == 0 || colours == 0 || colours > MAX_COLOURS || !(1..=MAX_CPP).contains(&cpp) {
        return Err(bad("invalid XPM dimensions, palette size or chars per pixel"));
    }
    let (w, h) = (u32::try_from(width).map_err(|_| bad("width overflow"))?, u32::try_from(height).map_err(|_| bad("height overflow"))?);
    limits.check(w, h, ChannelLayout::Rgba, SampleType::U8)?;
    let required = 1usize.checked_add(colours).and_then(|n| n.checked_add(height)).ok_or_else(|| bad("row count overflow"))?;
    if lines.len() < required {
        return Err(bad("truncated palette or image rows"));
    }
    let mut palette = HashMap::<Vec<u8>, [u8; 4]>::with_capacity(colours);
    for line in &lines[1..1 + colours] {
        if line.len() < cpp {
            return Err(bad("truncated colour key"));
        }
        let fields = std::str::from_utf8(&line[cpp..]).map_err(|_| bad("invalid colour name"))?;
        // XPM priority is colour, gray, four-level gray, then monochrome,
        // regardless of the order of keys in the palette entry.
        let terms: Vec<&str> = fields.split_ascii_whitespace().collect();
        let mut picked: Option<(u8, String)> = None;
        let mut i = 0;
        while i < terms.len() {
            let priority = match terms[i] {
                "c" => 4,
                "g" => 3,
                "g4" => 2,
                "m" => 1,
                _ => 0, // Symbolic names ('s') are not colour values.
            };
            let start = i + 1;
            i = start;
            while i < terms.len() && !matches!(terms[i], "c" | "g" | "g4" | "m" | "s") {
                i += 1;
            }
            if priority > 0 && i > start && picked.as_ref().is_none_or(|(current, _)| priority > *current) {
                picked = Some((priority, terms[start..i].join(" ")));
            }
        }
        let value = picked.as_ref().and_then(|(_, name)| color(name)).ok_or_else(|| bad("unsupported or missing XPM colour"))?;
        if palette.insert(line[..cpp].to_vec(), value).is_some() {
            return Err(bad("duplicate XPM palette key"));
        }
    }
    let row_len = width.checked_mul(cpp).ok_or_else(|| bad("row byte count overflow"))?;
    let mut pixels = vec![0u8; width.checked_mul(height).and_then(|n| n.checked_mul(4)).ok_or_else(|| bad("pixel size overflow"))?];
    for (y, line) in lines[1 + colours..required].iter().enumerate() {
        if line.len() != row_len {
            return Err(bad("XPM row has the wrong width"));
        }
        for x in 0..width {
            let key = &line[x * cpp..(x + 1) * cpp];
            let rgba = palette.get(key).ok_or_else(|| bad("unknown palette key"))?;
            pixels[(y * width + x) * 4..(y * width + x + 1) * 4].copy_from_slice(rgba);
        }
    }
    Image::from_u8(w, h, ChannelLayout::Rgba, pixels)
}
