//! Bounded CR3 container reading, independently implemented from ISO BMFF and
//! Laurent Clévy's public format description (canon_cr3/readme.md). This extracts
//! camera JPEGs and orientation, not CRX sensor data. Never scan compressed data
//! for JPEG signatures: offsets must come from a validated sample table or PRVW.

use crate::preview::{Preview, candidate};
use crate::tiff::{Tiff, tag};

const META_UUID: &[u8] = &[0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0, 0x81, 0x11, 0xf4, 0xce, 0x46, 0x2b, 0x6a, 0x48];
const PREVIEW_UUID: &[u8] = &[0xea, 0xf4, 0x2b, 0x5e, 0x1c, 0x98, 0x4b, 0x88, 0xb9, 0xfb, 0xb7, 0xdc, 0x40, 0x6e, 0x4d, 0x16];
const MAX_BOXES: usize = 8192;
const MAX_DEPTH: usize = 12;

fn u32_at(b: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(p..p.checked_add(4)?)?.try_into().ok()?))
}

fn u64_at(b: &[u8], p: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(p..p.checked_add(8)?)?.try_into().ok()?))
}

struct BoxRef<'a> {
    kind: &'a [u8],
    body: &'a [u8],
}

/// Walks only complete sibling boxes, including extended lengths and a final
/// zero-length box. Invalid lengths end this container; siblings never escape it.
fn boxes<'a>(mut b: &'a [u8], budget: &mut usize, mut f: impl FnMut(BoxRef<'a>, &mut usize)) {
    while b.len() >= 8 && *budget > 0 {
        *budget -= 1;
        let Some(n) = u32_at(b, 0) else {
            return;
        };
        let (size, header) = match n {
            0 => (b.len(), 8),
            1 => {
                let Some(size) = u64_at(b, 8).and_then(|n| usize::try_from(n).ok()) else {
                    return;
                };
                (size, 16)
            }
            n => (n as usize, 8),
        };
        if size < header {
            return;
        }
        let Some(body) = b.get(header..size) else {
            return;
        };
        let Some(kind) = b.get(4..8) else {
            return;
        };
        f(BoxRef { kind, body }, budget);
        let Some(rest) = b.get(size..) else {
            return;
        };
        b = rest;
    }
}

#[derive(Default)]
struct Track<'a> {
    jpeg: bool,
    sizes: Option<&'a [u8]>,
    offsets: Option<(&'a [u8], bool)>,
    mapping: bool,
}

/// Accept only a single JPEG sample description, with CRAW's fixed sample-entry
/// header followed by an explicit JPEG codec box.
fn jpeg_description(b: &[u8], budget: &mut usize) -> bool {
    if u32_at(b, 0) != Some(0) || u32_at(b, 4) != Some(1) {
        return false;
    }
    let mut jpeg = false;
    if let Some(entries) = b.get(8..) {
        boxes(entries, budget, |entry, budget| {
            if entry.kind == b"CRAW"
                && let Some(children) = entry.body.get(82..)
            {
                boxes(children, budget, |codec, _| {
                    jpeg |= codec.kind == b"JPEG";
                });
            }
        });
    }
    jpeg
}

fn track_boxes<'a>(b: &'a [u8], depth: usize, budget: &mut usize, t: &mut Track<'a>) {
    if depth > MAX_DEPTH {
        return;
    }
    boxes(b, budget, |item, budget| match item.kind {
        b"mdia" | b"minf" | b"stbl" => track_boxes(item.body, depth + 1, budget, t),
        b"stsd" => t.jpeg = jpeg_description(item.body, budget),
        b"stsz" => t.sizes = Some(item.body),
        b"co64" => t.offsets = Some((item.body, true)),
        b"stco" => t.offsets = Some((item.body, false)),
        // One entry, starting at chunk 1, one sample per chunk, description 1.
        b"stsc" => t.mapping = item.body.len() == 20 && (0..5).all(|i| u32_at(item.body, i * 4) == Some(if i == 0 { 0 } else { 1 })),
        _ => {}
    });
}

fn track_preview<'a>(raw: &'a [u8], track: &'a [u8], budget: &mut usize, media: &[(usize, usize)]) -> Option<Preview<'a>> {
    let mut t = Track::default();
    track_boxes(track, 0, budget, &mut t);
    if !t.jpeg || !t.mapping {
        return None;
    }
    let sizes = t.sizes?;
    let (offsets, wide) = t.offsets?;
    // Burst sequences are deliberately not paired by ordinal index: their
    // sample-to-chunk mapping and picture selection need a separate implementation.
    if u32_at(sizes, 0) != Some(0) || u32_at(sizes, 8) != Some(1) || u32_at(offsets, 0) != Some(0) || u32_at(offsets, 4) != Some(1) {
        return None;
    }
    let fixed = u32_at(sizes, 4)?;
    let len = if fixed == 0 { u32_at(sizes, 12)? } else { fixed };
    let off = if wide { u64_at(offsets, 8)? } else { u64::from(u32_at(offsets, 8)?) };
    let off = usize::try_from(off).ok()?;
    let end = off.checked_add(len as usize)?;
    if !media.iter().any(|(start, stop)| off >= *start && end <= *stop) {
        return None;
    }
    candidate(raw, off, len as usize)
}

#[derive(Default)]
struct Found<'a> {
    previews: Vec<Preview<'a>>,
    orientation: Option<u16>,
    media: Vec<(usize, usize)>,
}

fn walk<'a>(raw: &'a [u8], b: &'a [u8], depth: usize, budget: &mut usize, found: &mut Found<'a>) {
    if depth > MAX_DEPTH {
        return;
    }
    boxes(b, budget, |item, budget| match item.kind {
        b"moov" => walk(raw, item.body, depth + 1, budget, found),
        b"trak" if found.previews.len() < 128 => {
            found.previews.extend(track_preview(raw, item.body, budget, &found.media));
        }
        b"uuid" => {
            if item.body.get(..16) == Some(META_UUID)
                && let Some(children) = item.body.get(16..)
            {
                walk(raw, children, depth + 1, budget, found);
            } else if item.body.get(..16) == Some(PREVIEW_UUID)
                && let Some(children) = item.body.get(24..)
            {
                walk(raw, children, depth + 1, budget, found);
            }
        }
        b"PRVW" => {
            if found.previews.len() < 128
                && let Some(len) = u32_at(item.body, 12)
                && let Some(p) = candidate(item.body, 16, len as usize)
            {
                found.previews.push(p);
            }
        }
        b"CMT1" => {
            if let Some(tiff) = Tiff::new(item.body)
                && let Some(ifd) = tiff.ifd_at(tiff.first_ifd, 0)
                && let Some(o) = tiff.tag_uint(&ifd, tag::ORIENTATION).filter(|o| (1..=8).contains(o))
            {
                found.orientation = Some(o as u16);
            }
        }
        _ => {}
    });
}

fn read(raw: &[u8]) -> Found<'_> {
    let mut found = Found::default();
    if raw.get(4..12) == Some(b"ftypcrx ") {
        let mut budget = MAX_BOXES;
        boxes(raw, &mut budget, |item, _| {
            if item.kind == b"mdat" && found.media.len() < 128 {
                let start = (item.body.as_ptr() as usize).saturating_sub(raw.as_ptr() as usize);
                if let Some(end) = start.checked_add(item.body.len()) {
                    found.media.push((start, end));
                }
            }
        });
        walk(raw, raw, 0, &mut budget, &mut found);
    }
    found
}

pub(crate) fn previews(raw: &[u8]) -> Vec<Preview<'_>> {
    read(raw).previews
}
pub(crate) fn orientation(raw: &[u8]) -> Option<u16> {
    read(raw).orientation
}
