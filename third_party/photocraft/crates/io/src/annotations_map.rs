//! Notes and Image › Analysis data ⇄ PSD.
//!
//! * Notes: the global additional-info block `Anno` ("Annotations (Photoshop 6.0)" in the Adobe
//!   PSD spec): `u16` major version 2, `u16` minor version 1, `u32` count, then per annotation a
//!   `u32` length (including itself), type `txtA` (text) or `sndA` (sound), open flag `u8`,
//!   flags `u8`, optional blocks `u16`, icon rect and popup rect (four `i32` each: top, left,
//!   bottom, right), a colour (`u16` space + four `u16`), author, name and modification date as
//!   even-padded Pascal strings, then a `u32` length (including itself) of the data block:
//!   `txtC`/`sndM`, `u32` size and the data. Text is UTF-16BE with a byte-order mark, lines
//!   separated by CR. Layout checked against `corpus/psd/ag-psd/read-write/annotations`.
//!   Sound annotations are not modelled; they are carried over verbatim when notes are rewritten.
//! * Measurement scale: image resource 1074 (a version-16 descriptor). The key names
//!   (`pixelLength`, `logicalLength`, `logicalUnits`) follow Photoshop's scripting names for the
//!   measurement scale; no sample file was available, so parsing is lenient and the raw resource
//!   is kept byte-exact while the scale is unchanged.
//! * Count information: image resource 1080 is kept verbatim (its layout is not public) unless
//!   the document has its own count markers, which only `.pcraft` stores.

use std::sync::Arc;

use photocraft_doc::{Color, Document, MeasurementScale, Note};
use photocraft_psd::descriptor::{Descriptor, Value, VersionedDescriptor};

pub const MEASUREMENT_SCALE: u16 = 1074;
pub const COUNT_INFO: u16 = 1080;
pub const ANNO: [u8; 4] = *b"Anno";

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|b| u16::from_be_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }
    fn pascal(&mut self) -> Option<String> {
        let n = usize::from(self.u8()?);
        let s = self.take(n)?;
        if (n + 1) % 2 == 1 {
            self.take(1)?;
        }
        Some(s.iter().map(|&c| c as char).collect())
    }
}

fn pascal(out: &mut Vec<u8>, s: &str) {
    let bytes: Vec<u8> = s.chars().map(|c| if (c as u32) < 256 { c as u32 as u8 } else { b'?' }).take(255).collect();
    out.push(bytes.len() as u8);
    out.extend_from_slice(&bytes);
    if (bytes.len() + 1) % 2 == 1 {
        out.push(0);
    }
}

/// PSD colour struct (space + 4 × u16) → RGB.
fn color_from(space: u16, c: [u16; 4]) -> Color {
    let f = |v: u16| f32::from(v) / 65535.0;
    match space {
        // HSB: hue 0–65535 maps to 0–360°.
        1 => {
            let (h, s, v) = (f(c[0]) * 6.0, f(c[1]), f(c[2]));
            let i = (h.floor() as i32).rem_euclid(6);
            let fr = h - h.floor();
            let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * fr), v * (1.0 - s * (1.0 - fr)));
            let (r, g, b) = match i {
                0 => (v, t, p),
                1 => (q, v, p),
                2 => (p, v, t),
                3 => (p, q, v),
                4 => (t, p, v),
                _ => (v, p, q),
            };
            Color::rgb(r, g, b)
        }
        // CMYK (inverted: 0 = 100 % ink), naive.
        2 => {
            let k = 1.0 - f(c[3]);
            Color::rgb(f(c[0]) * (1.0 - k), f(c[1]) * (1.0 - k), f(c[2]) * (1.0 - k))
        }
        // Grayscale 0–10000.
        8 => {
            let g = 1.0 - (f32::from(c[0]) / 10000.0).clamp(0.0, 1.0);
            Color::rgb(g, g, g)
        }
        _ => Color::rgb(f(c[0]), f(c[1]), f(c[2])),
    }
}

fn decode_text(data: &[u8]) -> String {
    let s = if data.len() >= 2 && data[0] == 0xFE && data[1] == 0xFF {
        let units: Vec<u16> = data[2..].as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        String::from_utf16_lossy(&units)
    } else {
        data.iter().map(|&c| c as char).collect()
    };
    s.trim_end_matches('\0').replace("\r\n", "\n").replace('\r', "\n")
}

fn encode_text(s: &str) -> Vec<u8> {
    let mut out = vec![0xFE, 0xFF];
    for u in s.replace("\r\n", "\n").replace('\n', "\r").encode_utf16() {
        out.extend_from_slice(&u.to_be_bytes());
    }
    out
}

/// One parsed `Anno` entry: a note, or a raw entry we keep (sound annotations).
enum Entry {
    Text(Note),
    Raw(Vec<u8>),
}

fn parse_entries(data: &[u8]) -> Option<Vec<Entry>> {
    let mut r = Reader { d: data, p: 0 };
    let (_major, _minor) = (r.u16()?, r.u16()?);
    let count = r.u32()? as usize;
    let mut out = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let start = r.p;
        let len = r.u32()? as usize;
        // The length counts itself and the type: anything shorter would put the cursor back on
        // this entry, and a hostile count would then loop until memory runs out.
        if len < 8 {
            return None;
        }
        let end = start.checked_add(len)?;
        let raw = data.get(start..end)?;
        let kind = r.take(4)?;
        if kind != b"txtA" {
            out.push(Entry::Raw(raw.to_vec()));
            r.p = end;
            continue;
        }
        let open = r.u8()? != 0;
        let (_flags, _opt) = (r.u8()?, r.u16()?);
        let icon = [r.i32()?, r.i32()?, r.i32()?, r.i32()?];
        let popup = [r.i32()?, r.i32()?, r.i32()?, r.i32()?];
        let space = r.u16()?;
        let c = [r.u16()?, r.u16()?, r.u16()?, r.u16()?];
        let author = r.pascal()?;
        let _name = r.pascal()?;
        let modified = r.pascal()?;
        let _data_len = r.u32()?;
        let _key = r.take(4)?;
        let n = r.u32()? as usize;
        let text = decode_text(r.take(n)?);
        out.push(Entry::Text(Note {
            author,
            color: color_from(space, c),
            text,
            position: [f64::from(icon[1]), f64::from(icon[0])],
            popup: [f64::from(popup[1]), f64::from(popup[0]), f64::from(popup[3]), f64::from(popup[2])],
            open,
            modified,
        }));
        r.p = end;
    }
    Some(out)
}

/// Notes in an `Anno` block (None when it doesn't parse).
pub fn parse_anno(data: &[u8]) -> Option<Vec<Note>> {
    Some(parse_entries(data)?.into_iter().filter_map(|e| if let Entry::Text(n) = e { Some(n) } else { None }).collect())
}

fn write_note(n: &Note) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"txtA");
    b.push(u8::from(n.open));
    b.push(0x1C);
    b.extend_from_slice(&1u16.to_be_bytes());
    // The position comes from the file or a command and can sit anywhere in i32 (issue #939):
    // the cast saturates, and the icon's far edge must saturate too rather than wrap (release) or
    // overflow (debug) when the note is re-saved.
    let (x, y) = (n.position[0].round() as i32, n.position[1].round() as i32);
    for v in [y, x, y.saturating_add(20), x.saturating_add(16)] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    let pp = if n.popup == [0.0; 4] { [n.position[0] + 20.0, n.position[1] + 20.0, n.position[0] + 260.0, n.position[1] + 160.0] } else { n.popup };
    for v in [pp[1], pp[0], pp[3], pp[2]] {
        b.extend_from_slice(&(v.round() as i32).to_be_bytes());
    }
    b.extend_from_slice(&0u16.to_be_bytes());
    let rgb = n.color.to_rgb();
    for v in rgb {
        b.extend_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes());
    }
    b.extend_from_slice(&0u16.to_be_bytes());
    pascal(&mut b, &n.author);
    pascal(&mut b, "");
    pascal(&mut b, &n.modified);
    let text = encode_text(&n.text);
    b.extend_from_slice(&((text.len() + 12) as u32).to_be_bytes());
    b.extend_from_slice(b"txtC");
    b.extend_from_slice(&(text.len() as u32).to_be_bytes());
    b.extend_from_slice(&text);
    let mut out = ((b.len() + 4) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(&b);
    out
}

/// An `Anno` block holding `notes` (plus raw sound entries carried over).
pub fn write_anno(notes: &[Note], raw_extra: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&2u16.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&((notes.len() + raw_extra.len()) as u32).to_be_bytes());
    for n in notes {
        out.extend_from_slice(&write_note(n));
    }
    for r in raw_extra {
        out.extend_from_slice(r);
    }
    out
}

/// The document's notes from its raw global blocks (empty when there is no `Anno` block).
pub fn notes_from_blocks(doc: &Document) -> Vec<Note> {
    doc.metadata.psd_global_blocks.iter().find(|(_, k, _)| *k == ANNO).and_then(|(_, _, d)| parse_anno(d)).unwrap_or_default()
}

/// Whether the document holds an `Anno` block that doesn't parse (its notes can't be shown).
pub fn anno_unreadable(doc: &Document) -> bool {
    doc.metadata.psd_global_blocks.iter().find(|(_, k, _)| *k == ANNO).is_some_and(|(_, _, d)| parse_entries(d).is_none())
}

/// Global blocks to write: the raw `Anno` block while it still describes `doc.notes` (or doesn't
/// parse and no note was added), else a regenerated one in its place (appended when new, dropped
/// when the last note is deleted).
pub fn export_blocks(doc: &Document, mut blocks: Vec<photocraft_doc::PsdGlobalBlock>) -> Vec<photocraft_doc::PsdGlobalBlock> {
    let at = blocks.iter().position(|(_, k, _)| *k == ANNO);
    let raw = at.map(|i| blocks[i].2.clone());
    let entries = raw.as_deref().and_then(|d| parse_entries(d));
    let parsed: Option<Vec<Note>> = entries.as_ref().map(|e| e.iter().filter_map(|e| if let Entry::Text(n) = e { Some(n.clone()) } else { None }).collect());
    // Compare in stored precision (colours are 16-bit, positions whole pixels, empty popups get a
    // default rectangle).
    let want = parse_anno(&write_anno(&doc.notes, &[]));
    let same = |a: &[Note], b: &[Note]| {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| {
                let (cx, cy) = (x.color.to_rgb(), y.color.to_rgb());
                Note { color: y.color, ..x.clone() } == *y && cx.iter().zip(cy).all(|(p, q)| (p - q).abs() < 1.0 / 512.0)
            })
    };
    // With no notes, a block we can't read is kept as is: it isn't "the last note deleted", and
    // dropping it would delete the annotations it holds (issue #944).
    if parsed.as_deref().zip(want.as_deref()).is_some_and(|(p, w)| same(p, w)) || (entries.is_none() && doc.notes.is_empty()) {
        return blocks;
    }
    let extra: Vec<Vec<u8>> = entries.unwrap_or_default().into_iter().filter_map(|e| if let Entry::Raw(r) = e { Some(r) } else { None }).collect();
    let fresh = (!doc.notes.is_empty() || !extra.is_empty()).then(|| Arc::new(write_anno(&doc.notes, &extra)));
    match (at, fresh) {
        (Some(i), Some(f)) => blocks[i].2 = f,
        (Some(i), None) => {
            blocks.remove(i);
        }
        (None, Some(f)) => blocks.push((*b"8BIM", ANNO, f)),
        (None, None) => {}
    }
    blocks
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Double(d) => Some(*d),
        Value::Integer(i) => Some(f64::from(*i)),
        Value::LargeInteger(i) => Some(*i as f64),
        Value::UnitFloat { value, .. } => Some(*value),
        _ => None,
    }
}

fn find<'a>(d: &'a Descriptor, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| d.get(k)).or_else(|| {
        // The scale may be nested one object deep (`measurementScale` object).
        d.items.iter().find_map(|(_, v)| if let Value::Descriptor(o) = v { keys.iter().find_map(|k| o.get(k)) } else { None })
    })
}

/// Measurement scale described by resource 1074 (None when it doesn't parse).
pub fn scale_from_resource(data: &[u8]) -> Option<MeasurementScale> {
    let d = VersionedDescriptor::parse_prefix(data).ok()?.0.descriptor;
    let pixel_length = find(&d, &["pixelLength", "PxlL"]).and_then(num)?;
    let logical_length = find(&d, &["logicalLength", "LglL"]).and_then(num)?;
    let units = match find(&d, &["logicalUnits", "LglU"]) {
        Some(Value::Text(t)) => t.to_string_lossy(),
        _ => "pixels".into(),
    };
    Some(MeasurementScale { pixel_length, logical_length, units })
}

/// Resource 1074 data for a scale.
pub fn write_scale_resource(s: &MeasurementScale) -> Vec<u8> {
    let d = Descriptor::new("null")
        .with("pixelLength", Value::Double(s.pixel_length))
        .with("logicalLength", Value::Double(s.logical_length))
        .with("logicalUnits", Value::Text(photocraft_psd::descriptor::UnicodeString::new(&s.units)));
    let mut v = VersionedDescriptor::new(d).to_bytes();
    if v.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// The measurement scale a document's preserved resource 1074 holds (the default when absent).
pub fn raw_scale(doc: &Document) -> Option<MeasurementScale> {
    let raw = doc.metadata.psd_resources.iter().find(|(id, _, _)| *id == MEASUREMENT_SCALE)?;
    scale_from_resource(&raw.2)
}

/// Whether the preserved resource 1074 may be written as is: it still describes the document's
/// scale (or doesn't parse and the scale was never changed from the default).
pub fn keep_raw_scale(doc: &Document) -> bool {
    match raw_scale(doc) {
        Some(s) => s == doc.measurement.scale,
        None => doc.measurement.scale.is_default(),
    }
}

/// Resource 1074 to write when the preserved one can't be used (None = write nothing).
pub fn fresh_scale_resource(doc: &Document) -> Option<Vec<u8>> {
    let has_raw = doc.metadata.psd_resources.iter().any(|(id, _, _)| *id == MEASUREMENT_SCALE);
    if has_raw && keep_raw_scale(doc) {
        return None;
    }
    (!doc.measurement.scale.is_default()).then(|| write_scale_resource(&doc.measurement.scale))
}

/// Whether to keep a preserved resource with this id on export.
pub fn keep_resource(doc: &Document, id: u16) -> bool {
    match id {
        MEASUREMENT_SCALE => keep_raw_scale(doc),
        // Our count markers can't be written into Photoshop's (unpublished) layout; a stale copy
        // would contradict them, so it is dropped once the document has markers of its own.
        COUNT_INFO => doc.measurement.count_total() == 0,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "corpus")]
    const SAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../corpus/psd/ag-psd/read-write/annotations/src.psd");

    #[cfg(feature = "corpus")]
    fn anno_of(bytes: &[u8]) -> Option<Vec<u8>> {
        let i = bytes.windows(8).position(|w| w == b"8BIMAnno")?;
        let n = u32::from_be_bytes(bytes[i + 8..i + 12].try_into().ok()?) as usize;
        Some(bytes[i + 12..i + 12 + n].to_vec())
    }

    #[cfg(feature = "corpus")]
    #[test]
    fn parses_and_rewrites_the_sample_byte_exact() {
        let file = std::fs::read(SAMPLE).unwrap_or_else(|e| panic!("{SAMPLE}: {e}: run `cargo xtask corpus --all`"));
        let raw = anno_of(&file).unwrap();
        let notes = parse_anno(&raw).unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].author, "someone");
        assert_eq!(notes[0].text, "Example annotation\nis here\n\n~ annotator\n");
        assert_eq!(notes[0].position, [42.0, 28.0]);
        assert_eq!(notes[0].popup, [92.0, 74.0, 332.0, 214.0]);
        assert_eq!(notes[0].modified, "D:20210521231029+01'00'");
        assert!(notes[0].open);
        // Second note's colour is stored as HSB.
        assert_eq!(notes[1].author, "some author");
        assert_eq!(notes[1].text, "open note");
        let rgb = notes[1].color.to_rgb();
        // Hue 343°, saturation 0.49, brightness 1: pink.
        assert!(rgb[0] > 0.99 && rgb[1] < rgb[2] && rgb[2] < 0.7, "{rgb:?}");
        // Text, positions, flags and RGB colours are rewritten exactly.
        let again = write_anno(&notes[..1], &[]);
        assert_eq!(parse_anno(&again).unwrap()[..], notes[..1]);
        let n0 = 8 + u32::from_be_bytes(raw[8..12].try_into().unwrap()) as usize;
        assert_eq!(&again[8..], &raw[8..n0]);
    }

    #[test]
    fn export_blocks_keeps_replaces_and_drops() {
        let mut d = Document::new("n", photocraft_doc::Size::new(10, 10), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        assert!(export_blocks(&d, vec![]).is_empty());
        d.notes.push(Note { text: "héllo ✓".into(), author: "me".into(), position: [3.0, 4.0], ..Default::default() });
        let out = export_blocks(&d, vec![]);
        assert_eq!(out.len(), 1);
        assert_eq!(parse_anno(&out[0].2).unwrap()[0].text, "héllo ✓");
        // Unchanged → the very same bytes.
        d.metadata.psd_global_blocks = out.clone();
        assert!(Arc::ptr_eq(&export_blocks(&d, out.clone())[0].2, &out[0].2));
        d.notes.clear();
        assert!(export_blocks(&d, out).is_empty());
        assert!(parse_anno(&[0, 2, 0, 1, 0, 0, 0, 9]).is_none());
    }

    /// Issue #944: an `Anno` block that doesn't parse was taken for "no notes" and dropped on save.
    #[test]
    fn export_blocks_keeps_an_unreadable_block() {
        let mut d = Document::new("n", photocraft_doc::Size::new(10, 10), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        let bad: Vec<photocraft_doc::PsdGlobalBlock> = vec![(*b"8BIM", ANNO, Arc::new(vec![0, 2, 0, 1, 0, 0, 0, 9]))];
        d.metadata.psd_global_blocks = bad.clone();
        assert!(anno_unreadable(&d) && notes_from_blocks(&d).is_empty());
        let out = export_blocks(&d, bad.clone());
        assert_eq!(out.len(), 1);
        assert!(Arc::ptr_eq(&out[0].2, &bad[0].2));
        // A note added by the user still writes a fresh block in its place.
        d.notes.push(Note { text: "new".into(), ..Default::default() });
        let out = export_blocks(&d, bad);
        assert_eq!(parse_anno(&out[0].2).unwrap()[0].text, "new");
        // A readable block is not reported.
        d.metadata.psd_global_blocks = out;
        assert!(!anno_unreadable(&d));
    }

    /// An entry length below 8 (its own `u32` plus the type) put the cursor back on the same
    /// entry, so a count of 0xFFFFFFFF looped ~4.3e9 times and ran out of memory.
    #[test]
    fn short_entry_length_is_unreadable_not_a_loop() {
        for len in [0u8, 4, 7] {
            for kind in [b"sndA", b"txtA"] {
                let mut b = vec![0, 2, 0, 1, 0, 0, 0, 3, 0, 0, 0, len];
                b.extend_from_slice(kind);
                b.extend_from_slice(&[0; 64]);
                assert!(parse_entries(&b).is_none(), "len {len} {kind:?}");
            }
        }
        let hostile = [0, 2, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, b's', b'n', b'd', b'A'];
        assert!(parse_anno(&hostile).is_none());
    }

    #[test]
    fn scale_resource_roundtrip_and_keep_rules() {
        let s = MeasurementScale { pixel_length: 150.0, logical_length: 2.0, units: "mm".into() };
        assert_eq!(scale_from_resource(&write_scale_resource(&s)), Some(s.clone()));
        let mut d = Document::new("n", photocraft_doc::Size::new(10, 10), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        assert!(fresh_scale_resource(&d).is_none());
        // An unparseable resource stays while the scale is untouched.
        d.metadata.psd_resources.push((MEASUREMENT_SCALE, String::new(), Arc::new(vec![0, 0, 0, 16, 1])));
        assert!(keep_resource(&d, MEASUREMENT_SCALE) && fresh_scale_resource(&d).is_none());
        d.measurement.scale = s.clone();
        assert!(!keep_resource(&d, MEASUREMENT_SCALE));
        assert_eq!(scale_from_resource(&fresh_scale_resource(&d).unwrap()), Some(s));
        assert!(keep_resource(&d, COUNT_INFO));
        d.measurement.count_groups.push(photocraft_doc::CountGroup { points: vec![[1.0, 1.0]], ..Default::default() });
        assert!(!keep_resource(&d, COUNT_INFO));
    }
}
