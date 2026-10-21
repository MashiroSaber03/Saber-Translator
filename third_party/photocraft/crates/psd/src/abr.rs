//! Photoshop brush files (`.abr`): the legacy v1/v2 layout and the sectioned v6+ layout.
//!
//! Implemented clean-room from public descriptions of the format (the layout notes published with
//! open-source readers such as GIMP's and ag-psd's, and observation of files written by Photoshop).
//! No code was copied. All numbers are big-endian.
//!
//! ```text
//! v1 / v2   := version u16 (1|2), count u16, count × brush
//! brush     := type u16 (1 computed, 2 sampled), length u32, body[length]
//! computed  := misc u32, spacing u16 (%), diameter u16, roundness u16 (%), angle i16, hardness u16 (%)
//! sampled   := misc u32, spacing u16 (%), [v2: name (u32 length + UTF-16 incl. NUL)],
//!              anti-alias u8, short bounds 4 × i16, bounds 4 × i32 (top, left, bottom, right),
//!              depth u16 (8|16), compression u8 (0 raw, 1 PackBits with a u16 row-length table)
//!
//! v6+       := version u16 (6..=10), subversion u16 (1|2), sections…
//! section   := "8BIM", key [4], length u32, data[length] (padded to 4 bytes)
//! "samp"    := entries: length u32 (padded to 4), Pascal uuid, skip (10 if subversion 1,
//!              264 if 2), bounds 4 × i32, depth i16, compression u8, data
//! "patt"    := patterns as in the `Patt` global block
//! "desc"    := version u32 (16) + ActionDescriptor; key `Brsh` lists `brushPreset` objects
//! "phry"    := version u32 (16) + ActionDescriptor; key `hierarchy` lists objects read as a
//!              token stream: `Grup` { `Nm  ` name, `zuid` uuid } opens a folder, `preset` {} is
//!              the next preset of `desc` (in order), `groupEnd` {} closes the current folder
//! ```
//!
//! The `phry` (preset hierarchy) section keeps the Brushes panel folders of Photoshop CC (2018+)
//! files: [`AbrFile::folders`] gives each preset's folder path. Folders nest (a `Grup` inside a
//! `Grup`); presets outside any folder are at the top level. A damaged hierarchy never fails the
//! parse: mismatches become warnings and the presets they affect land at the top level.
//!
//! [`parse`] never panics: every length is bounds-checked, sizes are capped ([`MAX_EDGE`],
//! [`MAX_TOTAL_BYTES`]) and a damaged brush is skipped with a warning when the rest of the file
//! is still readable. [`write_v12`] and [`write_v6`] produce files in the same layouts (used by
//! the tests and fuzz seeds; exporting brushes reuses them).

use crate::compression::{Compression, PlaneLayout, decode_planes, encode_planes};
use crate::descriptor::{Descriptor, Value, VersionedDescriptor};
use crate::error::{PsdError, Result};
use crate::header::Version;
use crate::patterns::{PsdPattern, parse_pattern_block, write_pattern_block};

/// Largest sampled tip edge accepted (Photoshop's own limit is 5000 px).
pub const MAX_EDGE: u32 = 8192;
/// Most brushes read from one file.
pub const MAX_BRUSHES: usize = 20_000;
/// Cap on decoded sample bytes over the whole file.
pub const MAX_TOTAL_BYTES: usize = 512 << 20;

/// A sampled tip: one grayscale plane where the maximum value is full paint.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrSample {
    /// The uuid that `sampledData` in a preset descriptor refers to (v6+), or empty.
    pub id: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Bits per sample: 8 or 16.
    pub depth: u16,
    /// Decoded samples, row-major (`width × height × depth / 8` bytes, 16-bit big-endian).
    pub data: Vec<u8>,
}

impl AbrSample {
    /// Samples as 0..1 paint amounts.
    pub fn to_unit(&self) -> Vec<f32> {
        if self.depth == 16 {
            self.data.as_chunks::<2>().0.iter().map(|c| f32::from(u16::from_be_bytes(*c)) / 65535.0).collect()
        } else {
            self.data.iter().map(|&v| f32::from(v) / 255.0).collect()
        }
    }
}

/// A legacy (v1/v2) tip.
#[derive(Debug, Clone, PartialEq)]
pub enum LegacyTip {
    /// Computed elliptical tip.
    Computed {
        /// Diameter in pixels.
        diameter: u16,
        /// 0..100 %.
        hardness: u16,
        /// Degrees.
        angle: i16,
        /// 0..100 %.
        roundness: u16,
    },
    /// Sampled tip.
    Sampled(AbrSample),
}

/// A legacy (v1/v2) brush.
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyBrush {
    /// Name (v2 sampled brushes only; empty otherwise).
    pub name: String,
    /// Spacing in percent of the diameter (0 = spacing off).
    pub spacing: u16,
    /// Anti-aliasing flag (sampled tips).
    pub anti_alias: bool,
    /// The tip.
    pub tip: LegacyTip,
}

/// A parsed brush file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AbrFile {
    /// File version (1, 2, 6, 7, 9 or 10).
    pub version: u16,
    /// v6+ subversion (1 or 2); 0 for v1/v2.
    pub subversion: u16,
    /// v1/v2 brushes, in file order.
    pub legacy: Vec<LegacyBrush>,
    /// v6+ sampled tips (`samp`), in file order.
    pub samples: Vec<AbrSample>,
    /// v6+ embedded patterns (`patt`), referenced by texture settings.
    pub patterns: Vec<PsdPattern>,
    /// v6+ brush presets: the `brushPreset` objects of the `desc` section, in file order.
    pub presets: Vec<Descriptor>,
    /// v6+ folder of each preset, parallel to [`AbrFile::presets`]: the folder names from the
    /// outermost in, empty for a preset at the top level. Empty when the file has no readable
    /// preset hierarchy (`phry`), i.e. every preset is at the top level.
    pub folders: Vec<Vec<String>>,
    /// Problems that did not stop the parse (skipped brushes, unknown sections…).
    pub warnings: Vec<String>,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Rd { b, p: 0 }
    }
    fn left(&self) -> usize {
        self.b.len().saturating_sub(self.p)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or(PsdError::LimitExceeded("abr length overflow"))?;
        let s = self.b.get(self.p..end).ok_or(PsdError::UnexpectedEof { offset: self.p, needed: end.saturating_sub(self.b.len()) })?;
        self.p = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        let s = self.take(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.arr::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.arr()?))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.arr()?))
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }
}

/// Bytes budget shared by all samples of one file.
struct Budget(usize);

impl Budget {
    fn spend(&mut self, n: usize) -> Result<()> {
        self.0 = self.0.checked_sub(n).ok_or(PsdError::LimitExceeded("abr samples exceed MAX_TOTAL_BYTES"))?;
        Ok(())
    }
}

/// Reads bounds (top, left, bottom, right), depth and compression, then the pixel data.
fn read_sample_body(r: &mut Rd, id: String, budget: &mut Budget) -> Result<AbrSample> {
    let (top, left, bottom, right) = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
    let w = i64::from(right) - i64::from(left);
    let h = i64::from(bottom) - i64::from(top);
    if w <= 0 || h <= 0 {
        return Err(PsdError::invalid(format!("brush sample bounds {w}×{h}")));
    }
    if w > i64::from(MAX_EDGE) || h > i64::from(MAX_EDGE) {
        return Err(PsdError::LimitExceeded("brush sample larger than MAX_EDGE"));
    }
    let (w, h) = (w as u32, h as u32);
    let depth = r.u16()?;
    if depth != 8 && depth != 16 {
        return Err(PsdError::invalid(format!("brush sample depth {depth}")));
    }
    let comp = r.u8()?;
    let compression = match comp {
        0 => Compression::Raw,
        1 => Compression::Rle,
        c => return Err(PsdError::Unsupported(format!("brush sample compression {c}"))),
    };
    let layout = PlaneLayout { planes: 1, width: w as usize, height: h as usize, depth, version: Version::Psd };
    budget.spend(layout.decoded_len()?)?;
    let rest = r.take(r.left())?;
    let data = decode_planes(compression, rest, &layout)?;
    Ok(AbrSample { id, width: w, height: h, depth, data })
}

fn read_ucs2(r: &mut Rd) -> Result<String> {
    let n = r.u32()? as usize;
    if n > 4096 {
        return Err(PsdError::LimitExceeded("abr name length"));
    }
    let units: Vec<u16> = r.take(n * 2)?.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes(*c)).collect();
    Ok(String::from_utf16_lossy(&units).trim_end_matches('\0').to_string())
}

fn parse_v12(r: &mut Rd, version: u16, out: &mut AbrFile) -> Result<()> {
    let count = r.u16()?;
    let mut budget = Budget(MAX_TOTAL_BYTES);
    for i in 0..count {
        if r.left() == 0 {
            out.warnings.push(format!("file ends after {i} of {count} brushes"));
            break;
        }
        let ty = r.u16()?;
        let len = r.u32()? as usize;
        let body = match r.take(len) {
            Ok(b) => b,
            Err(_) => {
                out.warnings.push(format!("brush {}: truncated", i + 1));
                break;
            }
        };
        let mut b = Rd::new(body);
        let parsed = (|| -> Result<Option<LegacyBrush>> {
            match ty {
                1 => {
                    let _misc = b.u32()?;
                    let spacing = b.u16()?;
                    let diameter = b.u16()?;
                    let roundness = b.u16()?;
                    let angle = b.u16()? as i16;
                    let hardness = b.u16()?;
                    Ok(Some(LegacyBrush { name: String::new(), spacing, anti_alias: true, tip: LegacyTip::Computed { diameter, hardness, angle, roundness } }))
                }
                2 => {
                    let _misc = b.u32()?;
                    let spacing = b.u16()?;
                    let name = if version == 2 { read_ucs2(&mut b)? } else { String::new() };
                    let anti_alias = b.u8()? != 0;
                    b.skip(8)?; // short bounds (superseded by the 32-bit bounds)
                    let s = read_sample_body(&mut b, String::new(), &mut budget)?;
                    Ok(Some(LegacyBrush { name, spacing, anti_alias, tip: LegacyTip::Sampled(s) }))
                }
                other => {
                    out.warnings.push(format!("brush {}: unknown brush type {other} skipped", i + 1));
                    Ok(None)
                }
            }
        })();
        match parsed {
            Ok(Some(lb)) => out.legacy.push(lb),
            Ok(None) => {}
            Err(PsdError::LimitExceeded(m)) if m.contains("MAX_TOTAL_BYTES") => return Err(PsdError::LimitExceeded(m)),
            Err(e) => out.warnings.push(format!("brush {}: {e}; skipped", i + 1)),
        }
    }
    Ok(())
}

fn read_samp(data: &[u8], subversion: u16, out: &mut AbrFile, budget: &mut Budget) -> Result<()> {
    let mut r = Rd::new(data);
    while r.left() >= 4 {
        if out.samples.len() >= MAX_BRUSHES {
            out.warnings.push("too many sampled tips; the rest were skipped".into());
            break;
        }
        let len = r.u32()? as usize;
        if len == 0 {
            break;
        }
        let padded = len.checked_add(3).ok_or(PsdError::LimitExceeded("abr sample length"))? & !3;
        let body = match r.take(padded.min(r.left())) {
            Ok(b) if b.len() >= len => b,
            _ => {
                out.warnings.push(format!("sampled tip {}: truncated", out.samples.len() + 1));
                break;
            }
        };
        let mut b = Rd::new(body);
        let parsed = (|| -> Result<AbrSample> {
            let n = b.u8()? as usize;
            let id = String::from_utf8_lossy(b.take(n)?).to_string();
            b.skip(if subversion == 1 { 10 } else { 264 })?;
            read_sample_body(&mut b, id, budget)
        })();
        match parsed {
            Ok(s) => out.samples.push(s),
            Err(PsdError::LimitExceeded(m)) if m.contains("MAX_TOTAL_BYTES") => return Err(PsdError::LimitExceeded(m)),
            Err(e) => out.warnings.push(format!("sampled tip {}: {e}; skipped", out.samples.len() + 1)),
        }
    }
    Ok(())
}

fn read_desc(data: &[u8], out: &mut AbrFile) {
    match VersionedDescriptor::parse_prefix(data) {
        Ok((vd, _)) => match vd.descriptor.get("Brsh") {
            Some(Value::List(items)) => {
                for v in items.iter().take(MAX_BRUSHES) {
                    if let Value::Descriptor(d) = v {
                        out.presets.push(d.clone());
                    }
                }
            }
            _ => out.warnings.push("brush settings (desc) hold no brush list".into()),
        },
        Err(e) => out.warnings.push(format!("brush settings (desc) unreadable: {e}")),
    }
}

/// Deepest folder nesting read from or written to a preset hierarchy; deeper folders are merged
/// into their ancestor at this depth.
pub const MAX_FOLDER_DEPTH: usize = 32;
/// Longest folder name kept (characters).
const MAX_FOLDER_NAME: usize = 255;

/// Most bytes of folder names held across all presets' folder paths (each name also counts its
/// `String`). Presets past it are imported at the top level. 20,000 presets three folders deep
/// with 20-character names take about 3 MB.
const MAX_FOLDER_BYTES: usize = 16 << 20;

/// Reads the `phry` token stream into one folder path per preset (`presets` of them, the count
/// read from `desc`), in order. Paths are built only for those presets, within
/// [`MAX_FOLDER_BYTES`]: a small section can't make the reader allocate more than that. `None`
/// (with a warning) when the section is unreadable.
fn read_phry(data: &[u8], presets: usize, warnings: &mut Vec<String>) -> Option<Vec<Vec<String>>> {
    let vd = match VersionedDescriptor::parse_prefix(data) {
        Ok((vd, _)) => vd,
        Err(e) => {
            warnings.push(format!("brush folders (phry) unreadable: {e}; brushes imported without folders"));
            return None;
        }
    };
    let Some(Value::List(items)) = vd.descriptor.get("hierarchy") else {
        warnings.push("brush folders (phry) hold no hierarchy; brushes imported without folders".into());
        return None;
    };
    let mut stack: Vec<String> = Vec::new();
    // Folders opened past MAX_FOLDER_DEPTH: their `groupEnd`s close nothing on `stack`.
    let mut overflow = 0usize;
    let mut out = Vec::with_capacity(presets.min(MAX_BRUSHES));
    // Preset tokens seen (only counted past `presets`), and the bytes held by `out`'s paths.
    let (mut listed, mut bytes) = (0usize, 0usize);
    let (mut unbalanced, mut too_deep, mut too_big) = (false, false, false);
    for v in items {
        let Value::Descriptor(d) = v else { continue };
        if d.class_id.is("Grup") {
            if stack.len() >= MAX_FOLDER_DEPTH {
                overflow = overflow.saturating_add(1);
                too_deep = true;
                continue;
            }
            let name = match d.get("Nm  ") {
                Some(Value::Text(t)) => t.to_string_lossy(),
                _ => String::new(),
            };
            let name: String = name.trim().chars().take(MAX_FOLDER_NAME).collect();
            stack.push(if name.is_empty() { "Group".to_string() } else { name });
        } else if d.class_id.is("groupEnd") {
            if overflow > 0 {
                overflow -= 1;
            } else if stack.pop().is_none() {
                unbalanced = true;
            }
        } else if d.class_id.is("preset") {
            listed = listed.saturating_add(1);
            if out.len() >= presets {
                continue;
            }
            let size = stack.iter().map(|n| n.len().saturating_add(size_of::<String>())).fold(0usize, usize::saturating_add);
            match bytes.checked_add(size).filter(|&b| b <= MAX_FOLDER_BYTES) {
                Some(b) if !too_big => {
                    bytes = b;
                    out.push(stack.clone());
                }
                _ => {
                    too_big = true;
                    out.push(Vec::new());
                }
            }
        }
    }
    if unbalanced {
        warnings.push("brush folders (phry): a folder end without a folder was ignored".into());
    }
    if too_deep {
        warnings.push(format!("brush folders nested deeper than {MAX_FOLDER_DEPTH} levels were merged into their parent"));
    }
    if too_big {
        warnings.push("brush folders (phry) too large; the remaining brushes were imported at the top level".into());
    }
    if listed != presets {
        warnings.push(format!(
            "brush folders (phry) list {listed} brushes but the file holds {presets}; {}",
            if listed > presets { "the extra entries were ignored" } else { "the rest were imported at the top level" }
        ));
        out.resize(presets, Vec::new());
    }
    Some(out)
}

/// Pairs the `phry` section's preset tokens with the presets read from `desc`, in order.
fn apply_hierarchy(out: &mut AbrFile, phry: Option<&[u8]>) {
    let Some(data) = phry else { return };
    let Some(folders) = read_phry(data, out.presets.len(), &mut out.warnings) else { return };
    if folders.iter().any(|f| !f.is_empty()) {
        out.folders = folders;
    }
}

fn parse_v6(r: &mut Rd, out: &mut AbrFile) -> Result<()> {
    out.subversion = r.u16()?;
    if !matches!(out.subversion, 1 | 2) {
        return Err(PsdError::Unsupported(format!("brush file subversion {}", out.subversion)));
    }
    let mut budget = Budget(MAX_TOTAL_BYTES);
    let mut hierarchy = None;
    while r.left() >= 12 {
        let sig = r.arr::<4>()?;
        if &sig != b"8BIM" {
            out.warnings.push(format!("unexpected data at offset {} ignored", r.p - 4));
            break;
        }
        let key = r.arr::<4>()?;
        let len = r.u32()? as usize;
        let data = match r.take(len) {
            Ok(d) => d,
            Err(_) => {
                out.warnings.push(format!("section {} truncated", String::from_utf8_lossy(&key)));
                r.take(r.left())?
            }
        };
        // Sections are padded to 4 bytes; tolerate writers that don't pad.
        let pad = (4 - len % 4) % 4;
        if pad > 0 && r.b.get(r.p..r.p + 4) != Some(b"8BIM") && r.b.get(r.p + pad..r.p + pad + 4) == Some(b"8BIM") {
            r.p += pad;
        }
        match &key {
            b"samp" => read_samp(data, out.subversion, out, &mut budget)?,
            b"patt" => match parse_pattern_block(data) {
                Ok(p) => out.patterns.extend(p),
                Err(e) => out.warnings.push(format!("embedded patterns unreadable: {e}")),
            },
            b"desc" => read_desc(data, out),
            // Read once `desc` gives the preset count; `phry` may come first.
            b"phry" => hierarchy = Some(data),
            // Tool presets: not needed for the brushes.
            b"lPdc" => {}
            k => out.warnings.push(format!("section {} ignored", String::from_utf8_lossy(k))),
        }
    }
    apply_hierarchy(out, hierarchy);
    Ok(())
}

/// Parses an `.abr` file.
pub fn parse(data: &[u8]) -> Result<AbrFile> {
    let mut r = Rd::new(data);
    let version = r.u16()?;
    let mut out = AbrFile { version, ..Default::default() };
    match version {
        1 | 2 => parse_v12(&mut r, version, &mut out)?,
        6..=10 => parse_v6(&mut r, &mut out)?,
        v => return Err(PsdError::Unsupported(format!("brush file version {v}"))),
    }
    if out.legacy.is_empty() && out.samples.is_empty() && out.presets.is_empty() {
        let why = out.warnings.first().map(|w| format!(" ({w})")).unwrap_or_default();
        return Err(PsdError::invalid(format!("the brush file holds no readable brushes{why}")));
    }
    Ok(out)
}

// ------------------------------------------------------------------ writing

fn put_u16(o: &mut Vec<u8>, v: u16) {
    o.extend_from_slice(&v.to_be_bytes());
}
fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_be_bytes());
}

fn write_sample_body(o: &mut Vec<u8>, s: &AbrSample, rle: bool) -> Result<()> {
    if s.width == 0 || s.height == 0 || s.width > MAX_EDGE || s.height > MAX_EDGE || !matches!(s.depth, 8 | 16) {
        return Err(PsdError::LimitExceeded("brush sample size or depth"));
    }
    for v in [0, 0, s.height as i32, s.width as i32] {
        o.extend_from_slice(&v.to_be_bytes());
    }
    put_u16(o, s.depth);
    o.push(u8::from(rle));
    let layout = PlaneLayout { planes: 1, width: s.width as usize, height: s.height as usize, depth: s.depth, version: Version::Psd };
    o.extend_from_slice(&encode_planes(if rle { Compression::Rle } else { Compression::Raw }, &s.data, &layout)?);
    Ok(())
}

/// Writes a v1 or v2 file. Sampled tips are PackBits-compressed when `rle`.
pub fn write_v12(version: u16, brushes: &[LegacyBrush], rle: bool) -> Result<Vec<u8>> {
    if !matches!(version, 1 | 2) || brushes.len() > usize::from(u16::MAX) {
        return Err(PsdError::Unsupported(format!("writing brush file version {version}")));
    }
    let mut o = Vec::new();
    put_u16(&mut o, version);
    put_u16(&mut o, brushes.len() as u16);
    for b in brushes {
        let mut body = Vec::new();
        put_u32(&mut body, 0);
        put_u16(&mut body, b.spacing);
        let ty = match &b.tip {
            LegacyTip::Computed { diameter, hardness, angle, roundness } => {
                for v in [*diameter, *roundness, *angle as u16, *hardness] {
                    put_u16(&mut body, v);
                }
                1
            }
            LegacyTip::Sampled(s) => {
                if version == 2 {
                    let units: Vec<u16> = b.name.encode_utf16().chain(std::iter::once(0)).collect();
                    put_u32(&mut body, units.len() as u32);
                    for u in units {
                        put_u16(&mut body, u);
                    }
                }
                body.push(u8::from(b.anti_alias));
                for v in [0u16, 0, s.height.min(0xffff) as u16, s.width.min(0xffff) as u16] {
                    put_u16(&mut body, v);
                }
                write_sample_body(&mut body, s, rle)?;
                2
            }
        };
        put_u16(&mut o, ty);
        put_u32(&mut o, body.len() as u32);
        o.extend_from_slice(&body);
    }
    Ok(o)
}

fn section(o: &mut Vec<u8>, key: &[u8; 4], data: &[u8]) {
    o.extend_from_slice(b"8BIM");
    o.extend_from_slice(key);
    let pad = (4 - data.len() % 4) % 4;
    put_u32(o, (data.len() + pad) as u32);
    o.extend_from_slice(data);
    o.extend(std::iter::repeat_n(0u8, pad));
}

/// Writes a v6 (`subversion` 1 or 2) file: sampled tips, embedded patterns and the preset
/// descriptors (each a `brushPreset` object), listed under `Brsh`. Every preset is at the top
/// level ([`write_v6_folders`] keeps folders).
pub fn write_v6(subversion: u16, samples: &[AbrSample], patterns: &[PsdPattern], presets: &[Descriptor], rle: bool) -> Result<Vec<u8>> {
    write_v6_folders(subversion, samples, patterns, presets, &[], rle)
}

/// A stable uuid-shaped folder id (`zuid`) derived from the folder's path and position.
fn folder_uuid(path: &[&str], seq: usize) -> String {
    // FNV-1a over the path, with two seeds, for 128 bits.
    let hash = |seed: u64| {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
        for b in path.iter().flat_map(|s| s.bytes().chain(std::iter::once(0))).chain(seq.to_le_bytes()) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    };
    let (a, b) = (hash(1), hash(2));
    format!("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", a >> 32, (a >> 16) & 0xffff, a & 0xffff, b >> 48, b & 0xffff_ffff_ffff)
}

/// The `phry` descriptor for presets in `folders` (one path per preset, in preset order).
/// Presets of one folder that are not adjacent reopen it (a reader merges them by path).
fn hierarchy_descriptor(folders: &[Vec<String>]) -> Descriptor {
    let text = |s: &str| Value::Text(crate::descriptor::UnicodeString::new_nul(s));
    let mut items = Vec::new();
    let mut open: Vec<&str> = Vec::new();
    let mut seq = 0usize;
    for path in folders {
        let path: Vec<&str> = path.iter().take(MAX_FOLDER_DEPTH).map(String::as_str).collect();
        let common = open.iter().zip(&path).take_while(|(a, b)| a == b).count();
        while open.len() > common {
            open.pop();
            items.push(Value::Descriptor(Descriptor::new("groupEnd")));
        }
        for name in path.iter().skip(common) {
            open.push(name);
            seq += 1;
            items.push(Value::Descriptor(Descriptor::new("Grup").with("Nm  ", text(name)).with("zuid", text(&folder_uuid(&open, seq)))));
        }
        items.push(Value::Descriptor(Descriptor::new("preset")));
    }
    for _ in open {
        items.push(Value::Descriptor(Descriptor::new("groupEnd")));
    }
    Descriptor::new("null").with("hierarchy", Value::List(items))
}

/// [`write_v6`] that also writes the preset hierarchy (`phry`): `folders` holds each preset's
/// folder path (outermost first; empty = top level), parallel to `presets`. With no folders (an
/// empty slice, or only empty paths) no hierarchy is written.
pub fn write_v6_folders(
    subversion: u16,
    samples: &[AbrSample],
    patterns: &[PsdPattern],
    presets: &[Descriptor],
    folders: &[Vec<String>],
    rle: bool,
) -> Result<Vec<u8>> {
    if !folders.is_empty() && folders.len() != presets.len() {
        return Err(PsdError::invalid(format!("{} folder paths for {} brush presets", folders.len(), presets.len())));
    }
    if !matches!(subversion, 1 | 2) {
        return Err(PsdError::Unsupported(format!("brush file subversion {subversion}")));
    }
    let mut o = Vec::new();
    put_u16(&mut o, 6);
    put_u16(&mut o, subversion);
    let mut samp = Vec::new();
    for s in samples {
        let mut e = Vec::new();
        let id = s.id.as_bytes();
        let n = id.len().min(255);
        e.push(n as u8);
        e.extend_from_slice(&id[..n]);
        e.extend(std::iter::repeat_n(0u8, if subversion == 1 { 10 } else { 264 }));
        write_sample_body(&mut e, s, rle)?;
        put_u32(&mut samp, e.len() as u32);
        let pad = (4 - e.len() % 4) % 4;
        samp.extend_from_slice(&e);
        samp.extend(std::iter::repeat_n(0u8, pad));
    }
    section(&mut o, b"samp", &samp);
    if !patterns.is_empty() {
        section(&mut o, b"patt", &write_pattern_block(patterns)?);
    }
    let list = Value::List(presets.iter().cloned().map(Value::Descriptor).collect());
    let desc = VersionedDescriptor::new(Descriptor::new("null").with("Brsh", list));
    section(&mut o, b"desc", &desc.to_bytes());
    if folders.iter().any(|f| !f.is_empty()) {
        section(&mut o, b"phry", &VersionedDescriptor::new(hierarchy_descriptor(folders)).to_bytes());
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::UnicodeString;

    fn sample(id: &str, w: u32, h: u32, depth: u16) -> AbrSample {
        let bpp = usize::from(depth / 8);
        let data = (0..w as usize * h as usize * bpp).map(|i| if (i / bpp) % 3 == 0 { 255 } else { (i % 7) as u8 * 30 }).collect();
        AbrSample { id: id.into(), width: w, height: h, depth, data }
    }

    #[test]
    fn v1_v2_round_trip_raw_and_rle() {
        for version in [1u16, 2] {
            for rle in [false, true] {
                let brushes = vec![
                    LegacyBrush {
                        name: String::new(),
                        spacing: 25,
                        anti_alias: true,
                        tip: LegacyTip::Computed { diameter: 19, hardness: 80, angle: -30, roundness: 50 },
                    },
                    LegacyBrush {
                        name: if version == 2 { "Leaf ✓".into() } else { String::new() },
                        spacing: 40,
                        anti_alias: false,
                        tip: LegacyTip::Sampled(sample("", 13, 9, 8)),
                    },
                    LegacyBrush { name: String::new(), spacing: 10, anti_alias: true, tip: LegacyTip::Sampled(sample("", 4, 6, 16)) },
                ];
                let bytes = write_v12(version, &brushes, rle).unwrap();
                let f = parse(&bytes).unwrap();
                assert_eq!(f.version, version);
                assert_eq!(f.legacy, brushes, "v{version} rle={rle}");
                assert!(f.warnings.is_empty(), "{:?}", f.warnings);
            }
        }
    }

    #[test]
    fn v6_round_trip_both_subversions() {
        let samples = vec![sample("$abc", 20, 11, 8), sample("$def", 5, 5, 16)];
        let preset = Descriptor::new("brushPreset")
            .with("Nm  ", Value::Text(UnicodeString::new_nul("Grass")))
            .with("Brsh", Value::Descriptor(Descriptor::new("sampledBrush").with("sampledData", Value::Text(UnicodeString::new_nul("$abc")))));
        for sub in [1u16, 2] {
            for rle in [false, true] {
                let bytes = write_v6(sub, &samples, &[], std::slice::from_ref(&preset), rle).unwrap();
                let f = parse(&bytes).unwrap();
                assert_eq!((f.version, f.subversion), (6, sub));
                assert_eq!(f.samples, samples);
                assert_eq!(f.presets, vec![preset.clone()]);
                assert!(f.warnings.is_empty(), "{:?}", f.warnings);
            }
        }
        assert_eq!(samples[1].to_unit().len(), 25);
    }

    #[test]
    fn hostile_input_errors_without_panicking() {
        assert!(parse(&[]).is_err());
        assert!(parse(&[0, 3, 0, 0]).is_err());
        assert!(parse(&[0, 6, 0, 9]).is_err());
        // Absurd sample bounds.
        let mut s = sample("", 2, 2, 8);
        s.width = MAX_EDGE + 1;
        assert!(write_v12(1, &[LegacyBrush { name: String::new(), spacing: 1, anti_alias: true, tip: LegacyTip::Sampled(s) }], false).is_err());
        let good = write_v6(2, &[sample("$a", 30, 30, 8)], &[], &[Descriptor::new("brushPreset")], true).unwrap();
        for cut in 0..good.len() {
            let _ = parse(&good[..cut]);
        }
        let v2 = write_v12(2, &[LegacyBrush { name: "x".into(), spacing: 1, anti_alias: true, tip: LegacyTip::Sampled(sample("", 9, 9, 16)) }], true).unwrap();
        for cut in 0..v2.len() {
            let _ = parse(&v2[..cut]);
        }
        // Flip every byte once: never panics.
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0xa5;
            let _ = parse(&b);
        }
        // A huge declared count with no data is an error, not an allocation.
        assert!(parse(&[0, 2, 0xff, 0xff]).is_err());
    }

    fn named(n: &str) -> Descriptor {
        Descriptor::new("brushPreset").with("Nm  ", Value::Text(UnicodeString::new_nul(n)))
    }

    /// Hierarchy tokens: `G(name)` opens, `P` is a preset, `E` closes.
    enum Tok<'a> {
        G(&'a str),
        P,
        E,
    }

    /// A v6 file with `n` presets and a hand-made `phry` token stream.
    fn with_phry(n: usize, toks: &[Tok]) -> Vec<u8> {
        let presets: Vec<Descriptor> = (0..n).map(|i| named(&format!("B{i}"))).collect();
        let mut bytes = write_v6(2, &[], &[], &presets, false).unwrap();
        section(&mut bytes, b"phry", &phry_bytes(toks));
        bytes
    }

    /// A `phry` section body holding `toks`.
    fn phry_bytes(toks: &[Tok]) -> Vec<u8> {
        let items = toks
            .iter()
            .map(|t| {
                Value::Descriptor(match t {
                    Tok::G(name) => Descriptor::new("Grup")
                        .with("Nm  ", Value::Text(UnicodeString::new_nul(name)))
                        .with("zuid", Value::Text(UnicodeString::new_nul("c9ee8fd5-eb7d-924c-9e95-ddc98e40bb6c"))),
                    Tok::P => Descriptor::new("preset"),
                    Tok::E => Descriptor::new("groupEnd"),
                })
            })
            .collect();
        VersionedDescriptor::new(Descriptor::new("null").with("hierarchy", Value::List(items))).to_bytes()
    }

    fn path(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn phry_flat_folders_like_photoshop_25() {
        // The layout of Photoshop 25.1's default brushes: four folders, every preset in one.
        use Tok::*;
        let f = parse(&with_phry(5, &[G("General"), P, P, E, G("Dry Media"), P, E, G("Wet Media"), P, E, G("Special Effects"), P, E])).unwrap();
        assert_eq!(f.folders, vec![path(&["General"]), path(&["General"]), path(&["Dry Media"]), path(&["Wet Media"]), path(&["Special Effects"])]);
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    }

    #[test]
    fn phry_nested_folders_and_top_level_presets() {
        use Tok::*;
        let f = parse(&with_phry(5, &[P, G("Animals"), G("Birds"), P, E, P, G("Fish"), P, E, E, P])).unwrap();
        assert_eq!(f.folders, vec![path(&[]), path(&["Animals", "Birds"]), path(&["Animals"]), path(&["Animals", "Fish"]), path(&[])]);
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
        // No folder at all: no hierarchy.
        let f = parse(&with_phry(2, &[P, P])).unwrap();
        assert!(f.folders.is_empty());
    }

    #[test]
    fn phry_mismatches_become_warnings() {
        use Tok::*;
        // A stray groupEnd is ignored; an unclosed folder still holds its presets.
        let f = parse(&with_phry(2, &[E, G("A"), P, E, E, G("B"), P])).unwrap();
        assert_eq!(f.folders, vec![path(&["A"]), path(&["B"])]);
        assert!(f.warnings.iter().any(|w| w.contains("folder end")), "{:?}", f.warnings);
        // Fewer preset tokens than presets: the rest are at the top level.
        let f = parse(&with_phry(3, &[G("A"), P, E])).unwrap();
        assert_eq!(f.folders, vec![path(&["A"]), path(&[]), path(&[])]);
        assert!(f.warnings.iter().any(|w| w.contains("list 1 brushes but the file holds 3")), "{:?}", f.warnings);
        // More preset tokens than presets: extras ignored.
        let f = parse(&with_phry(1, &[G("A"), P, P, P, E])).unwrap();
        assert_eq!(f.folders, vec![path(&["A"])]);
        assert_eq!(f.presets.len(), 1);
        assert!(!f.warnings.is_empty());
        // A depth bomb: folders deeper than the limit merge into their ancestor at the limit.
        let mut toks: Vec<Tok> = (0..10_000).map(|_| G("deep")).collect();
        toks.push(P);
        toks.extend((0..10_000).map(|_| E));
        toks.extend([G("next"), P, E]);
        let f = parse(&with_phry(2, &toks)).unwrap();
        assert_eq!(f.folders[0].len(), MAX_FOLDER_DEPTH);
        assert_eq!(f.folders[1], path(&["next"]));
        assert!(f.warnings.iter().any(|w| w.contains("deeper")), "{:?}", f.warnings);
        // An unreadable hierarchy: imported flat, with a warning.
        let mut bytes = write_v6(2, &[], &[], &[named("x")], false).unwrap();
        section(&mut bytes, b"phry", &[0, 0, 0, 16, 0xff, 0xff]);
        let f = parse(&bytes).unwrap();
        assert!(f.folders.is_empty() && f.presets.len() == 1);
        assert!(f.warnings.iter().any(|w| w.contains("phry")), "{:?}", f.warnings);
        // Truncations and bit flips of a file with a hierarchy never panic.
        let good = with_phry(3, &[G("A"), P, G("B"), P, E, E, P]);
        for cut in 0..good.len() {
            let _ = parse(&good[..cut]);
        }
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0x5a;
            let _ = parse(&b);
        }
    }

    #[test]
    fn phry_builds_paths_only_for_the_files_presets() {
        use Tok::*;
        // A few-KB section listing 100,000 deep presets for a file with 2: two paths are built.
        let mut toks: Vec<Tok> = (0..MAX_FOLDER_DEPTH).map(|_| G("folder")).collect();
        toks.extend((0..100_000).map(|_| P));
        let mut warnings = Vec::new();
        let folders = read_phry(&phry_bytes(&toks), 2, &mut warnings).unwrap();
        assert_eq!(folders.len(), 2);
        assert_eq!(folders.capacity(), 2);
        assert_eq!(folders[1].len(), MAX_FOLDER_DEPTH);
        assert!(warnings.iter().any(|w| w.contains("list 100000 brushes but the file holds 2")), "{warnings:?}");
        // `phry` before `desc` still pairs with the file's presets.
        let mut bytes = write_v6(2, &[], &[], &[named("a"), named("b")], false).unwrap();
        let at = bytes.windows(8).position(|w| w == b"8BIMdesc").unwrap();
        let mut phry = Vec::new();
        section(&mut phry, b"phry", &phry_bytes(&[G("A"), P, E, P]));
        bytes.splice(at..at, phry);
        let f = parse(&bytes).unwrap();
        assert_eq!(f.folders, vec![path(&["A"]), path(&[])]);
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    }

    #[test]
    fn phry_folder_paths_stay_within_their_byte_budget() {
        use Tok::*;
        // Thousands of presets under 32 folders of 255-character names would need ~27 MB of paths.
        let long = "x".repeat(MAX_FOLDER_NAME);
        let mut toks: Vec<Tok> = (0..MAX_FOLDER_DEPTH).map(|_| G(&long)).collect();
        toks.extend((0..3_000).map(|_| P));
        let mut warnings = Vec::new();
        let folders = read_phry(&phry_bytes(&toks), 3_000, &mut warnings).unwrap();
        assert_eq!(folders.len(), 3_000);
        let held: usize = folders.iter().flatten().map(|n| n.len() + size_of::<String>()).sum();
        assert!(held <= MAX_FOLDER_BYTES, "{held}");
        assert_eq!(folders[0].len(), MAX_FOLDER_DEPTH);
        assert!(folders[2_999].is_empty(), "past the budget: top level");
        assert!(warnings.iter().any(|w| w.contains("too large")), "{warnings:?}");
    }

    #[test]
    fn phry_round_trips_through_the_writer() {
        let presets: Vec<Descriptor> = (0..6).map(|i| named(&format!("B{i}"))).collect();
        let folders = vec![path(&[]), path(&["Set", "Ink"]), path(&["Set", "Ink"]), path(&["Set"]), path(&["Other"]), path(&["Set", "Ink"])];
        let bytes = write_v6_folders(2, &[], &[], &presets, &folders, true).unwrap();
        let f = parse(&bytes).unwrap();
        assert_eq!(f.presets, presets);
        assert_eq!(f.folders, folders);
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
        // No folders: no phry section, same bytes as write_v6.
        assert_eq!(write_v6_folders(2, &[], &[], &presets, &vec![Vec::new(); 6], true).unwrap(), write_v6(2, &[], &[], &presets, true).unwrap());
        // Mismatched lengths are an error.
        assert!(write_v6_folders(2, &[], &[], &presets, &folders[..2], true).is_err());
    }
}
