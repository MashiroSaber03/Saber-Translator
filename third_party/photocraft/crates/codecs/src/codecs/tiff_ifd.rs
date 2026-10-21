//! A bounds-checked reader for the TIFF directory structure: classic TIFF (version 42, 4-byte
//! offsets) and BigTIFF (version 43, 8-byte offsets and the LONG8 / SLONG8 / IFD8 types), in
//! either byte order.
//!
//! Every offset and count comes from the file and is untrusted: reads go through `get` with
//! checked arithmetic, directory tables must lie wholly inside the file, the IFD chain and the
//! SubIFD trees (tag 330) are walked with a seen-set, a depth limit and a cap on the number of
//! directories, and nothing is allocated in proportion to a declared count before the bytes
//! behind it are known to exist.
//!
//! This module lists the pages of a file ([`TiffInfo`]) and reads the metadata of a directory
//! without decoding pixels; `tiff.rs` decodes the pixels of one directory.

use std::collections::HashSet;

use crate::error::CodecError;
use crate::format::Format;

/// At most this many directories are listed (main chain and SubIFDs together).
pub(crate) const MAX_IFDS: usize = 10_000;
/// Entries per directory: the classic format's own limit (a `u16` count), applied to BigTIFF too.
const MAX_ENTRIES: u64 = u16::MAX as u64;
/// SubIFD nesting depth that is followed (real files use one level).
const MAX_SUB_DEPTH: usize = 4;
/// SubIFD offsets followed per directory.
const MAX_SUBIFDS: u64 = 256;

pub(crate) const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub(crate) const TAG_SUBFILE_TYPE: u16 = 255;
pub(crate) const TAG_WIDTH: u16 = 256;
pub(crate) const TAG_HEIGHT: u16 = 257;
pub(crate) const TAG_BITS: u16 = 258;
pub(crate) const TAG_COMPRESSION: u16 = 259;
pub(crate) const TAG_PHOTOMETRIC: u16 = 262;
pub(crate) const TAG_ORIENTATION: u16 = 274;
pub(crate) const TAG_SAMPLES: u16 = 277;
pub(crate) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(crate) const TAG_PLANAR: u16 = 284;
pub(crate) const TAG_COLOR_MAP: u16 = 320;
pub(crate) const TAG_TILE_WIDTH: u16 = 322;
pub(crate) const TAG_TILE_LENGTH: u16 = 323;
pub(crate) const TAG_TILE_OFFSETS: u16 = 324;
pub(crate) const TAG_SUB_IFDS: u16 = 330;
pub(crate) const TAG_EXTRA_SAMPLES: u16 = 338;
pub(crate) const TAG_SAMPLE_FORMAT: u16 = 339;

/// Tags the `tiff` crate reads to set up decoding. Their values must lie inside the file, so a
/// hostile count can neither make it allocate nor read past the end.
pub(crate) const STRUCTURAL_TAGS: &[u16] = &[256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325, 338, 339, 347, 530];

/// Bytes per value of a TIFF field type; `None` for unknown types (whose entries are skipped).
fn type_size(ty: u16) -> Option<u64> {
    Some(match ty {
        1 | 2 | 6 | 7 => 1,              // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,                      // SHORT, SSHORT
        4 | 9 | 11 | 13 => 4,            // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 | 16 | 17 | 18 => 8, // RATIONAL, SRATIONAL, DOUBLE, LONG8, SLONG8, IFD8
        _ => return None,
    })
}

/// One directory entry, located but not read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub tag: u16,
    pub ty: u16,
    pub count: u64,
    /// Position of the value: inside the entry when it fits, else the stored offset.
    pub value_at: u64,
}

/// One directory: its entries in file order and the next-IFD offset (0 at the end of a chain).
#[derive(Clone, Debug)]
pub(crate) struct Ifd {
    pub at: u64,
    pub entries: Vec<Entry>,
    pub next: u64,
    /// Position of the next-IFD pointer field.
    pub next_at: u64,
}

impl Ifd {
    /// The first entry with this tag.
    pub fn get(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|e| e.tag == tag)
    }
}

/// A TIFF or BigTIFF file's header and bytes.
#[derive(Clone, Copy)]
pub(crate) struct File<'a> {
    pub b: &'a [u8],
    pub le: bool,
    pub big: bool,
    pub first_ifd: u64,
}

impl<'a> File<'a> {
    /// The header, or `None` when this is not a (Big)TIFF.
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        let le = match b.get(0..2)? {
            b"II" => true,
            b"MM" => false,
            _ => return None,
        };
        let mut f = File { b, le, big: false, first_ifd: 0 };
        match f.u16_at(2)? {
            42 => f.first_ifd = u64::from(f.u32_at(4)?),
            43 => {
                // Bytesize of offsets (always 8) and a reserved zero.
                if f.u16_at(4)? != 8 || f.u16_at(6)? != 0 {
                    return None;
                }
                f.big = true;
                f.first_ifd = f.u64_at(8)?;
            }
            _ => return None,
        }
        Some(f)
    }

    /// Length of the header; no directory can start inside it.
    pub fn header_len(&self) -> u64 {
        if self.big { 16 } else { 8 }
    }

    fn slice(&self, at: u64, len: u64) -> Option<&'a [u8]> {
        let at = usize::try_from(at).ok()?;
        let len = usize::try_from(len).ok()?;
        self.b.get(at..at.checked_add(len)?)
    }

    pub fn u16_at(&self, at: u64) -> Option<u16> {
        let s: [u8; 2] = self.slice(at, 2)?.try_into().ok()?;
        Some(if self.le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) })
    }

    pub fn u32_at(&self, at: u64) -> Option<u32> {
        let s: [u8; 4] = self.slice(at, 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
    }

    pub fn u64_at(&self, at: u64) -> Option<u64> {
        let s: [u8; 8] = self.slice(at, 8)?.try_into().ok()?;
        Some(if self.le { u64::from_le_bytes(s) } else { u64::from_be_bytes(s) })
    }

    /// The declared entry count of the directory at `at`, before its table is parsed.
    pub fn entry_count(&self, at: u64) -> Option<u64> {
        if at < self.header_len() {
            return None;
        }
        if self.big { self.u64_at(at) } else { self.u16_at(at).map(u64::from) }
    }

    /// The directory at `at`: its whole table, up to and including the next-IFD pointer, must
    /// be inside the file and after the header.
    pub fn ifd(&self, at: u64) -> Option<Ifd> {
        if at < self.header_len() {
            return None;
        }
        let (n, first, entry, inline) =
            if self.big { (self.u64_at(at)?, at.checked_add(8)?, 20u64, 8u64) } else { (u64::from(self.u16_at(at)?), at.checked_add(2)?, 12, 4) };
        if n > MAX_ENTRIES {
            return None;
        }
        let next_at = first.checked_add(n.checked_mul(entry)?)?;
        let next = if self.big { self.u64_at(next_at)? } else { u64::from(self.u32_at(next_at)?) };
        let mut entries = Vec::with_capacity(usize::try_from(n).ok()?);
        for i in 0..n {
            let e = first.checked_add(i.checked_mul(entry)?)?;
            let tag = self.u16_at(e)?;
            let ty = self.u16_at(e.checked_add(2)?)?;
            let count = if self.big { self.u64_at(e.checked_add(4)?)? } else { u64::from(self.u32_at(e.checked_add(4)?)?) };
            let field = e.checked_add(if self.big { 12 } else { 8 })?;
            // Unknown types are skipped, as the specification says.
            let Some(size) = type_size(ty) else { continue };
            let fits = count.checked_mul(size).is_some_and(|len| len <= inline);
            let value_at = if fits {
                field
            } else if self.big {
                self.u64_at(field)?
            } else {
                u64::from(self.u32_at(field)?)
            };
            entries.push(Entry { tag, ty, count, value_at });
        }
        Some(Ifd { at, entries, next, next_at })
    }

    /// The value bytes of an entry, when they are all inside the file.
    pub fn data(&self, e: &Entry) -> Option<&'a [u8]> {
        self.slice(e.value_at, e.count.checked_mul(type_size(e.ty)?)?)
    }

    /// Value `i` of an unsigned integer entry (BYTE, SHORT, LONG, LONG8, IFD, IFD8).
    pub fn uint(&self, e: &Entry, i: u64) -> Option<u64> {
        if i >= e.count {
            return None;
        }
        let size = type_size(e.ty)?;
        let at = e.value_at.checked_add(i.checked_mul(size)?)?;
        match e.ty {
            1 => self.slice(at, 1)?.first().map(|&v| u64::from(v)),
            3 => self.u16_at(at).map(u64::from),
            4 | 13 => self.u32_at(at).map(u64::from),
            16 | 18 => self.u64_at(at),
            _ => None,
        }
    }

    /// The first value of an unsigned integer tag.
    pub fn tag_uint(&self, ifd: &Ifd, tag: u16) -> Option<u64> {
        self.uint(ifd.get(tag)?, 0)
    }

    /// Up to `max` values of an unsigned integer tag (`None` if any of them is unreadable).
    pub fn tag_uints(&self, ifd: &Ifd, tag: u16, max: u64) -> Option<Vec<u64>> {
        let e = ifd.get(tag)?;
        (0..e.count.min(max)).map(|i| self.uint(e, i)).collect()
    }

    /// The first value of a numeric tag as a float: RATIONAL, FLOAT, DOUBLE or an integer.
    pub fn tag_f64(&self, ifd: &Ifd, tag: u16) -> Option<f64> {
        let e = ifd.get(tag)?;
        if e.count == 0 {
            return None;
        }
        match e.ty {
            5 => {
                let (n, d) = (self.u32_at(e.value_at)?, self.u32_at(e.value_at.checked_add(4)?)?);
                (d != 0).then(|| f64::from(n) / f64::from(d))
            }
            11 => Some(f64::from(f32::from_bits(self.u32_at(e.value_at)?))),
            12 => Some(f64::from_bits(self.u64_at(e.value_at)?)),
            _ => self.uint(e, 0).map(|v| v as f64),
        }
    }

    /// The bytes of a one-byte-per-value tag (BYTE, ASCII, SBYTE or UNDEFINED).
    pub fn tag_bytes(&self, ifd: &Ifd, tag: u16) -> Option<&'a [u8]> {
        let e = ifd.get(tag)?;
        matches!(e.ty, 1 | 2 | 6 | 7).then(|| self.data(e)).flatten()
    }

    /// SubIFD offsets of a directory (tag 330 as LONG, IFD, LONG8 or IFD8).
    fn sub_ifds(&self, ifd: &Ifd) -> Vec<u64> {
        let Some(e) = ifd.get(TAG_SUB_IFDS) else { return Vec::new() };
        if !matches!(e.ty, 4 | 13 | 16 | 18) {
            return Vec::new();
        }
        (0..e.count.min(MAX_SUBIFDS)).map_while(|i| self.uint(e, i)).filter(|&o| o != 0).collect()
    }
}

/// A directory's Orientation (1–8), with the rules of [`crate::exif_orientation`]: the first
/// Orientation entry wins, it must be exactly one SHORT or LONG, and anything else reads as 1.
pub(crate) fn orientation_of(f: &File<'_>, ifd: &Ifd) -> u16 {
    ifd.get(TAG_ORIENTATION)
        .filter(|e| e.count == 1 && matches!(e.ty, 3 | 4))
        .and_then(|e| f.uint(e, 0))
        .and_then(|v| u16::try_from(v).ok())
        .filter(|v| (1..=8).contains(v))
        .unwrap_or(1)
}

/// What a TIFF directory holds, from `NewSubfileType` (254) or the older `SubfileType` (255).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffPageKind {
    /// A full-resolution image: a page of the document.
    Page,
    /// A reduced-resolution copy of another image (a thumbnail or a pyramid level).
    ReducedResolution,
    /// A transparency mask for another image.
    Mask,
}

/// One image directory of a TIFF file, described without decoding its pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiffPage {
    /// Byte offset of the directory in the file.
    pub ifd_offset: u64,
    /// For a SubIFD (tag 330): the index (in [`TiffInfo::pages`]) of the directory listing it.
    /// `None` for the directories of the main IFD chain.
    pub parent: Option<usize>,
    pub kind: TiffPageKind,
    /// 0 when the directory has no readable dimensions.
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u16,
    pub bits_per_sample: u16,
    /// TIFF `SampleFormat`: 1 unsigned integer, 2 signed integer, 3 IEEE float.
    pub sample_format: u16,
    /// TIFF `PhotometricInterpretation`, when present.
    pub photometric: Option<u16>,
    /// TIFF `Compression` (1 = none).
    pub compression: u16,
    /// Tiles rather than strips.
    pub tiled: bool,
    /// `PlanarConfiguration` 2: each sample in its own plane.
    pub planar: bool,
}

/// The directory structure of a TIFF file: its byte order, BigTIFF or not, and every image
/// directory of the main chain and its SubIFDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiffInfo {
    pub big_tiff: bool,
    pub little_endian: bool,
    /// Main-chain directories in file order, each followed by its SubIFDs (depth first).
    pub pages: Vec<TiffPage>,
    /// `false` when the walk stopped early: an unreadable directory, a cycle, or more than
    /// 10 000 directories. The pages listed up to that point are still valid.
    pub complete: bool,
}

impl TiffInfo {
    /// The page opened by default, like Photoshop: the first full-resolution directory of the
    /// main chain (reduced-resolution copies and masks are skipped), else the first directory.
    pub fn default_page(&self) -> Option<usize> {
        self.pages
            .iter()
            .position(|p| p.parent.is_none() && p.kind == TiffPageKind::Page && p.width > 0 && p.height > 0)
            .or_else(|| (!self.pages.is_empty()).then_some(0))
    }

    /// Full-resolution pages in the main chain (what Photoshop calls the pages of the file).
    pub fn page_count(&self) -> usize {
        self.pages.iter().filter(|p| p.parent.is_none() && p.kind == TiffPageKind::Page).count()
    }
}

fn describe(f: &File<'_>, ifd: &Ifd, parent: Option<usize>) -> TiffPage {
    let new_kind = f.tag_uint(ifd, TAG_NEW_SUBFILE_TYPE).unwrap_or(0);
    let old_kind = f.tag_uint(ifd, TAG_SUBFILE_TYPE).unwrap_or(1);
    let kind = if new_kind & 4 != 0 {
        TiffPageKind::Mask
    } else if new_kind & 1 != 0 || old_kind == 2 {
        TiffPageKind::ReducedResolution
    } else {
        TiffPageKind::Page
    };
    let u32_of = |tag| f.tag_uint(ifd, tag).and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    let u16_of = |tag, default| f.tag_uint(ifd, tag).and_then(|v| u16::try_from(v).ok()).unwrap_or(default);
    TiffPage {
        ifd_offset: ifd.at,
        parent,
        kind,
        width: u32_of(TAG_WIDTH),
        height: u32_of(TAG_HEIGHT),
        samples_per_pixel: u16_of(TAG_SAMPLES, 1),
        bits_per_sample: u16_of(TAG_BITS, 1),
        sample_format: u16_of(TAG_SAMPLE_FORMAT, 1),
        photometric: f.tag_uint(ifd, TAG_PHOTOMETRIC).and_then(|v| u16::try_from(v).ok()),
        compression: u16_of(TAG_COMPRESSION, 1),
        tiled: ifd.get(TAG_TILE_OFFSETS).is_some(),
        planar: f.tag_uint(ifd, TAG_PLANAR) == Some(2),
    }
}

struct Walk<'a, 'f> {
    f: &'f File<'a>,
    seen: HashSet<u64>,
    pages: Vec<TiffPage>,
    complete: bool,
    /// Directory entries that may still be parsed. Real directory tables are disjoint, so their
    /// total size is bounded by the file; overlapping tables (a hostile file) would otherwise
    /// make every one of thousands of directories cost a full 65,535-entry parse.
    entry_budget: u64,
}

impl Walk<'_, '_> {
    /// Follows a chain of directories starting at `at`, listing each with `parent` and then
    /// its SubIFDs.
    fn chain(&mut self, mut at: u64, parent: Option<usize>, depth: usize) {
        while at != 0 {
            if self.pages.len() >= MAX_IFDS || !self.seen.insert(at) {
                self.complete = false;
                return;
            }
            let Some(n) = self.f.entry_count(at) else {
                self.complete = false;
                return;
            };
            if n > self.entry_budget {
                self.complete = false;
                return;
            }
            self.entry_budget -= n;
            let Some(ifd) = self.f.ifd(at) else {
                self.complete = false;
                return;
            };
            let index = self.pages.len();
            self.pages.push(describe(self.f, &ifd, parent));
            if depth < MAX_SUB_DEPTH {
                for sub in self.f.sub_ifds(&ifd) {
                    self.chain(sub, Some(index), depth + 1);
                }
            }
            at = ifd.next;
        }
    }
}

/// Lists the directories of a (Big)TIFF file. An error only when the bytes are not a TIFF or
/// the first directory cannot be read; a damaged chain after it ends the list early
/// (`complete == false`).
pub fn tiff_info(bytes: &[u8]) -> Result<TiffInfo, CodecError> {
    let f = File::parse(bytes).ok_or_else(|| CodecError::malformed(Format::Tiff, "not a TIFF or BigTIFF header"))?;
    info_of(&f)
}

pub(crate) fn info_of(f: &File<'_>) -> Result<TiffInfo, CodecError> {
    let mut w = Walk { f, seen: HashSet::new(), pages: Vec::new(), complete: true, entry_budget: (f.b.len() as u64) / 12 + 64 };
    w.chain(f.first_ifd, None, 0);
    if w.pages.is_empty() {
        return Err(CodecError::malformed(Format::Tiff, "the first image directory is missing or cut off"));
    }
    Ok(TiffInfo { big_tiff: f.big, little_endian: f.le, pages: w.pages, complete: w.complete })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10,000 directories 4 bytes apart, each declaring 65,535 overlapping entries: the walk must
    /// stop after the file-size entry budget instead of parsing ~6.5e8 entries.
    #[test]
    fn overlapping_directory_tables_exhaust_the_entry_budget() {
        let first = 8usize;
        let table = 2 + 65_535 * 12 + 4;
        let mut b = vec![0u8; first + table + 4 * 10_000];
        b[..8].copy_from_slice(&[b'I', b'I', 42, 0, first as u8, 0, 0, 0]);
        for i in 0..10_000usize {
            let at = first + 4 * i;
            b[at..at + 2].copy_from_slice(&u16::MAX.to_le_bytes());
            let next_cell = at + 2 + 65_535 * 12;
            let next = (first + 4 * (i + 1)) as u32;
            if let Some(cell) = b.get_mut(next_cell..next_cell + 4) {
                cell.copy_from_slice(&next.to_le_bytes());
            }
        }
        let info = tiff_info(&b).expect("the first directory is readable");
        assert!(!info.complete);
        assert!(info.pages.len() < 200, "{} directories parsed", info.pages.len());
    }
}
