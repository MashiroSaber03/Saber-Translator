//! A small, bounds-checked TIFF 6.0 / TIFF-EP structure reader.
//!
//! Only what raw decoding needs: IFD entries with typed value access, the IFD
//! chain, SubIFDs (tag 330), the EXIF IFD (34665) and IFDs embedded at other
//! offsets (maker notes). Every offset is checked against the buffer; reads
//! that fall outside return `None`. The walk is bounded in IFD count and
//! entry count so hostile files cannot loop or blow up memory.

/// Tag numbers (TIFF 6.0, TIFF/EP, EXIF and DNG 1.7).
pub(crate) mod tag {
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PREDICTOR: u16 = 317;
    pub const PHOTOMETRIC: u16 = 262;
    pub const MAKE: u16 = 271;
    pub const MODEL: u16 = 272;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const ORIENTATION: u16 = 274;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    pub const PLANAR_CONFIGURATION: u16 = 284;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const SUB_IFDS: u16 = 330;
    pub const SAMPLE_FORMAT: u16 = 339;
    pub const JPEG_INTERCHANGE_FORMAT: u16 = 513;
    pub const JPEG_INTERCHANGE_FORMAT_LENGTH: u16 = 514;
    pub const CFA_REPEAT_PATTERN_DIM: u16 = 33421;
    pub const CFA_PATTERN: u16 = 33422;
    pub const EXIF_IFD: u16 = 34665;
    pub const EXIF_CFA_PATTERN: u16 = 41730;
    pub const MAKER_NOTE: u16 = 37500;
    pub const COLOR_SPACE: u16 = 40961;
    pub const INTEROP_IFD: u16 = 40965;
    /// In the Interoperability IFD.
    pub const INTEROP_INDEX: u16 = 1;
    pub const DNG_VERSION: u16 = 50706;
    pub const UNIQUE_CAMERA_MODEL: u16 = 50708;
    pub const CFA_PLANE_COLOR: u16 = 50710;
    pub const CFA_LAYOUT: u16 = 50711;
    pub const LINEARIZATION_TABLE: u16 = 50712;
    pub const BLACK_LEVEL_REPEAT_DIM: u16 = 50713;
    pub const BLACK_LEVEL: u16 = 50714;
    pub const BLACK_LEVEL_DELTA_H: u16 = 50715;
    pub const BLACK_LEVEL_DELTA_V: u16 = 50716;
    pub const WHITE_LEVEL: u16 = 50717;
    pub const DEFAULT_SCALE: u16 = 50718;
    pub const DEFAULT_CROP_ORIGIN: u16 = 50719;
    pub const DEFAULT_CROP_SIZE: u16 = 50720;
    pub const COLOR_MATRIX_1: u16 = 50721;
    pub const COLOR_MATRIX_2: u16 = 50722;
    pub const CAMERA_CALIBRATION_1: u16 = 50723;
    pub const CAMERA_CALIBRATION_2: u16 = 50724;
    pub const ANALOG_BALANCE: u16 = 50727;
    pub const AS_SHOT_NEUTRAL: u16 = 50728;
    pub const AS_SHOT_WHITE_XY: u16 = 50729;
    pub const BASELINE_EXPOSURE: u16 = 50730;
    pub const CALIBRATION_ILLUMINANT_1: u16 = 50778;
    pub const CALIBRATION_ILLUMINANT_2: u16 = 50779;
    pub const ACTIVE_AREA: u16 = 50829;
    pub const FORWARD_MATRIX_1: u16 = 50964;
    pub const FORWARD_MATRIX_2: u16 = 50965;
    pub const OPCODE_LIST_1: u16 = 51008;
    pub const OPCODE_LIST_2: u16 = 51009;
    pub const OPCODE_LIST_3: u16 = 51022;
    pub const BASELINE_EXPOSURE_OFFSET: u16 = 51109;
    pub const PROFILE_GAIN_TABLE_MAP: u16 = 52525;
    // CR2 private tag in the raw IFD: slice count, slice width, last slice width.
    pub const CR2_SLICE: u16 = 50752;
}

/// Most IFDs a walk will visit.
const MAX_IFDS: usize = 64;
/// Most entries read from one IFD.
const MAX_ENTRIES: usize = 4096;
/// Most values decoded from one entry by the list accessors.
pub(crate) const MAX_VALUES: usize = 1 << 20;

/// One directory entry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Entry {
    pub tag: u16,
    pub typ: u16,
    pub count: u32,
    /// Absolute offset of the value bytes in the buffer (may be out of range;
    /// accessors check).
    pub at: usize,
}

impl Entry {
    /// Size in bytes of one value of this entry's type (0 = unknown type).
    pub(crate) fn unit(&self) -> usize {
        type_size(self.typ)
    }
}

/// One image file directory.
#[derive(Debug, Clone, Default)]
pub(crate) struct Ifd {
    pub offset: usize,
    pub entries: Vec<Entry>,
}

impl Ifd {
    pub fn get(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|e| e.tag == tag)
    }
    pub fn has(&self, tag: u16) -> bool {
        self.get(tag).is_some()
    }
}

pub(crate) fn type_size(typ: u16) -> usize {
    match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => 0,
    }
}

/// A TIFF-structured buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tiff<'a> {
    pub data: &'a [u8],
    pub le: bool,
    /// Offset of IFD0 from the header.
    pub first_ifd: usize,
}

impl<'a> Tiff<'a> {
    /// Parses the 8-byte header. Accepts the TIFF magics and the TIFF-like
    /// variants used by camera raws (ORF `IIRO`/`IIRS`/`MMOR`, RW2 `IIU\0`).
    pub fn new(data: &'a [u8]) -> Option<Self> {
        let head = data.get(0..4)?;
        let le = match head {
            b"II*\0" | b"IIRO" | b"IIRS" | b"IIU\0" => true,
            b"MM\0*" | b"MMOR" => false,
            _ => return None,
        };
        let mut t = Tiff { data, le, first_ifd: 0 };
        t.first_ifd = t.u32_at(4)? as usize;
        Some(t)
    }

    pub fn bytes(&self, at: usize, len: usize) -> Option<&'a [u8]> {
        self.data.get(at..at.checked_add(len)?)
    }

    pub fn u16_at(&self, at: usize) -> Option<u16> {
        let s: [u8; 2] = self.bytes(at, 2)?.try_into().ok()?;
        Some(if self.le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) })
    }

    pub fn u32_at(&self, at: usize) -> Option<u32> {
        let s: [u8; 4] = self.bytes(at, 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
    }

    fn u64_at(&self, at: usize) -> Option<u64> {
        let s: [u8; 8] = self.bytes(at, 8)?.try_into().ok()?;
        Some(if self.le { u64::from_le_bytes(s) } else { u64::from_be_bytes(s) })
    }

    /// Reads the IFD at `offset`. Value offsets are relative to `base`
    /// (0 for ordinary IFDs; maker notes sometimes use their own base).
    pub fn ifd_at(&self, offset: usize, base: usize) -> Option<Ifd> {
        let n = usize::from(self.u16_at(offset)?);
        if n == 0 || n > MAX_ENTRIES {
            return None;
        }
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let e = offset.checked_add(2 + 12 * i)?;
            let (Some(tag), Some(typ), Some(count)) = (self.u16_at(e), self.u16_at(e.saturating_add(2)), self.u32_at(e.saturating_add(4))) else {
                break;
            };
            let size = type_size(typ).saturating_mul(count as usize);
            let at = if size <= 4 { e.saturating_add(8) } else { (self.u32_at(e.saturating_add(8))? as usize).saturating_add(base) };
            entries.push(Entry { tag, typ, count, at });
        }
        Some(Ifd { offset, entries })
    }

    /// Offset of the IFD that follows the one at `ifd` (0 = none).
    pub fn next_ifd(&self, ifd: &Ifd) -> usize {
        let n = ifd.entries.len();
        self.u32_at(ifd.offset.saturating_add(2 + 12 * n)).unwrap_or(0) as usize
    }

    /// Every IFD reachable from the header: the main chain, SubIFDs and the
    /// EXIF IFD, depth-first, deduplicated and bounded.
    pub fn all_ifds(&self) -> Vec<Ifd> {
        let mut out: Vec<Ifd> = Vec::new();
        let mut seen: Vec<usize> = Vec::new();
        let mut stack = vec![self.first_ifd];
        while let Some(off) = stack.pop() {
            if off == 0 || seen.contains(&off) || seen.len() >= MAX_IFDS {
                continue;
            }
            seen.push(off);
            let Some(ifd) = self.ifd_at(off, 0) else { continue };
            let next = self.next_ifd(&ifd);
            if next != 0 {
                stack.push(next);
            }
            for t in [tag::SUB_IFDS, tag::EXIF_IFD] {
                if let Some(e) = ifd.get(t) {
                    // Reverse so the first SubIFD is visited first.
                    let mut v = self.uints(e);
                    v.truncate(16);
                    stack.extend(v.into_iter().rev().map(|o| o as usize));
                }
            }
            out.push(ifd);
        }
        out
    }

    /// Value `i` of an entry as an unsigned integer (BYTE/SHORT/LONG/IFD and
    /// their signed variants, negative values are rejected).
    pub fn uint(&self, e: &Entry, i: usize) -> Option<u32> {
        if i >= e.count as usize {
            return None;
        }
        let at = e.at.checked_add(i.checked_mul(e.unit())?)?;
        match e.typ {
            1 | 7 => self.bytes(at, 1).map(|b| u32::from(b[0])),
            6 => self.bytes(at, 1).and_then(|b| u32::try_from(b[0] as i8).ok()),
            3 => self.u16_at(at).map(u32::from),
            8 => self.u16_at(at).and_then(|v| u32::try_from(v as i16).ok()),
            4 | 13 => self.u32_at(at),
            9 => self.u32_at(at).and_then(|v| u32::try_from(v as i32).ok()),
            _ => None,
        }
    }

    /// Value `i` as a float (any numeric type, rationals included).
    pub fn float(&self, e: &Entry, i: usize) -> Option<f64> {
        if i >= e.count as usize {
            return None;
        }
        let at = e.at.checked_add(i.checked_mul(e.unit())?)?;
        let v = match e.typ {
            1 | 7 => f64::from(self.bytes(at, 1)?[0]),
            6 => f64::from(self.bytes(at, 1)?[0] as i8),
            3 => f64::from(self.u16_at(at)?),
            8 => f64::from(self.u16_at(at)? as i16),
            4 | 13 => f64::from(self.u32_at(at)?),
            9 => f64::from(self.u32_at(at)? as i32),
            5 => {
                let (n, d) = (self.u32_at(at)?, self.u32_at(at.saturating_add(4))?);
                if d == 0 {
                    return None;
                }
                f64::from(n) / f64::from(d)
            }
            10 => {
                let (n, d) = (self.u32_at(at)? as i32, self.u32_at(at.saturating_add(4))? as i32);
                if d == 0 {
                    return None;
                }
                f64::from(n) / f64::from(d)
            }
            11 => f64::from(f32::from_bits(self.u32_at(at)?)),
            12 => f64::from_bits(self.u64_at(at)?),
            _ => return None,
        };
        v.is_finite().then_some(v)
    }

    /// All values as unsigned integers (empty if any is unreadable).
    pub fn uints(&self, e: &Entry) -> Vec<u32> {
        let n = (e.count as usize).min(MAX_VALUES);
        (0..n).map(|i| self.uint(e, i)).collect::<Option<Vec<_>>>().unwrap_or_default()
    }

    /// All values as floats (empty if any is unreadable).
    pub fn floats(&self, e: &Entry) -> Vec<f64> {
        let n = (e.count as usize).min(MAX_VALUES);
        (0..n).map(|i| self.float(e, i)).collect::<Option<Vec<_>>>().unwrap_or_default()
    }

    /// The raw value bytes of an entry.
    pub fn raw(&self, e: &Entry) -> Option<&'a [u8]> {
        let len = e.unit().checked_mul(e.count as usize)?;
        self.bytes(e.at, len)
    }

    /// An ASCII value, trimmed of NULs and whitespace.
    pub fn ascii(&self, e: &Entry) -> Option<String> {
        let b = self.raw(e)?;
        let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        let s = String::from_utf8_lossy(&b[..end]).trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    pub fn tag_uint(&self, ifd: &Ifd, t: u16) -> Option<u32> {
        ifd.get(t).and_then(|e| self.uint(e, 0))
    }

    pub fn tag_uints(&self, ifd: &Ifd, t: u16) -> Vec<u32> {
        ifd.get(t).map(|e| self.uints(e)).unwrap_or_default()
    }

    pub fn tag_floats(&self, ifd: &Ifd, t: u16) -> Vec<f64> {
        ifd.get(t).map(|e| self.floats(e)).unwrap_or_default()
    }

    pub fn tag_ascii(&self, ifd: &Ifd, t: u16) -> Option<String> {
        ifd.get(t).and_then(|e| self.ascii(e))
    }
}

/// The EXIF orientation (1..=8) of a raw tag value; anything else is the default 1. The range is
/// checked on the `u32`: narrowing first let a LONG like 65538 pass as 2 (#1817).
pub(crate) fn orientation(v: Option<u32>) -> u16 {
    v.filter(|o| (1..=8).contains(o)).and_then(|o| u16::try_from(o).ok()).unwrap_or(1)
}

/// A human-readable listing of every IFD and entry (debugging aid).
pub(crate) fn dump(data: &[u8]) -> String {
    use std::fmt::Write;
    let Some(t) = Tiff::new(data) else { return "not a TIFF-structured file".into() };
    let mut s = String::new();
    for ifd in t.all_ifds() {
        let _ = writeln!(s, "IFD @{}", ifd.offset);
        for e in &ifd.entries {
            let preview: String = if e.typ == 2 {
                t.ascii(e).unwrap_or_default()
            } else {
                let n = (e.count as usize).min(12);
                let v: Vec<String> = (0..n).map(|i| t.float(e, i).map(|f| format!("{f}")).unwrap_or_else(|| "?".into())).collect();
                format!("{}{}", v.join(" "), if e.count as usize > n { " …" } else { "" })
            };
            let _ = writeln!(s, "  {:5} (0x{:04X}) type {:2} count {:7} @{:9}: {}", e.tag, e.tag, e.typ, e.count, e.at, preview);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_is_range_checked_before_narrowing() {
        // #1817: 65538 as u16 is 2, which passed the 1..=8 check.
        for (v, want) in [(None, 1), (Some(0), 1), (Some(9), 1), (Some(65_538), 1), (Some(65_537), 1), (Some(u32::MAX), 1)] {
            assert_eq!(orientation(v), want, "{v:?}");
        }
        for o in 1..=8u32 {
            assert_eq!(orientation(Some(o)), o as u16);
        }
    }

    #[test]
    fn header_variants() {
        assert!(Tiff::new(b"II*\0\x08\0\0\0").is_some());
        assert!(Tiff::new(b"MM\0*\0\0\0\x08").is_some());
        assert!(Tiff::new(b"IIRO\x08\0\0\0").is_some());
        assert!(Tiff::new(b"IIU\0\x08\0\0\0").is_some());
        assert!(Tiff::new(b"II*\0").is_none());
        assert!(Tiff::new(b"PNG\0\0\0\0\0").is_none());
    }

    #[test]
    fn self_referencing_ifds_terminate() {
        // IFD0 at 8 with one entry whose SubIFD points at itself, and a next
        // pointer that also points at itself.
        let mut b = b"II*\0\x08\0\0\0".to_vec();
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&330u16.to_le_bytes());
        b.extend_from_slice(&4u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        let t = Tiff::new(&b).unwrap();
        assert_eq!(t.all_ifds().len(), 1);
    }

    #[test]
    fn rational_with_zero_denominator_is_none() {
        let mut b = b"II*\0\x08\0\0\0".to_vec();
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&50730u16.to_le_bytes());
        b.extend_from_slice(&10u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&26u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&5i32.to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        let t = Tiff::new(&b).unwrap();
        let ifd = t.ifd_at(8, 0).unwrap();
        assert_eq!(t.float(ifd.get(50730).unwrap(), 0), None);
    }
}
