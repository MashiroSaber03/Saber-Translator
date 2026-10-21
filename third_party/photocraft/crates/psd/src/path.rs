//! Path records: saved paths (image resources 2000–2997), the work path
//! (resource 1025) and vector masks (`vmsk` / `vsms` layer blocks).
//!
//! Implemented from the public Adobe "Photoshop File Formats Specification",
//! section "Path resource format": a path is a list of 26-byte records, each a
//! 2-byte selector followed by 24 bytes of data. Knot coordinates are signed
//! 8.24 fixed-point fractions of the document height (vertical) and width
//! (horizontal), stored vertical first.
//!
//! Undocumented detail (as documented by the MIT-licensed psd-tools): bytes
//! 2–3 of a subpath length record's data (after the knot count) hold the
//! *path operation* of the subpath (`-1` or `1` combine/or, `2` subtract,
//! `3` intersect, `0` exclude/xor). Everything else is kept raw, so parsing
//! and writing a record list is byte-exact.

use crate::error::{PsdError, Result};

/// Length of one path record in bytes (selector + data).
pub const RECORD_LEN: usize = 26;

/// Record selectors (spec table "Path data record types").
pub mod selector {
    /// Closed subpath length record.
    pub const CLOSED_LENGTH: u16 = 0;
    /// Closed subpath Bézier knot, linked.
    pub const CLOSED_KNOT_LINKED: u16 = 1;
    /// Closed subpath Bézier knot, unlinked.
    pub const CLOSED_KNOT_UNLINKED: u16 = 2;
    /// Open subpath length record.
    pub const OPEN_LENGTH: u16 = 3;
    /// Open subpath Bézier knot, linked.
    pub const OPEN_KNOT_LINKED: u16 = 4;
    /// Open subpath Bézier knot, unlinked.
    pub const OPEN_KNOT_UNLINKED: u16 = 5;
    /// Path fill rule record.
    pub const FILL_RULE: u16 = 6;
    /// Clipboard record.
    pub const CLIPBOARD: u16 = 7;
    /// Initial fill rule record.
    pub const INITIAL_FILL: u16 = 8;
}

/// One 26-byte path record, kept raw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathRecord {
    /// Record type (see [`selector`]).
    pub selector: u16,
    /// The 24 data bytes.
    pub data: [u8; 24],
}

impl PathRecord {
    fn u16_at(&self, at: usize) -> u16 {
        u16::from_be_bytes([self.data[at], self.data[at + 1]])
    }
    fn i32_at(&self, at: usize) -> i32 {
        i32::from_be_bytes([self.data[at], self.data[at + 1], self.data[at + 2], self.data[at + 3]])
    }
    /// Whether this is a (closed or open) subpath length record.
    pub fn is_length(&self) -> bool {
        matches!(self.selector, selector::CLOSED_LENGTH | selector::OPEN_LENGTH)
    }
    /// Whether this is a Bézier knot record.
    pub fn is_knot(&self) -> bool {
        matches!(self.selector, 1 | 2 | 4 | 5)
    }
}

/// A Bézier knot as fractions of the document size: `(x, y)` pairs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsdKnot {
    /// Linked (smooth) knot: the control points move together.
    pub linked: bool,
    /// Control point preceding the anchor (`(x, y)` fractions).
    pub pre: (f64, f64),
    /// Anchor point.
    pub anchor: (f64, f64),
    /// Control point leaving the anchor.
    pub post: (f64, f64),
}

/// A subpath: its knots, closed flag and path operation.
#[derive(Clone, Debug, PartialEq)]
pub struct PsdSubpath {
    /// Closed subpath.
    pub closed: bool,
    /// Path operation (bytes 2–3 of the length record data): `-1`/`1`
    /// combine, `2` subtract, `3` intersect, `0` exclude.
    pub operation: i16,
    /// Knots in order.
    pub knots: Vec<PsdKnot>,
}

/// Fixed 8.24 → fraction.
pub fn fixed_to_f64(v: i32) -> f64 {
    f64::from(v) / 16_777_216.0
}

/// Fraction → fixed 8.24 (rounded, saturating).
pub fn f64_to_fixed(v: f64) -> i32 {
    (v * 16_777_216.0).round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// A parsed path record list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathData {
    /// Records in file order.
    pub records: Vec<PathRecord>,
}

impl PathData {
    /// Parses path records. Trailing bytes shorter than a record are ignored.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() / RECORD_LEN > 4_000_000 {
            return Err(PsdError::LimitExceeded("path records"));
        }
        let records = data
            .as_chunks::<RECORD_LEN>()
            .0
            .iter()
            .map(|c| {
                let mut d = [0u8; 24];
                d.copy_from_slice(&c[2..]);
                PathRecord { selector: u16::from_be_bytes([c[0], c[1]]), data: d }
            })
            .collect();
        Ok(PathData { records })
    }

    /// Serializes the records.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.records.len() * RECORD_LEN);
        for r in &self.records {
            out.extend_from_slice(&r.selector.to_be_bytes());
            out.extend_from_slice(&r.data);
        }
        out
    }

    /// Initial fill rule: `true` when the path starts with all pixels filled
    /// (inverted path).
    pub fn initial_fill(&self) -> bool {
        self.records.iter().find(|r| r.selector == selector::INITIAL_FILL).is_some_and(|r| r.u16_at(0) != 0)
    }

    /// Subpaths with their knots. Knot records not preceded by a length
    /// record, or beyond a length record's count, start/extend a subpath
    /// leniently instead of failing.
    pub fn subpaths(&self) -> Vec<PsdSubpath> {
        let mut out: Vec<PsdSubpath> = Vec::new();
        let mut remaining = 0usize;
        for r in &self.records {
            if r.is_length() {
                remaining = usize::from(r.u16_at(0));
                out.push(PsdSubpath {
                    closed: r.selector == selector::CLOSED_LENGTH,
                    operation: r.u16_at(2) as i16,
                    // No preallocation: the count is untrusted (u16) and PsdKnot is 56 bytes.
                    knots: Vec::new(),
                });
            } else if r.is_knot() {
                let closed = r.selector <= 2;
                if remaining == 0 && out.last().is_none_or(|s| s.closed != closed) {
                    out.push(PsdSubpath { closed, operation: -1, knots: Vec::new() });
                }
                remaining = remaining.saturating_sub(1);
                let pt = |i: usize| (fixed_to_f64(r.i32_at(i * 8 + 4)), fixed_to_f64(r.i32_at(i * 8)));
                if let Some(s) = out.last_mut() {
                    s.knots.push(PsdKnot { linked: matches!(r.selector, 1 | 4), pre: pt(0), anchor: pt(1), post: pt(2) });
                }
            }
        }
        out
    }

    /// Builds a record list: a fill rule record, an initial fill record,
    /// then each subpath's length record and knots.
    pub fn from_subpaths(subpaths: &[PsdSubpath], initial_fill: bool) -> Self {
        let mut records = vec![PathRecord { selector: selector::FILL_RULE, data: [0; 24] }, PathRecord { selector: selector::INITIAL_FILL, data: [0; 24] }];
        records[1].data[..2].copy_from_slice(&u16::from(initial_fill).to_be_bytes());
        for s in subpaths {
            let mut data = [0u8; 24];
            data[..2].copy_from_slice(&(s.knots.len().min(u16::MAX as usize) as u16).to_be_bytes());
            data[2..4].copy_from_slice(&s.operation.to_be_bytes());
            // Photoshop writes 1 here (psd-tools default); meaning unknown.
            data[4..6].copy_from_slice(&1u16.to_be_bytes());
            records.push(PathRecord { selector: if s.closed { selector::CLOSED_LENGTH } else { selector::OPEN_LENGTH }, data });
            for k in s.knots.iter().take(u16::MAX as usize) {
                let sel = match (s.closed, k.linked) {
                    (true, true) => selector::CLOSED_KNOT_LINKED,
                    (true, false) => selector::CLOSED_KNOT_UNLINKED,
                    (false, true) => selector::OPEN_KNOT_LINKED,
                    (false, false) => selector::OPEN_KNOT_UNLINKED,
                };
                let mut d = [0u8; 24];
                for (i, p) in [k.pre, k.anchor, k.post].iter().enumerate() {
                    d[i * 8..i * 8 + 4].copy_from_slice(&f64_to_fixed(p.1).to_be_bytes());
                    d[i * 8 + 4..i * 8 + 8].copy_from_slice(&f64_to_fixed(p.0).to_be_bytes());
                }
                records.push(PathRecord { selector: sel, data: d });
            }
        }
        PathData { records }
    }
}

/// A vector mask block (`vmsk` / `vsms`): version, flags and path records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorMaskBlock {
    /// Version (3).
    pub version: u32,
    /// Flags: bit 0 invert, bit 1 not linked, bit 2 disabled.
    pub flags: u32,
    /// The path.
    pub path: PathData,
}

impl VectorMaskBlock {
    /// Invert flag.
    pub const FLAG_INVERT: u32 = 1;
    /// Not-linked flag.
    pub const FLAG_NOT_LINKED: u32 = 2;
    /// Disabled flag.
    pub const FLAG_DISABLED: u32 = 4;

    /// Parses the block data.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(PsdError::UnexpectedEof { offset: 0, needed: 8 - data.len() });
        }
        let version = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let flags = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        Ok(VectorMaskBlock { version, flags, path: PathData::from_bytes(&data[8..])? })
    }

    /// Serializes the block data.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.path.records.len() * RECORD_LEN);
        out.extend_from_slice(&self.version.to_be_bytes());
        out.extend_from_slice(&self.flags.to_be_bytes());
        out.extend_from_slice(&self.path.to_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<PsdSubpath> {
        vec![
            PsdSubpath {
                closed: true,
                operation: 1,
                knots: vec![
                    PsdKnot { linked: false, pre: (0.25, 0.25), anchor: (0.25, 0.25), post: (0.25, 0.25) },
                    PsdKnot { linked: true, pre: (0.625, 0.125), anchor: (0.75, 0.25), post: (0.875, 0.375) },
                    PsdKnot { linked: false, pre: (0.5, 0.75), anchor: (0.5, 0.75), post: (0.5, 0.75) },
                ],
            },
            PsdSubpath { closed: false, operation: 2, knots: vec![PsdKnot { linked: false, pre: (-0.5, 1.5), anchor: (0.0, 1.0), post: (0.0625, 0.9375) }] },
        ]
    }

    #[test]
    fn subpaths_does_not_preallocate_from_untrusted_counts() {
        let mut bytes = Vec::new();
        for _ in 0..100_000 {
            bytes.extend_from_slice(&selector::CLOSED_LENGTH.to_be_bytes());
            bytes.extend_from_slice(&0xFFFFu16.to_be_bytes());
            bytes.extend_from_slice(&[0u8; 22]);
        }
        let subs = PathData::from_bytes(&bytes).unwrap().subpaths();
        assert_eq!(subs.len(), 100_000);
        assert!(subs.iter().all(|s| s.knots.capacity() == 0));
    }

    #[test]
    fn subpaths_roundtrip_and_bytes_are_stable() {
        let p = PathData::from_subpaths(&sample(), true);
        let bytes = p.to_bytes();
        assert_eq!(bytes.len(), (2 + 2 + 4) * RECORD_LEN);
        let q = PathData::from_bytes(&bytes).unwrap();
        assert_eq!(q.to_bytes(), bytes);
        assert_eq!(q.subpaths(), sample());
        assert!(q.initial_fill());
    }

    #[test]
    fn vector_mask_block_roundtrip() {
        let b = VectorMaskBlock { version: 3, flags: VectorMaskBlock::FLAG_INVERT, path: PathData::from_subpaths(&sample(), false) };
        let bytes = b.to_bytes();
        assert_eq!(VectorMaskBlock::from_bytes(&bytes).unwrap(), b);
        assert!(VectorMaskBlock::from_bytes(&[0; 4]).is_err());
    }

    #[test]
    fn fixed_point_conversion() {
        assert_eq!(f64_to_fixed(1.0), 1 << 24);
        assert_eq!(fixed_to_f64(-(1 << 23)), -0.5);
        assert_eq!(f64_to_fixed(1e12), i32::MAX);
    }

    #[test]
    fn lenient_knots_without_length_record() {
        let mut p = PathData::from_subpaths(&sample(), false);
        p.records.retain(|r| !r.is_length());
        let s = p.subpaths();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].knots.len(), 3);
    }
}
