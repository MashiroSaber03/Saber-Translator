//! Bounds-checked big-endian reader and writer helpers.

use crate::error::{PsdError, Result};

/// A name as UTF-16 code units, capped at `max` **without ever splitting a surrogate pair**.
/// `encode_utf16().take(max)` would cut between the high and low half of a pair above U+FFFF and
/// write ill-formed UTF-16 to disk, so whole scalar values are added until one would not fit.
/// Both the `.aco` and `.ase` writers share this; keeping it in one place is deliberate, so the
/// surrogate fix cannot drift between the two formats.
pub(crate) fn utf16_units(name: &str, max: usize) -> Vec<u16> {
    let mut units = Vec::new();
    for c in name.chars() {
        if units.len() + c.len_utf16() > max {
            break;
        }
        let mut buf = [0u16; 2];
        units.extend_from_slice(c.encode_utf16(&mut buf));
    }
    units
}

/// A cursor over a byte slice. Every read is bounds checked. Integers are big-endian (the PSD
/// byte order) unless the reader was made with [`Reader::new_little`], which only the TIFF
/// byte-order transcoder uses ([`crate::tiff`]).
#[derive(Debug, Clone)]
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    little: bool,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0, little: false }
    }

    /// A reader of little-endian integers (Photoshop data inside an Intel-order TIFF).
    pub(crate) fn new_little(data: &'a [u8]) -> Self {
        Reader { data, pos: 0, little: true }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Bytes from the current position without consuming.
    pub(crate) fn peek_rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(PsdError::UnexpectedEof { offset: self.pos, needed: n - self.remaining() });
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Reads `n` bytes where `n` comes from a (possibly 64-bit) length field.
    pub(crate) fn bytes_u64(&mut self, n: u64) -> Result<&'a [u8]> {
        let n = usize::try_from(n).map_err(|_| PsdError::LimitExceeded("length exceeds address space"))?;
        self.bytes(n)
    }

    pub(crate) fn sub(&mut self, n: u64) -> Result<Reader<'a>> {
        let little = self.little;
        Ok(Reader { data: self.bytes_u64(n)?, pos: 0, little })
    }

    pub(crate) fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let s = self.bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(s);
        Ok(out)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }
    pub(crate) fn u16(&mut self) -> Result<u16> {
        let b = self.array()?;
        Ok(if self.little { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    }
    pub(crate) fn i16(&mut self) -> Result<i16> {
        let b = self.array()?;
        Ok(if self.little { i16::from_le_bytes(b) } else { i16::from_be_bytes(b) })
    }
    pub(crate) fn u32(&mut self) -> Result<u32> {
        let b = self.array()?;
        Ok(if self.little { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }
    pub(crate) fn i32(&mut self) -> Result<i32> {
        let b = self.array()?;
        Ok(if self.little { i32::from_le_bytes(b) } else { i32::from_be_bytes(b) })
    }
    pub(crate) fn u64(&mut self) -> Result<u64> {
        let b = self.array()?;
        Ok(if self.little { u64::from_le_bytes(b) } else { u64::from_be_bytes(b) })
    }
    pub(crate) fn i64(&mut self) -> Result<i64> {
        let b = self.array()?;
        Ok(if self.little { i64::from_le_bytes(b) } else { i64::from_be_bytes(b) })
    }
    pub(crate) fn f64(&mut self) -> Result<f64> {
        let b = self.array()?;
        Ok(if self.little { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) })
    }

    /// Reads a length field that is 4 bytes, or 8 bytes when `long`.
    pub(crate) fn len_field(&mut self, long: bool) -> Result<u64> {
        if long { self.u64() } else { Ok(u64::from(self.u32()?)) }
    }

    /// Returns `Err` unless `count * min_item_size` bytes remain. Used to
    /// validate element counts before allocating.
    pub(crate) fn check_count(&self, count: u64, min_item_size: u64) -> Result<()> {
        let need = count.saturating_mul(min_item_size);
        if need > self.remaining() as u64 {
            return Err(PsdError::UnexpectedEof { offset: self.pos, needed: usize::try_from(need - self.remaining() as u64).unwrap_or(usize::MAX) });
        }
        Ok(())
    }
}

/// Big-endian writing helpers on `Vec<u8>`.
pub(crate) trait WriteExt {
    fn put_u8(&mut self, v: u8);
    fn put_u16(&mut self, v: u16);
    fn put_i16(&mut self, v: i16);
    fn put_u32(&mut self, v: u32);
    fn put_i32(&mut self, v: i32);
    fn put_u64(&mut self, v: u64);
    fn put_i64(&mut self, v: i64);
    fn put_f64(&mut self, v: f64);
    fn put(&mut self, b: &[u8]);
    /// Writes a length field (4 or 8 bytes).
    fn put_len(&mut self, v: u64, long: bool) -> Result<()>;
    /// Reserves a length placeholder, returning its position.
    fn begin_len(&mut self, long: bool) -> usize;
    /// Back-patches a placeholder with the number of bytes written since.
    fn end_len(&mut self, at: usize, long: bool) -> Result<()>;
}

impl WriteExt for Vec<u8> {
    fn put_u8(&mut self, v: u8) {
        self.push(v);
    }
    fn put_u16(&mut self, v: u16) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put_i16(&mut self, v: i16) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put_u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put_i32(&mut self, v: i32) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put_u64(&mut self, v: u64) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put_i64(&mut self, v: i64) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put_f64(&mut self, v: f64) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn put(&mut self, b: &[u8]) {
        self.extend_from_slice(b);
    }
    fn put_len(&mut self, v: u64, long: bool) -> Result<()> {
        if long {
            self.put_u64(v);
        } else {
            let v = u32::try_from(v).map_err(|_| PsdError::LimitExceeded("section too large for 32-bit length (use PSB)"))?;
            self.put_u32(v);
        }
        Ok(())
    }
    fn begin_len(&mut self, long: bool) -> usize {
        let at = self.len();
        self.extend_from_slice(if long { &[0u8; 8] } else { &[0u8; 4] });
        at
    }
    fn end_len(&mut self, at: usize, long: bool) -> Result<()> {
        let hdr = if long { 8 } else { 4 };
        let n = (self.len() - at - hdr) as u64;
        if long {
            self[at..at + 8].copy_from_slice(&n.to_be_bytes());
        } else {
            let v = u32::try_from(n).map_err(|_| PsdError::LimitExceeded("section too large for 32-bit length (use PSB)"))?;
            self[at..at + 4].copy_from_slice(&v.to_be_bytes());
        }
        Ok(())
    }
}

/// Reads a Pascal string (length byte + bytes) whose total size is padded to
/// a multiple of `align`. Returns the raw name bytes.
pub(crate) fn read_pascal(r: &mut Reader<'_>, align: usize) -> Result<Vec<u8>> {
    let n = r.u8()? as usize;
    let name = r.bytes(n)?.to_vec();
    let total = 1 + n;
    let pad = (align - total % align) % align;
    r.skip(pad)?;
    Ok(name)
}

pub(crate) fn write_pascal(out: &mut Vec<u8>, name: &[u8], align: usize) {
    let n = name.len().min(255);
    out.put_u8(n as u8);
    out.put(&name[..n]);
    let total = 1 + n;
    let pad = (align - total % align) % align;
    out.extend(std::iter::repeat_n(0u8, pad));
}

/// Decodes legacy (Pascal) names: UTF-8 when valid, otherwise Latin-1.
pub(crate) fn decode_legacy_name(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => s.to_owned(),
        Err(_) => b.iter().map(|&c| c as char).collect(),
    }
}

/// Encodes a name for a legacy Pascal field: non-ASCII chars become `?`.
pub(crate) fn encode_legacy_name(s: &str) -> Vec<u8> {
    s.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c as u8 } else { b'?' }).take(255).collect()
}

/// Reads a Photoshop Unicode string: u32 code-unit count + UTF-16BE units.
pub(crate) fn read_unicode_units(r: &mut Reader<'_>) -> Result<Vec<u16>> {
    let n = r.u32()? as u64;
    r.check_count(n, 2)?;
    let mut v = Vec::with_capacity(n as usize);
    for _ in 0..n {
        v.push(r.u16()?);
    }
    Ok(v)
}

pub(crate) fn write_unicode_units(out: &mut Vec<u8>, units: &[u16]) {
    out.put_u32(units.len() as u32);
    for &u in units {
        out.put_u16(u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_reads_big_endian() {
        let data = [0x00, 0x01, 0xff, 0xfe, 0x00, 0x00, 0x00, 0x02];
        let mut r = Reader::new(&data);
        assert_eq!(r.u16().unwrap(), 1);
        assert_eq!(r.i16().unwrap(), -2);
        assert_eq!(r.u32().unwrap(), 2);
        assert!(r.is_empty());
        assert!(matches!(r.u8(), Err(PsdError::UnexpectedEof { .. })));
    }

    #[test]
    fn reader_eof_reports_needed() {
        let mut r = Reader::new(&[1, 2]);
        match r.u32() {
            Err(PsdError::UnexpectedEof { offset, needed }) => {
                assert_eq!(offset, 0);
                assert_eq!(needed, 2);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn check_count_rejects_huge() {
        let r = Reader::new(&[0; 8]);
        assert!(r.check_count(u64::MAX, 4).is_err());
        assert!(r.check_count(2, 4).is_ok());
        assert!(r.check_count(3, 4).is_err());
    }

    #[test]
    fn len_patch_roundtrip() {
        let mut v = Vec::new();
        let at = v.begin_len(false);
        v.put(&[1, 2, 3]);
        v.end_len(at, false).unwrap();
        assert_eq!(v, [0, 0, 0, 3, 1, 2, 3]);
        let mut v = Vec::new();
        let at = v.begin_len(true);
        v.put(&[9]);
        v.end_len(at, true).unwrap();
        assert_eq!(v, [0, 0, 0, 0, 0, 0, 0, 1, 9]);
    }

    #[test]
    fn pascal_padding() {
        for (name, align, total) in [(&b""[..], 2, 2), (&b"a"[..], 2, 2), (&b"ab"[..], 2, 4), (&b""[..], 4, 4), (&b"abc"[..], 4, 4), (&b"abcd"[..], 4, 8)] {
            let mut v = Vec::new();
            write_pascal(&mut v, name, align);
            assert_eq!(v.len(), total, "{name:?} align {align}");
            let mut r = Reader::new(&v);
            assert_eq!(read_pascal(&mut r, align).unwrap(), name);
            assert!(r.is_empty());
        }
    }

    #[test]
    fn legacy_names() {
        assert_eq!(decode_legacy_name(b"Layer 1"), "Layer 1");
        assert_eq!(decode_legacy_name(&[0x41, 0xe9]), "A\u{e9}");
        assert_eq!(encode_legacy_name("h\u{e9}llo"), b"h?llo");
    }

    #[test]
    fn unicode_units_roundtrip() {
        let mut v = Vec::new();
        write_unicode_units(&mut v, &[0x48, 0x69, 0]);
        let mut r = Reader::new(&v);
        assert_eq!(read_unicode_units(&mut r).unwrap(), vec![0x48, 0x69, 0]);
        let mut r = Reader::new(&[0xff, 0xff, 0xff, 0xff]);
        assert!(read_unicode_units(&mut r).is_err());
    }
}
