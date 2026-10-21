//! Minimal ZIP container: we write STORE entries (the payloads are already
//! zstd-compressed) and read STORE or DEFLATE entries (so a bundle that was
//! unzipped and re-zipped by another tool still loads). No ZIP64, no
//! encryption, no multi-disk archives.

use std::collections::HashMap;
use std::io::Read;

use crate::FormatError;

const LOCAL_SIG: u32 = 0x0403_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;

/// CRC-32 (IEEE), as stored in ZIP headers.
pub fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Streaming writer for STORE entries.
pub struct ZipWriter {
    out: Vec<u8>,
    central: Vec<u8>,
    count: u16,
}

impl Default for ZipWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ZipWriter {
    pub fn new() -> Self {
        ZipWriter { out: Vec::new(), central: Vec::new(), count: 0 }
    }

    pub fn add(&mut self, name: &str, data: &[u8]) -> Result<(), FormatError> {
        if self.count == u16::MAX {
            return Err(FormatError::Unsupported("more than 65535 zip entries (ZIP64 not supported)".into()));
        }
        let size = u32::try_from(data.len()).map_err(|_| FormatError::Unsupported("zip entry over 4 GiB".into()))?;
        let offset = u32::try_from(self.out.len()).map_err(|_| FormatError::Unsupported("zip over 4 GiB".into()))?;
        let crc = crc32(data);
        let name_b = name.as_bytes();
        // Local file header.
        let o = &mut self.out;
        o.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        o.extend_from_slice(&20u16.to_le_bytes()); // version needed
        o.extend_from_slice(&0x0800u16.to_le_bytes()); // flags: UTF-8 names
        o.extend_from_slice(&0u16.to_le_bytes()); // method: store
        o.extend_from_slice(&0u16.to_le_bytes()); // time
        o.extend_from_slice(&0x21u16.to_le_bytes()); // date: 1980-01-01
        o.extend_from_slice(&crc.to_le_bytes());
        o.extend_from_slice(&size.to_le_bytes());
        o.extend_from_slice(&size.to_le_bytes());
        o.extend_from_slice(&(name_b.len() as u16).to_le_bytes());
        o.extend_from_slice(&0u16.to_le_bytes());
        o.extend_from_slice(name_b);
        o.extend_from_slice(data);
        // Central directory entry.
        let c = &mut self.central;
        c.extend_from_slice(&CENTRAL_SIG.to_le_bytes());
        c.extend_from_slice(&20u16.to_le_bytes()); // made by
        c.extend_from_slice(&20u16.to_le_bytes()); // needed
        c.extend_from_slice(&0x0800u16.to_le_bytes());
        c.extend_from_slice(&0u16.to_le_bytes());
        c.extend_from_slice(&0u16.to_le_bytes());
        c.extend_from_slice(&0x21u16.to_le_bytes());
        c.extend_from_slice(&crc.to_le_bytes());
        c.extend_from_slice(&size.to_le_bytes());
        c.extend_from_slice(&size.to_le_bytes());
        c.extend_from_slice(&(name_b.len() as u16).to_le_bytes());
        c.extend_from_slice(&[0; 8]); // extra len, comment len, disk, internal attrs
        c.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        c.extend_from_slice(&offset.to_le_bytes());
        c.extend_from_slice(name_b);
        self.count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<u8>, FormatError> {
        let cd_offset = u32::try_from(self.out.len()).map_err(|_| FormatError::Unsupported("zip over 4 GiB".into()))?;
        let cd_size = self.central.len() as u32;
        self.out.extend_from_slice(&self.central);
        self.out.extend_from_slice(&EOCD_SIG.to_le_bytes());
        self.out.extend_from_slice(&[0; 4]); // disk numbers
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&cd_size.to_le_bytes());
        self.out.extend_from_slice(&cd_offset.to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes());
        Ok(self.out)
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    method: u16,
    crc: u32,
    compressed: usize,
    uncompressed: usize,
    data_offset: usize,
}

/// Read-only view of an archive in memory.
pub struct ZipReader<'a> {
    bytes: &'a [u8],
    /// All entries, in central-directory order. Treat as read-only: `find`
    /// looks names up through an index built from this list in `new`.
    pub entries: Vec<Entry>,
    /// Name -> index of the first entry with that name, so `find` is O(1)
    /// instead of a scan per lookup (a bundle reads one entry per tile).
    by_name: HashMap<String, usize>,
}

fn u16_at(b: &[u8], o: usize) -> Result<u16, FormatError> {
    o.checked_add(2).and_then(|end| b.get(o..end)).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| FormatError::corrupt("truncated zip"))
}
fn u32_at(b: &[u8], o: usize) -> Result<u32, FormatError> {
    o.checked_add(4).and_then(|end| b.get(o..end)).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| FormatError::corrupt("truncated zip"))
}

impl<'a> ZipReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, FormatError> {
        if bytes.len() < 22 {
            return Err(FormatError::corrupt("not a zip archive (too short)"));
        }
        // End of central directory: scan back over a possible comment.
        let min = bytes.len().saturating_sub(22 + 65535);
        let eocd = (min..=bytes.len() - 22)
            .rev()
            .find(|&i| u32_at(bytes, i).ok() == Some(EOCD_SIG))
            .ok_or_else(|| FormatError::corrupt("zip end-of-central-directory not found"))?;
        let count = u16_at(bytes, eocd + 10)? as usize;
        let cd_size = u32_at(bytes, eocd + 12)? as usize;
        let cd_off = u32_at(bytes, eocd + 16)? as usize;
        let cd_end = cd_off.checked_add(cd_size).filter(|end| *end <= eocd).ok_or_else(|| FormatError::corrupt("zip central directory out of range"))?;
        let mut entries = Vec::with_capacity(count.min(bytes.len() / 46));
        let mut p = cd_off;
        for _ in 0..count {
            let header_end = p.checked_add(46).filter(|end| *end <= cd_end).ok_or_else(|| FormatError::corrupt("truncated zip central directory entry"))?;
            if u32_at(bytes, p)? != CENTRAL_SIG {
                return Err(FormatError::corrupt("bad zip central directory entry"));
            }
            let flags = u16_at(bytes, p + 8)?;
            let method = u16_at(bytes, p + 10)?;
            let crc = u32_at(bytes, p + 16)?;
            let compressed = u32_at(bytes, p + 20)? as usize;
            let uncompressed = u32_at(bytes, p + 24)? as usize;
            let nlen = u16_at(bytes, p + 28)? as usize;
            let elen = u16_at(bytes, p + 30)? as usize;
            let clen = u16_at(bytes, p + 32)? as usize;
            let local = u32_at(bytes, p + 42)? as usize;
            let name_end = header_end.checked_add(nlen).filter(|end| *end <= cd_end).ok_or_else(|| FormatError::corrupt("zip entry name out of range"))?;
            let name = bytes.get(header_end..name_end).ok_or_else(|| FormatError::corrupt("truncated zip name"))?;
            let name = String::from_utf8_lossy(name).into_owned();
            if flags & 1 != 0 {
                return Err(FormatError::Unsupported("encrypted zip entries".into()));
            }
            // Bound each entry by the declared central-directory extent before advancing.
            let entry_end = name_end
                .checked_add(elen)
                .and_then(|end| end.checked_add(clen))
                .filter(|end| *end <= cd_end)
                .ok_or_else(|| FormatError::corrupt("zip central directory entry out of range"))?;
            // Local header gives the real data offset.
            if u32_at(bytes, local)? != LOCAL_SIG {
                return Err(FormatError::corrupt("bad zip local header"));
            }
            let local_name_len = local.checked_add(26).ok_or_else(|| FormatError::corrupt("zip local header offset overflow"))?;
            let local_extra_len = local.checked_add(28).ok_or_else(|| FormatError::corrupt("zip local header offset overflow"))?;
            let lnlen = u16_at(bytes, local_name_len)? as usize;
            let lelen = u16_at(bytes, local_extra_len)? as usize;
            let data_offset = local
                .checked_add(30)
                .and_then(|offset| offset.checked_add(lnlen))
                .and_then(|offset| offset.checked_add(lelen))
                .ok_or_else(|| FormatError::corrupt("zip data offset overflow"))?;
            if data_offset.checked_add(compressed).is_none_or(|end| end > bytes.len()) {
                return Err(FormatError::corrupt(format!("zip entry `{name}` out of range")));
            }
            entries.push(Entry { name, method, crc, compressed, uncompressed, data_offset });
            p = entry_end;
        }
        let mut by_name = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            // Duplicate names: the first entry wins, as with a front-to-back scan.
            by_name.entry(e.name.clone()).or_insert(i);
        }
        Ok(ZipReader { bytes, entries, by_name })
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.by_name.get(name).and_then(|&i| self.entries.get(i))
    }

    /// Entry payload (decompressed), at most `max` bytes, CRC-checked.
    pub fn read(&self, e: &Entry, max: usize) -> Result<Vec<u8>, FormatError> {
        if e.uncompressed > max {
            return Err(FormatError::LimitExceeded(format!("zip entry `{}` is {} bytes (max {max})", e.name, e.uncompressed)));
        }
        // `Entry` is public and cloneable: one from another archive must not index out of range.
        let raw = e
            .data_offset
            .checked_add(e.compressed)
            .and_then(|end| self.bytes.get(e.data_offset..end))
            .ok_or_else(|| FormatError::corrupt(format!("zip entry `{}` out of range", e.name)))?;
        let data = match e.method {
            0 => raw.to_vec(),
            8 => {
                // The uncompressed length comes from an untrusted ZIP header: a
                // small forged archive must not reserve gigabytes up front, so the
                // preallocation is capped (DEFLATE expands at most ~1032x).
                let mut out = Vec::with_capacity(e.uncompressed.min(raw.len().saturating_mul(1032)).min(64 << 20));
                flate2::read::DeflateDecoder::new(raw)
                    .take((max as u64).saturating_add(1))
                    .read_to_end(&mut out)
                    .map_err(|err| FormatError::corrupt(format!("zip entry `{}`: {err}", e.name)))?;
                out
            }
            m => {
                return Err(FormatError::Unsupported(format!("zip compression method {m} for `{}`", e.name)));
            }
        };
        if data.len() != e.uncompressed || data.len() > max {
            return Err(FormatError::corrupt(format!("zip entry `{}` has wrong size", e.name)));
        }
        if crc32(&data) != e.crc {
            return Err(FormatError::corrupt(format!("zip entry `{}` failed its CRC check", e.name)));
        }
        Ok(data)
    }

    pub fn read_by_name(&self, name: &str, max: usize) -> Result<Vec<u8>, FormatError> {
        let e = self.find(name).ok_or_else(|| FormatError::corrupt(format!("missing `{name}`")))?;
        self.read(e, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn roundtrip() {
        let mut w = ZipWriter::new();
        w.add("a.txt", b"hello").unwrap();
        w.add("dir/b.bin", &[0, 1, 2, 3]).unwrap();
        w.add("empty", &[]).unwrap();
        let bytes = w.finish().unwrap();
        let r = ZipReader::new(&bytes).unwrap();
        assert_eq!(r.entries.len(), 3);
        assert_eq!(r.read_by_name("a.txt", 100).unwrap(), b"hello");
        assert_eq!(r.read_by_name("dir/b.bin", 100).unwrap(), [0, 1, 2, 3]);
        assert!(r.read_by_name("empty", 0).unwrap().is_empty());
        assert!(matches!(r.read_by_name("a.txt", 3), Err(FormatError::LimitExceeded(_))));
    }

    #[test]
    fn crc_matches_bitwise_reference() {
        // Bit-at-a-time IEEE CRC-32 as the oracle, over lengths that cross
        // any SIMD / slicing block size.
        fn reference(data: &[u8]) -> u32 {
            let mut c = 0xFFFF_FFFFu32;
            for &b in data {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                }
            }
            !c
        }
        let data: Vec<u8> = (0..1000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for len in [0, 1, 7, 15, 16, 17, 63, 64, 65, 255, 256, 1000] {
            assert_eq!(crc32(&data[..len]), reference(&data[..len]), "len {len}");
        }
    }

    #[test]
    fn find_locates_every_entry_in_a_large_archive() {
        let mut w = ZipWriter::new();
        let n = 3000u32;
        for i in 0..n {
            w.add(&format!("tiles/{i:05}.zst"), &i.to_le_bytes()).unwrap();
        }
        let bytes = w.finish().unwrap();
        let r = ZipReader::new(&bytes).unwrap();
        assert_eq!(r.entries.len(), n as usize);
        for i in (0..n).rev() {
            let name = format!("tiles/{i:05}.zst");
            let e = r.find(&name).unwrap();
            assert_eq!(e.name, name);
            assert_eq!(r.read(e, 4).unwrap(), i.to_le_bytes());
        }
        assert!(r.find("tiles/missing.zst").is_none());
        assert!(r.find("").is_none());
    }

    #[test]
    fn duplicate_names_resolve_to_the_first_entry() {
        let mut w = ZipWriter::new();
        w.add("a", b"first").unwrap();
        w.add("b", b"other").unwrap();
        w.add("a", b"second").unwrap();
        let bytes = w.finish().unwrap();
        let r = ZipReader::new(&bytes).unwrap();
        assert_eq!(r.entries.len(), 3);
        assert_eq!(r.read_by_name("a", 100).unwrap(), b"first");
        assert_eq!(r.read_by_name("b", 100).unwrap(), b"other");
    }

    #[test]
    fn crc_mismatch_detected() {
        let mut w = ZipWriter::new();
        w.add("a", b"hello").unwrap();
        let mut bytes = w.finish().unwrap();
        let pos = bytes.windows(5).position(|x| x == b"hello").unwrap();
        bytes[pos] = b'j';
        let r = ZipReader::new(&bytes).unwrap();
        assert!(r.read_by_name("a", 100).is_err());
    }

    /// A valid DEFLATE-compressed bundle may be read with an effectively
    /// unlimited caller budget (e.g. corruption/round-trip checks). On 64-bit
    /// targets `usize::MAX as u64 + 1` overflows, so the decoder must saturate.
    #[test]
    fn deflated_entry_with_unbounded_limit_round_trips() {
        use std::io::Write as _;

        let content = b"DEFLATE-compressed PhotoCraft project content, with a valid CRC.";
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(content).unwrap();
        let compressed = encoder.finish().unwrap();

        // Construct a standards-conforming DEFLATE entry from the store-only
        // writer: keep its actual compressed bytes, but set the ZIP method,
        // original length and CRC in *both* the local and central headers.
        let mut writer = ZipWriter::new();
        writer.add("manifest.json", &compressed).unwrap();
        let mut bytes = writer.finish().unwrap();
        let eocd = bytes.len() - 22;
        let central = u32_at(&bytes, eocd + 16).unwrap() as usize;
        let crc = crc32(content).to_le_bytes();
        let length = (content.len() as u32).to_le_bytes();

        bytes[8..10].copy_from_slice(&8u16.to_le_bytes());
        bytes[14..18].copy_from_slice(&crc);
        bytes[22..26].copy_from_slice(&length);
        bytes[central + 10..central + 12].copy_from_slice(&8u16.to_le_bytes());
        bytes[central + 16..central + 20].copy_from_slice(&crc);
        bytes[central + 24..central + 28].copy_from_slice(&length);

        let archive = ZipReader::new(&bytes).unwrap();
        assert_eq!(archive.read_by_name("manifest.json", usize::MAX).unwrap(), content);
        assert_eq!(archive.read_by_name("manifest.json", content.len()).unwrap(), content);
        assert!(matches!(archive.read_by_name("manifest.json", content.len() - 1), Err(FormatError::LimitExceeded(_))));

        // A tiny archive with a forged 4 GiB size must be rejected without
        // trying to allocate its claimed output capacity in advance.
        let mut forged = bytes;
        forged[22..26].copy_from_slice(&u32::MAX.to_le_bytes());
        forged[central + 24..central + 28].copy_from_slice(&u32::MAX.to_le_bytes());
        let archive = ZipReader::new(&forged).unwrap();
        assert!(matches!(archive.read_by_name("manifest.json", usize::MAX), Err(FormatError::Corrupt(_))));
    }

    #[test]
    fn an_entry_from_another_archive_is_an_error_not_a_panic() {
        let mut big = ZipWriter::new();
        big.add("a", &[7; 4096]).unwrap();
        let big = big.finish().unwrap();
        let foreign = ZipReader::new(&big).unwrap().entries[0].clone();
        let mut small = ZipWriter::new();
        small.add("a", b"x").unwrap();
        let small = small.finish().unwrap();
        let r = ZipReader::new(&small).unwrap();
        assert!(matches!(r.read(&foreign, usize::MAX), Err(FormatError::Corrupt(_))));
    }

    #[test]
    fn offset_access_rejects_overflow() {
        assert!(u16_at(&[], usize::MAX).is_err());
        assert!(u32_at(&[], usize::MAX - 3).is_err());
    }
}
