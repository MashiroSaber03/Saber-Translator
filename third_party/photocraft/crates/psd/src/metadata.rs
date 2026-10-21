//! `shmd` (metadata setting) layer block: a list of keyed sub-blocks, such as `cmls` (layer comp
//! state, a versioned descriptor), `cust` and `mlst`.
//!
//! Layout (Adobe PSD spec, "Metadata setting"): u32 count, then per item the signature `8BIM`,
//! a 4-byte key, a copy-on-sheet-duplication byte, 3 padding bytes, a u32 length and the data.
//! Photoshop pads item data to an even length (the padding is part of the stored length), so
//! parsing and writing are exact: `write(parse(b)) == b`.

use crate::error::{PsdError, Result};
use crate::io::{Reader, WriteExt};

/// One `shmd` item.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetadataItem {
    /// Usually `8BIM`.
    pub signature: [u8; 4],
    /// Item key, e.g. `cmls`.
    pub key: [u8; 4],
    /// Copy on sheet duplication.
    pub copy: bool,
    /// The 3 bytes after the copy flag (zero in Photoshop files), kept for exactness.
    pub reserved: [u8; 3],
    /// Data including any padding counted in the stored length.
    pub data: Vec<u8>,
}

impl MetadataItem {
    /// A new item; `data` is zero-padded to an even length as Photoshop writes it.
    pub fn new(key: [u8; 4], mut data: Vec<u8>) -> Self {
        if data.len() % 2 == 1 {
            data.push(0);
        }
        MetadataItem { signature: *b"8BIM", key, copy: false, reserved: [0; 3], data }
    }
}

/// Parses `shmd` block data.
pub fn parse_shmd(data: &[u8]) -> Result<Vec<MetadataItem>> {
    let mut r = Reader::new(data);
    let count = r.u32()?;
    r.check_count(u64::from(count), 16)?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let signature = r.array()?;
        let key = r.array()?;
        let copy = r.u8()? != 0;
        let reserved = r.array()?;
        let len = r.u32()?;
        let data = r.bytes_u64(u64::from(len))?.to_vec();
        out.push(MetadataItem { signature, key, copy, reserved, data });
    }
    if !r.is_empty() {
        return Err(PsdError::invalid("trailing bytes after metadata items"));
    }
    Ok(out)
}

/// Serializes `shmd` block data.
pub fn write_shmd(items: &[MetadataItem]) -> Vec<u8> {
    let mut out = Vec::new();
    out.put_u32(items.len() as u32);
    for it in items {
        out.put(&it.signature);
        out.put(&it.key);
        out.push(u8::from(it.copy));
        out.put(&it.reserved);
        out.put_u32(it.data.len() as u32);
        out.put(&it.data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_exact() {
        let items = vec![MetadataItem::new(*b"cmls", vec![1, 2, 3]), MetadataItem { copy: true, ..MetadataItem::new(*b"cust", vec![9; 4]) }];
        assert_eq!(items[0].data.len(), 4);
        let bytes = write_shmd(&items);
        assert_eq!(parse_shmd(&bytes).unwrap(), items);
        assert_eq!(write_shmd(&parse_shmd(&bytes).unwrap()), bytes);
    }

    #[test]
    fn malformed_is_an_error() {
        assert!(parse_shmd(&[]).is_err());
        assert!(parse_shmd(&[0, 0, 0, 9]).is_err());
        let mut b = write_shmd(&[MetadataItem::new(*b"cmls", vec![0; 2])]);
        b.push(0);
        assert!(parse_shmd(&b).is_err());
        b.truncate(b.len() - 3);
        assert!(parse_shmd(&b).is_err());
    }
}
