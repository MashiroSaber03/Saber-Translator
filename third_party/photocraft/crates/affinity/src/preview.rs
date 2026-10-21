//! The indexed PNG thumbnail Affinity stores next to the archive: Affinity's own render of the
//! document, at most 512 pixels on the long side in real files, never the document itself.

use crate::Error;
use crate::container::header;

pub const MAX_PREVIEW_BYTES: usize = 16 << 20;
pub const MAX_PREVIEW_DIMENSION: u32 = 4096;

/// Borrowed embedded thumbnail, at its own dimensions, never the native document dimensions.
#[derive(Debug)]
pub struct Preview<'a> {
    pub png: &'a [u8],
    pub width: u32,
    pub height: u32,
}

fn bytes_at(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], Error> {
    let end = offset.checked_add(length).ok_or(Error::Malformed("offset overflow"))?;
    bytes.get(offset..end).ok_or(Error::Malformed("truncated record"))
}

fn le32(bytes: &[u8], offset: usize) -> Result<u32, Error> {
    let b = bytes_at(bytes, offset, 4)?;
    Ok(u32::from_le_bytes(b.try_into().map_err(|_| Error::Malformed("integer"))?))
}

fn be32(bytes: &[u8], offset: usize) -> Result<u32, Error> {
    let b = bytes_at(bytes, offset, 4)?;
    Ok(u32::from_be_bytes(b.try_into().map_err(|_| Error::Malformed("PNG integer"))?))
}

/// Read only the indexed `Thmb` record (identical in container versions 8 to 12), checking every PNG chunk CRC. Compressed
/// pixels must still be validated by the consuming image decoder, with allocation limits.
pub fn preview(bytes: &[u8]) -> Result<Preview<'_>, Error> {
    let h = header(bytes)?;
    if h.thumbnail_offset == 0 {
        return Err(Error::Unsupported("no embedded preview"));
    }
    let offset = usize::try_from(h.thumbnail_offset).map_err(|_| Error::Malformed("thumbnail offset overflow"))?;
    if offset < 72 {
        return Err(Error::Malformed("thumbnail overlaps the file header"));
    }
    let record = bytes.get(offset..).ok_or(Error::Malformed("thumbnail offset outside file"))?;
    if bytes_at(record, 0, 8)? != b"\xff\xff\xff\xffThmb" {
        return Err(Error::Malformed("missing indexed thumbnail"));
    }
    if le32(record, 8)? != 1 || bytes_at(record, 28, 1)? != [1] {
        return Err(Error::Unsupported("thumbnail record version or encoding"));
    }
    let size = usize::try_from(le32(record, 24)?).map_err(|_| Error::Limit("encoded size"))?;
    if size > MAX_PREVIEW_BYTES {
        return Err(Error::Limit("encoded preview exceeds 16 MiB"));
    }
    if le32(record, 16)? != 29 || le32(record, 20)? != 0 || u64::from(le32(record, 12)?) != size as u64 + 13 {
        return Err(Error::Malformed("inconsistent thumbnail record lengths"));
    }
    let png = bytes_at(record, 29, size)?;
    if bytes_at(png, 0, 8)? != b"\x89PNG\r\n\x1a\n" || be32(png, 8)? != 13 || bytes_at(png, 12, 4)? != b"IHDR" {
        return Err(Error::Malformed("thumbnail is not a PNG"));
    }
    let width = be32(png, 16)?;
    let height = be32(png, 20)?;
    if width == 0 || height == 0 || width > MAX_PREVIEW_DIMENSION || height > MAX_PREVIEW_DIMENSION {
        return Err(Error::Limit("preview dimensions must be 1..4096"));
    }
    // Bound work by bytes, not a file-supplied chunk count, and require exactly one complete PNG.
    let mut pos = 8usize;
    let (mut palette, mut image_data, mut image_data_ended) = (false, false, false);
    loop {
        let length = usize::try_from(be32(png, pos)?).map_err(|_| Error::Limit("PNG chunk length"))?;
        let chunk_size = length.checked_add(12).ok_or(Error::Malformed("PNG chunk overflow"))?;
        let chunk = bytes_at(png, pos, chunk_size)?;
        let checksum_input = bytes_at(chunk, 4, chunk_size - 8)?;
        if crc32fast::hash(checksum_input) != be32(chunk, chunk_size - 4)? {
            return Err(Error::Malformed("PNG chunk checksum"));
        }
        let kind = bytes_at(chunk, 4, 4)?;
        if !kind.iter().all(u8::is_ascii_alphabetic) {
            return Err(Error::Malformed("PNG chunk type"));
        }
        match kind {
            b"IHDR" if pos == 8 => {}
            b"IHDR" => return Err(Error::Malformed("duplicate PNG header")),
            b"PLTE" => {
                if palette || image_data {
                    return Err(Error::Malformed("PNG palette order"));
                }
                palette = true;
            }
            b"IDAT" => {
                if image_data_ended {
                    return Err(Error::Malformed("non-contiguous PNG image data"));
                }
                image_data = true;
            }
            b"IEND" if !image_data => return Err(Error::Malformed("PNG has no image data")),
            b"IEND" => {}
            _ if kind.first().is_some_and(u8::is_ascii_uppercase) => {
                return Err(Error::Unsupported("unknown critical PNG chunk"));
            }
            _ => {}
        }
        if image_data && kind != b"IDAT" {
            image_data_ended = true;
        }
        // Native sample previews are plain, single-frame PNGs. Pixel allocation limits do not
        // necessarily bound aggregate compressed metadata; leave these layouts unsupported.
        if matches!(kind, b"zTXt" | b"iTXt" | b"iCCP") {
            return Err(Error::Unsupported("compressed PNG metadata in the preview"));
        }
        if matches!(kind, b"acTL" | b"fcTL" | b"fdAT") {
            return Err(Error::Unsupported("animated PNG preview"));
        }
        pos = pos.checked_add(chunk_size).ok_or(Error::Malformed("PNG offset overflow"))?;
        if kind == b"IEND" {
            if length != 0 || pos != png.len() {
                return Err(Error::Malformed("PNG end or trailing data"));
            }
            break;
        }
    }
    Ok(Preview { png, width, height })
}
