//! Finding the camera-rendered JPEG preview embedded in a raw file.

use crate::tiff::{Tiff, tag};

/// An embedded baseline-JPEG preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preview<'a> {
    /// The complete JPEG stream.
    pub jpeg: &'a [u8],
    pub width: u32,
    pub height: u32,
}

/// Dimensions of a JPEG whose frame is DCT-coded (baseline / extended /
/// progressive), or `None` (lossless JPEG raw data, or not a JPEG).
fn dct_jpeg_size(b: &[u8]) -> Option<(u32, u32)> {
    if b.get(0..3) != Some(&[0xFF, 0xD8, 0xFF]) {
        return None;
    }
    let mut pos = 2usize;
    for _ in 0..256 {
        if *b.get(pos)? != 0xFF {
            return None;
        }
        let m = *b.get(pos + 1)?;
        if m == 0xFF {
            pos += 1;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([*b.get(pos + 2)?, *b.get(pos + 3)?]));
        if matches!(m, 0xC0..=0xC2) {
            let h = u16::from_be_bytes([*b.get(pos + 5)?, *b.get(pos + 6)?]);
            let w = u16::from_be_bytes([*b.get(pos + 7)?, *b.get(pos + 8)?]);
            return (w > 0 && h > 0).then_some((u32::from(w), u32::from(h)));
        }
        if matches!(m, 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF | 0xDA | 0xD9) || len < 2 {
            return None;
        }
        pos = pos.checked_add(2 + len)?;
    }
    None
}

pub(crate) fn candidate(data: &[u8], offset: usize, len: usize) -> Option<Preview<'_>> {
    let jpeg = data.get(offset..offset.checked_add(len)?)?;
    let (width, height) = dct_jpeg_size(jpeg)?;
    Some(Preview { jpeg, width, height })
}

/// The largest embedded JPEG preview of a raw file (TIFF-based raws, RAF and CR3).
pub fn embedded_preview(bytes: &[u8]) -> Option<Preview<'_>> {
    embedded_previews(bytes).into_iter().max_by_key(|p| u64::from(p.width) * u64::from(p.height))
}

/// Every embedded DCT-JPEG preview of a raw file, each stream once (cameras often store a
/// full-size and a smaller rendering of the same picture).
pub fn embedded_previews(bytes: &[u8]) -> Vec<Preview<'_>> {
    let mut found: Vec<Preview> = Vec::new();
    if bytes.get(4..12) == Some(b"ftypcrx ") {
        found = crate::cr3::previews(bytes);
    } else if bytes.starts_with(crate::raf::MAGIC) {
        // RAF header: big-endian JPEG offset and length at bytes 84 and 88.
        if let Some((off, len)) = crate::raf::preview_range(bytes) {
            found.extend(candidate(bytes, off, len));
        }
    } else if let Some(t) = Tiff::new(bytes) {
        // Olympus ORF: the preview is referenced from the maker note.
        if matches!(bytes.get(0..4), Some(b"IIRO" | b"IIRS" | b"MMOR"))
            && let Some((off, len)) = crate::orf::preview_range(&t)
        {
            found.extend(candidate(bytes, off, len));
        }
        for ifd in t.all_ifds() {
            if let (Some(off), Some(len)) = (t.tag_uint(&ifd, tag::JPEG_INTERCHANGE_FORMAT), t.tag_uint(&ifd, tag::JPEG_INTERCHANGE_FORMAT_LENGTH)) {
                found.extend(candidate(bytes, off as usize, len as usize));
            }
            if matches!(t.tag_uint(&ifd, tag::COMPRESSION), Some(6 | 7)) {
                let offs = t.tag_uints(&ifd, tag::STRIP_OFFSETS);
                let lens = t.tag_uints(&ifd, tag::STRIP_BYTE_COUNTS);
                if let ([off], [len]) = (offs.as_slice(), lens.as_slice()) {
                    found.extend(candidate(bytes, *off as usize, *len as usize));
                }
            }
            // Panasonic RW2 JpgFromRaw (0x002E) holds a complete JPEG.
            if let Some(e) = ifd.get(0x002E).filter(|e| e.typ == 7)
                && let Some(raw) = t.raw(e)
                && let Some((width, height)) = dct_jpeg_size(raw)
            {
                found.push(Preview { jpeg: raw, width, height });
            }
        }
    }
    let mut seen = Vec::new();
    found.retain(|p| {
        let key = (p.jpeg.as_ptr() as usize, p.jpeg.len());
        let new = !seen.contains(&key);
        seen.push(key);
        new
    });
    found
}

/// The colour space a raw file states for its embedded previews, which carry no ICC profile of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewColorSpace {
    Srgb,
    AdobeRgb,
}

/// The colour space the raw's EXIF gives for the camera's JPEG rendering, for a preview that declares none itself:
/// ColorSpace (0xA001) 1 = sRGB, or Uncalibrated (0xFFFF) with the Interoperability IFD's InteropIndex "R03", the
/// DCF option file for Adobe RGB (as a Sony body set to Adobe RGB writes; Exif and DCF specifications). `None` when
/// the file doesn't say, and for a DNG, whose EXIF describes the original capture while its previews are the
/// converter's. A preview's own ICC profile takes precedence over this.
pub fn embedded_preview_color_space(bytes: &[u8]) -> Option<PreviewColorSpace> {
    let t = Tiff::new(bytes)?;
    let ifd0 = t.ifd_at(t.first_ifd, 0)?;
    if ifd0.has(tag::DNG_VERSION) {
        return None;
    }
    let exif = t.ifd_at(t.tag_uint(&ifd0, tag::EXIF_IFD)? as usize, 0)?;
    match t.tag_uint(&exif, tag::COLOR_SPACE)? {
        1 => Some(PreviewColorSpace::Srgb),
        0xFFFF => {
            let interop = t.ifd_at(t.tag_uint(&exif, tag::INTEROP_IFD)? as usize, 0)?;
            t.raw(interop.get(tag::INTEROP_INDEX)?)?.starts_with(b"R03").then_some(PreviewColorSpace::AdobeRgb)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgen::{TiffBuilder, Val};

    /// A raw-like TIFF whose EXIF IFD has `color_space` and, when given, an Interoperability IFD with `index`.
    fn raw_with(color_space: Option<u16>, index: Option<&str>, dng: bool) -> Vec<u8> {
        let mut t = TiffBuilder::default();
        let mut exif = Vec::new();
        if let Some(cs) = color_space {
            exif.push((tag::COLOR_SPACE, Val::Short(vec![cs])));
        }
        if let Some(index) = index {
            let interop = t.ifd(vec![(tag::INTEROP_INDEX, Val::Ascii(index.into()))]);
            exif.push((tag::INTEROP_IFD, Val::Ifds(vec![interop])));
        }
        let exif = t.ifd(exif);
        let mut ifd0 = vec![(tag::MAKE, Val::Ascii("SONY".into())), (tag::EXIF_IFD, Val::Ifds(vec![exif]))];
        if dng {
            ifd0.push((tag::DNG_VERSION, Val::Byte(vec![1, 6, 0, 0])));
        }
        t.chain = vec![t.ifd(ifd0)];
        t.build()
    }

    #[test]
    fn exif_colour_space_of_the_previews() {
        assert_eq!(embedded_preview_color_space(&raw_with(Some(1), Some("R98"), false)), Some(PreviewColorSpace::Srgb));
        assert_eq!(embedded_preview_color_space(&raw_with(Some(0xFFFF), Some("R03"), false)), Some(PreviewColorSpace::AdobeRgb));
        for (cs, index) in [(Some(0xFFFF), Some("R98")), (Some(0xFFFF), None), (Some(2), Some("R03")), (None, Some("R03"))] {
            assert_eq!(embedded_preview_color_space(&raw_with(cs, index, false)), None, "{cs:?} {index:?}");
        }
        // a DNG's previews are the converter's, whatever the capture was
        assert_eq!(embedded_preview_color_space(&raw_with(Some(0xFFFF), Some("R03"), true)), None);
        for junk in [&b""[..], b"II*\0", b"II*\0\x08\0\0\0\xff\xff", b"not a tiff at all"] {
            assert_eq!(embedded_preview_color_space(junk), None);
        }
    }
}
