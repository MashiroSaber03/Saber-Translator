//! Camera raw files are TIFF-structured (TIFF/EP, DNG, CR2, NEF, ARW…) but
//! hold sensor data this crate does not develop. Opening one must say so,
//! not fail with a TIFF decoder detail or silently return the embedded
//! preview. The files here are tiny synthetic TIFFs with the structural
//! markers from the public TIFF/EP and DNG specifications.

use photocraft_codecs::*;

const SHORT: u16 = 3;
const LONG: u16 = 4;
const BYTE4: u16 = 1; // BYTE, written with count 4 (the value's 4 bytes inline)
const CFA: u32 = 32803; // TIFF/EP PhotometricInterpretation: colour filter array
const LINEAR_RAW: u32 = 34892; // DNG PhotometricInterpretation: LinearRaw

/// A little-endian TIFF with the given IFDs (each a list of
/// `(tag, type, value)` entries with count 1), chained in order. Every IFD
/// gets 2×2 8-bit strip data. `header8` replaces bytes 8..12 (CR2 puts
/// "CR" + version there and starts IFD0 at 16).
fn tiff(ifds: &[Vec<(u16, u16, u32)>], header8: Option<[u8; 4]>) -> Vec<u8> {
    let mut b = b"II*\0".to_vec();
    let first = if header8.is_some() { 16u32 } else { 8 };
    b.extend_from_slice(&first.to_le_bytes());
    if let Some(h) = header8 {
        b.extend_from_slice(&h);
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    // Layout: IFDs back to back, then one shared 12-byte pixel strip.
    let sizes: Vec<u32> = ifds.iter().map(|e| 2 + 12 * (e.len() as u32 + 1) + 4).collect();
    let strip = first + sizes.iter().sum::<u32>();
    let mut at = first;
    for (i, entries) in ifds.iter().enumerate() {
        let mut entries = entries.clone();
        entries.push((273, LONG, strip)); // StripOffsets
        entries.sort_by_key(|e| e.0);
        b.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, ty, v) in entries {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            let count: u32 = if ty == BYTE4 { 4 } else { 1 };
            b.extend_from_slice(&count.to_le_bytes());
            if ty == SHORT {
                b.extend_from_slice(&(v as u16).to_le_bytes());
                b.extend_from_slice(&[0, 0]);
            } else {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        at += sizes[i];
        let next = if i + 1 < ifds.len() { at } else { 0 };
        b.extend_from_slice(&next.to_le_bytes());
    }
    b.extend_from_slice(&[128; 12]);
    b
}

/// A 2×2 image IFD with the given photometric interpretation.
fn image_ifd(photometric: u32, samples: u32) -> Vec<(u16, u16, u32)> {
    vec![
        (256, SHORT, 2),           // ImageWidth
        (257, SHORT, 2),           // ImageLength
        (258, SHORT, 8),           // BitsPerSample
        (259, SHORT, 1),           // Compression: none
        (262, SHORT, photometric), // PhotometricInterpretation
        (277, SHORT, samples),     // SamplesPerPixel
        (278, SHORT, 2),           // RowsPerStrip
        (279, LONG, 4 * samples),  // StripByteCounts
    ]
}

fn assert_camera_raw(bytes: &[u8]) {
    assert_eq!(detect(bytes), Some(Format::Tiff));
    match decode(bytes) {
        Err(CodecError::Unsupported { reason, .. }) => {
            assert!(reason.contains("camera raw"), "unexpected reason: {reason}");
        }
        Err(e) => panic!("expected a camera raw error, got: {e}"),
        Ok(img) => panic!("expected a camera raw error, decoded {:?}", img.dimensions()),
    }
}

#[test]
fn cfa_image_is_reported_as_camera_raw() {
    // TIFF/EP-style raw (NEF/ARW-like): the main image is a CFA.
    assert_camera_raw(&tiff(&[image_ifd(CFA, 1)], None));
}

#[test]
fn linear_raw_image_is_reported_as_camera_raw() {
    assert_camera_raw(&tiff(&[image_ifd(LINEAR_RAW, 3)], None));
}

#[test]
fn raw_behind_a_preview_is_reported_as_camera_raw() {
    // IFD0 is a small RGB preview, the CFA raw is a SubIFD (tag 330): decoding
    // must not quietly return the preview as the photo.
    let mut ifd0 = image_ifd(2, 3);
    let ifd0_size = 2 + 12 * (ifd0.len() as u32 + 2) + 4;
    ifd0.push((330, LONG, 8 + ifd0_size)); // SubIFDs → the next IFD
    let mut b = tiff(&[ifd0, image_ifd(CFA, 1)], None);
    // Unchain the raw IFD from IFD0's next-IFD pointer so it is reachable only
    // as a SubIFD.
    let next_at = 8 + ifd0_size as usize - 4;
    b[next_at..next_at + 4].copy_from_slice(&0u32.to_le_bytes());
    assert_camera_raw(&b);
}

#[test]
fn dng_is_reported_as_camera_raw() {
    let mut ifd0 = image_ifd(2, 3);
    ifd0.push((50706, BYTE4, 0x0000_0401)); // DNGVersion 1.4.0.0
    assert_camera_raw(&tiff(&[ifd0], None));
}

#[test]
fn cr2_is_reported_as_camera_raw() {
    // CR2: TIFF header + "CR", major 2, minor 0 at byte 8; IFD0 at 16.
    assert_camera_raw(&tiff(&[image_ifd(6, 3)], Some(*b"CR\x02\0")));
}

#[test]
fn ordinary_tiffs_still_decode() {
    let rgb = decode(&tiff(&[image_ifd(2, 3)], None)).unwrap();
    assert_eq!(rgb.dimensions(), (2, 2));
    let gray = decode(&tiff(&[image_ifd(1, 1)], None)).unwrap();
    assert_eq!(gray.dimensions(), (2, 2));
}

#[test]
fn truncated_raw_markers_do_not_panic() {
    let b = tiff(&[image_ifd(CFA, 1)], Some(*b"CR\x02\0"));
    for n in 0..b.len() {
        let _ = decode(&b[..n]);
    }
}
