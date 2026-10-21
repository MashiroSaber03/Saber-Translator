//! HDR toning settings: the Color Mode Data of 32-bit-per-channel documents.
//!
//! Adobe's specification only documents the Color Mode Data section for
//! indexed and duotone images, but every 32-bit file Photoshop saves carries
//! an `hdrt` record there (the defaults of Image › Mode › 8/16 Bits/Channel ›
//! HDR Toning), and current versions append an `hdra` record. Photoshop 2026
//! refuses to open a 32-bit file whose Color Mode Data is empty or not in this
//! form ("the open options are incorrect", #291), so writers must emit it.
//!
//! The layout below was established by observing Photoshop-saved files (the
//! `photocraft-corpus` oracles); fields whose meaning is not known are written
//! with the values Photoshop writes for a new document.

use crate::io::WriteExt;

/// Key of the HDR toning record.
pub const TONING_KEY: &[u8; 4] = b"hdrt";
/// Key of the record current Photoshop versions append after the toning record.
pub const EXTRA_KEY: &[u8; 4] = b"hdra";

/// `true` if `data` (a file's Color Mode Data) starts with an HDR toning record.
pub fn is_toning_data(data: &[u8]) -> bool {
    data.starts_with(TONING_KEY)
}

/// The Color Mode Data Photoshop writes for a new 32-bit document: a version 3
/// `hdrt` record (method 2, preset "Linear", an identity toning curve) followed
/// by a version 6 `hdra` record. 110 bytes.
pub fn default_toning_data() -> Vec<u8> {
    let mut out = Vec::with_capacity(110);
    // `hdrt`, version 3.
    out.put(TONING_KEY);
    out.put_u32(3);
    out.put(&0.23f32.to_be_bytes());
    out.put_u32(2); // toning method
    // Preset name: Unicode string (UTF-16 code units including the terminator).
    let name: Vec<u16> = "Linear".encode_utf16().chain(std::iter::once(0)).collect();
    out.put_u32(name.len() as u32);
    for u in name {
        out.put_u16(u);
    }
    // Toning curve: two points, (0, 0) and (255, 255), both corners.
    out.put_u16(2);
    out.put_u16(2);
    for (input, output) in [(0u16, 0u16), (255, 255)] {
        out.put_u16(output);
        out.put_u16(input);
    }
    out.put(&[1, 1]);
    out.put(&[0; 8]);
    out.put(&16.0f32.to_be_bytes());
    out.put_u32(1);
    out.put_u32(0);
    out.put(&1.0f32.to_be_bytes());
    // `hdra`, version 6.
    out.put(EXTRA_KEY);
    out.put_u32(6);
    out.put_u32(0);
    out.put(&20.0f32.to_be_bytes());
    out.put(&30.0f32.to_be_bytes());
    out.put_u32(0);
    out.put_u32(0);
    out.put(&1.0f32.to_be_bytes());
    out.put(&[0; 6]);
    out
}

/// The Color Mode Data a writer should emit for a file of `depth` bits per
/// channel when it has nothing of its own to keep: the default HDR toning
/// records for 32-bit files, nothing otherwise (indexed and duotone data come
/// from the document).
pub fn color_mode_data_for_depth(depth: u16) -> Vec<u8> {
    if depth == 32 { default_toning_data() } else { Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_record_layout() {
        let d = default_toning_data();
        assert_eq!(d.len(), 110);
        assert!(is_toning_data(&d));
        assert_eq!(&d[72..76], EXTRA_KEY);
        // Version fields.
        assert_eq!(&d[4..8], &3u32.to_be_bytes());
        assert_eq!(&d[76..80], &6u32.to_be_bytes());
        assert!(color_mode_data_for_depth(8).is_empty());
        assert!(color_mode_data_for_depth(16).is_empty());
        assert_eq!(color_mode_data_for_depth(32), d);
    }
}
