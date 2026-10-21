//! Without the `heif` feature a HEIC file is still recognised, and opening it is a clear error
//! naming what this build lacks, never a panic or "unrecognized format".

#![cfg(not(feature = "heif"))]

use photocraft_io::import;

#[test]
fn heic_without_the_feature_says_it_is_not_in_this_build() {
    let header = b"\0\0\0\x18ftypheic\0\0\0\0mif1heic";
    for name in ["IMG_0001.HEIC", "photo.heif", "noext"] {
        let e = import(name, header).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("isn't included in this build"), "{name}: {e}");
    }
}
