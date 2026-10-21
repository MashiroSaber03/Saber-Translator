#![no_main]
//! Detect + decode arbitrary bytes, then re-encode what decoded (symmetry
//! path) — neither may panic.

use libfuzzer_sys::fuzz_target;
use photocraft_codecs::{decode_with, encode, DecodeOptions, EncodeOptions, Limits};

fuzz_target!(|data: &[u8]| {
    let opts = DecodeOptions {
        limits: Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 22, max_alloc: 256 << 20 },
        ..Default::default()
    };
    if let Ok(img) = decode_with(data, &opts)
        && let Some(f) = photocraft_codecs::detect(data)
    {
        let _ = encode(&img, f, &EncodeOptions::default());
    }
});
