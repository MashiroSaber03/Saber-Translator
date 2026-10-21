#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_codecs::{decode_as_with, DecodeOptions, Format, Limits};

fuzz_target!(|data: &[u8]| {
    let opts = DecodeOptions {
        limits: Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 22, max_alloc: 256 << 20 },
        ..Default::default()
    };
    let _ = decode_as_with(Format::Png, data, &opts);
});
