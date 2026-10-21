#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_codecs::{decode_as_with, DecodeOptions, Format, Limits};

// heic-rs is young and parses untrusted HEIF/HEVC: run with `-timeout=10` so a hang is reported,
// not just a panic. The codec turns a heic-rs panic into an error, so heic-rs is also called
// directly, where its panics still reach the fuzzer. Both orientation paths are covered.
fuzz_target!(|data: &[u8]| {
    let keep_orientation = data.first().is_some_and(|b| b & 1 == 1);
    let raw = heic_rs::DecodeOptions::default().with_max_pixels(Some(1 << 22)).with_transforms(!keep_orientation).with_alpha(true);
    let _ = heic_rs::decode(data, &raw);
    let limits = Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 22, max_alloc: 256 << 20 };
    let _ = decode_as_with(Format::Heif, data, &DecodeOptions { limits, keep_orientation });
});
