#![no_main]

use libfuzzer_sys::fuzz_target;

// The lossless-JPEG decoder on its own: never panic, never allocate past the cap.
fuzz_target!(|data: &[u8]| {
    let _ = photocraft_raw::decode_lossless_jpeg(data, 1 << 22);
});
