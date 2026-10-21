#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The complete import: native document mapping, or the preview fallback.
    if photocraft_affinity::is_affinity(data) {
        let _ = photocraft_io::import("fuzz.af", data);
    }
});
