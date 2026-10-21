#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_raw::{Demosaic, DevelopOptions, Limits, decode, develop_sensor, embedded_preview, identify};

// Any bytes: identify, find the preview, decode and develop (fast demosaic)
// under tight limits. Must never panic or exceed the limits.
fuzz_target!(|data: &[u8]| {
    let limits = Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 20, max_alloc: 64 << 20 };
    let _ = identify(data);
    let _ = embedded_preview(data);
    if let Ok(s) = decode(data, &limits) {
        let _ = develop_sensor(&s, &DevelopOptions { demosaic: Demosaic::Bilinear, limits, ..Default::default() });
    }
});
