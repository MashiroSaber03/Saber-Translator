//! Fuzz the `.abr` brush parser: any input must return `Ok` or `Err`, never panic, and anything
//! that parses must keep its samples within the advertised bounds.
#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_psd::abr;

fuzz_target!(|data: &[u8]| {
    if let Ok(f) = abr::parse(data) {
        for s in &f.samples {
            assert_eq!(s.data.len(), s.width as usize * s.height as usize * usize::from(s.depth / 8));
        }
    }
});
