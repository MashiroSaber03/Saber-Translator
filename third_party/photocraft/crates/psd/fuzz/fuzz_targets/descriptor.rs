//! Fuzz the ActionDescriptor parser; anything that parses must be byte
//! stable after one normalization pass (e.g. `bool` bytes other than 0/1).
#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_psd::descriptor::Descriptor;

fuzz_target!(|data: &[u8]| {
    if let Ok(d) = Descriptor::from_bytes(data) {
        let b = d.to_bytes();
        assert_eq!(Descriptor::from_bytes(&b).expect("re-parse").to_bytes(), b);
    }
});
