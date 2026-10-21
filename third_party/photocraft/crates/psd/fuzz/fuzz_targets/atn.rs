//! Fuzz the `.atn` actions parser: any input must return `Ok` or `Err`, never panic, and
//! anything that parses must survive a write and parse back equal.
#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_psd::atn;

fuzz_target!(|data: &[u8]| {
    if let Ok(f) = atn::parse(data) {
        let bytes = atn::write(&f).expect("a parsed set is writable");
        let again = atn::parse(&bytes).expect("a written set parses");
        assert_eq!(again.actions, f.actions);
    }
});
