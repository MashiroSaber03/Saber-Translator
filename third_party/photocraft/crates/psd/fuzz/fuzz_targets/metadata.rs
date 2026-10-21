//! Fuzz the `shmd` (metadata setting) parser: any input must return `Ok` or `Err`, never panic.
//!
//! The `copy` field is a bool, so a non-canonical copy byte (anything but 0 or 1) normalises to 1
//! on write; exact byte equality therefore cannot hold for arbitrary input, only for canonical
//! files. The invariant that does hold, and that catches reader/writer drift, is that a
//! parsed-then-written block re-parses to the *same items* (semantic equality, not just byte
//! stability -- byte stability alone would also pass if the writer silently dropped a field).
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::metadata::{parse_shmd, write_shmd};
fuzz_target!(|data: &[u8]| {
    let Ok(items) = parse_shmd(data) else { return };
    let out = write_shmd(&items);
    let back = parse_shmd(&out).expect("a written shmd block must re-parse");
    assert_eq!(back, items, "a written shmd block does not re-parse to the same items");
    assert_eq!(write_shmd(&back), out, "shmd is not byte-stable across a round trip");
});
