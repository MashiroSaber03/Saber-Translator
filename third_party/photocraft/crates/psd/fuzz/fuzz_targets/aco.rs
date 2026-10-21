//! Fuzz the `.aco` Photoshop Color Swatches parser: any input must return `Ok` or `Err`, never
//! panic; anything that parses must round-trip through `write` and re-parse with the same swatch
//! count, and no name may exceed the name-unit cap.
//!
//! `MAX_SWATCHES` is `u16::MAX` and the on-disk count is a `u16`, so the count is unfalsifiable by
//! construction; the assertion is kept as documentation of that invariant, not as a check that can
//! fail. The name cap, which is reachable, is the one that bites.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::aco;
fuzz_target!(|data: &[u8]| {
    let Ok(f) = aco::parse(data) else { return };
    assert!(f.swatches.len() <= aco::MAX_SWATCHES, "{} swatches exceeds MAX_SWATCHES", f.swatches.len());
    for s in &f.swatches {
        assert!(s.name.encode_utf16().count() <= aco::MAX_NAME_UNITS, "name longer than MAX_NAME_UNITS");
    }
    let out = aco::write(&f.swatches).expect("a parsed .aco must be writable");
    let back = aco::parse(&out).expect("a written .aco must re-parse");
    assert_eq!(back.swatches.len(), f.swatches.len(), "swatch count changed across a round trip");
    assert_eq!(aco::write(&back.swatches).expect("rewritable"), out, "a second round trip is not byte-stable");
});
