//! Fuzz the `.ase` Adobe Swatch Exchange parser: any input must return `Ok` or `Err`, never panic;
//! anything that parses must hold only finite colour components, stay within the swatch cap and
//! round-trip through `write` and re-parse with the same number of entries.
//!
//! A colour outside any group is a legitimate `AseEntry::Swatch`, and a group left open at the
//! end is closed by the parser, so entry shape is checked only through the round trip below.
//! The cap assertion is an invariant, not a fast check: no 4 KB input can carry 100 001 blocks.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::ase::{self, AseColor, AseEntry};
fuzz_target!(|data: &[u8]| {
    let Ok(f) = ase::parse(data) else { return };
    assert!(f.swatch_count() <= ase::MAX_SWATCHES, "{} swatches exceeds MAX_SWATCHES", f.swatch_count());
    for e in &f.entries {
        let swatches = match e {
            AseEntry::Swatch(s) => std::slice::from_ref(s),
            AseEntry::Group { swatches, .. } => swatches.as_slice(),
        };
        for s in swatches {
            assert!(s.name.encode_utf16().count() <= ase::MAX_NAME_UNITS, "name longer than MAX_NAME_UNITS");
            // Rgb/Lab carry three components, Cmyk four, Gray one.
            let check = |c: &[f32]| {
                for v in c {
                    assert!(v.is_finite(), "colour component {v} is not finite");
                }
            };
            match &s.color {
                AseColor::Rgb(v) | AseColor::Lab(v) => check(v),
                AseColor::Cmyk(v) => check(v),
                AseColor::Gray(v) => check(std::slice::from_ref(v)),
                AseColor::Other { .. } => {}
            }
        }
    }
    let out = ase::write(&f).expect("a parsed .ase must be writable");
    let back = ase::parse(&out).expect("a written .ase must re-parse");
    assert_eq!(back.swatch_count(), f.swatch_count(), "swatch count changed across a round trip");
    assert_eq!(ase::write(&back).expect("rewritable"), out, "a second round trip is not byte-stable");
});
