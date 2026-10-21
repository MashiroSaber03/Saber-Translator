//! Fuzz the `.grd` gradient parser: any input must return `Ok` or `Err`, never panic, and every
//! number it hands back must be finite and in range.
//!
//! The cap assertions below are invariants, not fast checks: at libFuzzer default `-max_len`
//! of 4096 no input can carry 10 001 gradients or 1001 stops, so they only fire on long runs with
//! a raised limit. The inputs that do reach those caps are in `crates/psd/tests/swatch_malformed.rs`.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::grd::{self, GrdColor};
fuzz_target!(|data: &[u8]| {
    let Ok(g) = grd::parse(data) else { return };
    assert!(g.len() <= grd::MAX_GRADIENTS, "{} gradients exceeds MAX_GRADIENTS", g.len());
    for grad in &g {
        assert!(grad.stops.len() <= grd::MAX_STOPS, "{} stops exceeds MAX_STOPS", grad.stops.len());
        assert!(grad.opacity.len() <= grd::MAX_STOPS, "{} transparency stops exceeds MAX_STOPS", grad.opacity.len());
        for &(l, o, m) in &grad.opacity {
            assert!(l.is_finite() && (0.0..=1.0).contains(&l), "transparency location {l} out of range");
            assert!((0.0..=1.0).contains(&o), "opacity {o} outside 0..=1");
            assert!(m.is_finite() && (0.0..=1.0).contains(&m), "transparency midpoint {m} out of range");
        }
        for s in &grad.stops {
            assert!(s.location.is_finite() && (0.0..=1.0).contains(&s.location), "stop location {} out of range", s.location);
            assert!(s.midpoint.is_finite() && (0.0..=1.0).contains(&s.midpoint), "stop midpoint {} out of range", s.midpoint);
            // Each model has its own documented range, not just "finite": the parser clamps RGB,
            // CMYK and Gray to 0..=1, Hue to 0..=360, L* to 0..=100 and a*/b* to the CIELAB gamut.
            let unit = |c: &[f32]| {
                for v in c {
                    assert!(v.is_finite() && (0.0..=1.0).contains(v), "colour component {v} outside 0..=1");
                }
            };
            match s.color {
                GrdColor::Rgb(v) => unit(&v),
                GrdColor::Cmyk(v) => unit(&v),
                GrdColor::Hsb([h, sat, br]) => {
                    assert!(h.is_finite() && (0.0..=360.0).contains(&h), "hue {h} outside 0..=360");
                    unit(&[sat, br]);
                }
                GrdColor::Lab([l, a, b]) => {
                    assert!(l.is_finite() && (0.0..=100.0).contains(&l), "L* {l} outside 0..=100");
                    assert!(a.is_finite() && (-128.0..=127.0).contains(&a), "a* {a} outside the CIELAB gamut");
                    assert!(b.is_finite() && (-128.0..=127.0).contains(&b), "b* {b} outside the CIELAB gamut");
                }
                GrdColor::Gray(v) => unit(&[v]),
                GrdColor::Foreground | GrdColor::Background => {}
            }
        }
    }
    let out = grd::write(&g);
    let back = grd::parse(&out).expect("a written .grd must re-parse");
    assert_eq!(back.len(), g.len(), "gradient count changed across a round trip");
    assert_eq!(grd::write(&back), out, "a second round trip is not byte-stable");
});
