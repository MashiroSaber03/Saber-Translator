//! `.grd`, `.ase` and `.aco` hostile-input handling.
//!
//! These three parsers are standalone entry points that `PsdFile::from_bytes` never reaches
//! (nothing in `file.rs` calls them), so the `parse.rs` fuzz target cannot exercise them. They are
//! also user-reachable: `grd::parse` backs the gradient-preset import in
//! `engine::preset_import_cmds`, and `ase`/`aco` back swatch import and export. Each module has its
//! own `hostile_input` unit test, but those assert `let _ = parse(...)` on mutations: a panic would
//! abort the test without saying which parser and which byte caused it.
//!
//! What this file adds:
//!   1. every single-bit flip in a valid file, each wrapped in `catch_unwind` so a panic is
//!      reported with the exact byte and parser rather than as an opaque abort (§0.8: "parsers and
//!      decoders get mutation-fuzz tests run under `catch_unwind`");
//!   2. the round-trip and cap invariants the fuzz targets assert, checked deterministically so CI
//!      covers them even though the `fuzz` job is workflow-dispatch only;
//!   3. inputs sized to *exceed* each advertised cap, because a cap asserted only against a file
//!      that fits under it is a tautology — it cannot fail, so it proves nothing.
//!
//! The rule under test throughout: hostile input returns `Err`, never a panic and never an
//! unbounded allocation.

use photocraft_psd::aco::{AcoColor, AcoSwatch};
use photocraft_psd::grd::{GrdColor, GrdGradient, GrdStop};
use photocraft_psd::{PsdError, aco, ase, grd};
use std::panic::{RefUnwindSafe, UnwindSafe, catch_unwind};

fn sample_ase() -> Vec<u8> {
    use photocraft_psd::ase::{AseColor, AseEntry, AseFile, AseKind, AseSwatch as Sw, write};
    let sw = |name: &str| Sw { name: name.into(), color: AseColor::Rgb([1.0, 0.0, 0.0]), kind: AseKind::Process };
    write(&AseFile { entries: vec![AseEntry::Group { name: "Brand".into(), swatches: vec![sw("Red"), sw("Blue")] }, AseEntry::Swatch(sw("Loose"))] })
        .expect("writing a sample .ase")
}

fn sample_aco_swatches() -> Vec<AcoSwatch> {
    vec![
        AcoSwatch { name: "Pure Red".into(), color: AcoColor::Rgb([65535, 0, 0]) },
        AcoSwatch { name: "Cyan ink".into(), color: AcoColor::Cmyk([65535, 0, 0, 1234]) },
        AcoSwatch { name: "Lab".into(), color: AcoColor::Lab { l: 5321, a: -12800, b: 12700 } },
        AcoSwatch { name: "Gray".into(), color: AcoColor::Gray(5000) },
        AcoSwatch { name: "Book".into(), color: AcoColor::Other { space: 3, values: [1, 2, 3, 4] } },
    ]
}

fn sample_aco() -> Vec<u8> {
    aco::write(&sample_aco_swatches()).expect("writing a sample .aco")
}

fn sample_grd() -> Vec<u8> {
    use photocraft_psd::grd::write;
    write(&[
        GrdGradient {
            name: "Sunset".into(),
            noise: false,
            stops: vec![
                GrdStop { location: 0.0, midpoint: 0.5, color: GrdColor::Rgb([1.0, 0.0, 0.0]) },
                GrdStop { location: 1.0, midpoint: 0.5, color: GrdColor::Background },
            ],
            opacity: vec![(0.0, 1.0, 0.5), (1.0, 0.25, 0.5)],
        },
        GrdGradient { name: "Noisy".into(), noise: true, stops: Vec::new(), opacity: Vec::new() },
    ])
}

/// Flips every single bit of a valid file and requires that the parser returns or fails without
/// panicking. A flipped length or count must never be trusted into an allocation.
fn bitflip_sweep<T, F>(bytes: &[u8], name: &str, parse: F)
where
    F: Fn(&[u8]) -> Result<T, PsdError> + RefUnwindSafe + UnwindSafe,
{
    for byte in 0..bytes.len() {
        for bit in 0..8 {
            let mut b = bytes.to_vec();
            b[byte] ^= 1 << bit;
            let r = catch_unwind(|| parse(&b));
            assert!(r.is_ok(), "{name}: parse panicked after flipping bit {bit} of byte {byte} of {}", bytes.len());
        }
    }
}

#[test]
fn ase_bitflips_never_panic() {
    bitflip_sweep(&sample_ase(), "ase", ase::parse);
}

#[test]
fn aco_bitflips_never_panic() {
    bitflip_sweep(&sample_aco(), "aco", aco::parse);
}

#[test]
fn grd_bitflips_never_panic() {
    bitflip_sweep(&sample_grd(), "grd", grd::parse);
}

fn assert_finite(values: &[f32], what: &str) {
    for v in values {
        assert!(v.is_finite(), "{what} component {v} is not finite");
    }
}

/// Every number a parsed gradient carries must be finite and inside the ranges the module
/// documents, so a caller never divides by or formats a NaN and never reads past a 0..=1 bound.
fn assert_gradient_sane(g: &GrdGradient) {
    for s in &g.stops {
        assert!(s.location.is_finite() && (0.0..=1.0).contains(&s.location), "stop location {} out of range", s.location);
        assert!(s.midpoint.is_finite() && (0.0..=1.0).contains(&s.midpoint), "stop midpoint {} out of range", s.midpoint);
        match s.color {
            GrdColor::Rgb(v) => assert_finite(&v, "RGB"),
            GrdColor::Hsb(v) => assert_finite(&v, "HSB"),
            GrdColor::Cmyk(v) => assert_finite(&v, "CMYK"),
            GrdColor::Lab(v) => assert_finite(&v, "Lab"),
            GrdColor::Gray(v) => assert!(v.is_finite(), "gray {v} is not finite"),
            GrdColor::Foreground | GrdColor::Background => {}
        }
    }
    for &(l, o, m) in &g.opacity {
        assert!(l.is_finite() && (0.0..=1.0).contains(&l), "transparency location {l} out of range");
        assert!((0.0..=1.0).contains(&o), "opacity {o} outside 0..=1");
        assert!(m.is_finite() && (0.0..=1.0).contains(&m), "transparency midpoint {m} out of range");
    }
}

/// A gradient file that parses must obey the caps the module advertises and keep every numeric
/// field finite, so callers never divide by or format a NaN.
#[test]
fn grd_parsed_values_respect_caps_and_are_finite() {
    // No cap assertion here: the sample has two gradients and two stops, so a cap check would
    // hold for any writer and prove nothing. `grd_gradient_count_is_capped` and
    // `grd_stop_counts_are_capped` drive inputs over the caps instead.
    for g in &grd::parse(&sample_grd()).expect("sample .grd parses") {
        assert_gradient_sane(g);
    }
}

/// A file declaring more gradients than [`grd::MAX_GRADIENTS`] must be capped at the cap, not
/// trusted: the cap is the only thing between a file's own list length and the allocation it asks
/// for. The input deliberately exceeds the cap, otherwise the assertion below cannot fail.
#[test]
fn grd_gradient_count_is_capped() {
    let over = vec![GrdGradient { name: "g".into(), ..Default::default() }; grd::MAX_GRADIENTS + 1];
    let bytes = grd::write(&over);
    // Precondition: the file really must carry more gradients than the cap, or the assertion
    // below would hold for the wrong reason (the writer capping rather than the reader).
    // Precondition, by size: a file carrying one gradient more than the cap must be larger than
    // the same file at the cap. Counting `Grdn` windows would not work — the object nests — and
    // would keep passing even if the writer started capping, which is what this must catch.
    let at_cap = grd::write(&over[..grd::MAX_GRADIENTS]).len();
    assert!(bytes.len() > at_cap, "the written file is not larger than the capped one: {} vs {at_cap}", bytes.len());
    let f = grd::parse(&bytes).expect("a file over the gradient cap still parses");
    assert_eq!(f.len(), grd::MAX_GRADIENTS, "gradients must be capped at MAX_GRADIENTS, not taken as declared");
}

/// Same for the per-gradient stop lists: colour stops and transparency stops are capped
/// separately, so both are driven over their cap.
#[test]
fn grd_stop_counts_are_capped() {
    let stop = GrdStop { location: 0.5, midpoint: 0.5, color: GrdColor::Rgb([1.0, 0.0, 0.0]) };
    let over = GrdGradient { name: "wide".into(), noise: false, stops: vec![stop; grd::MAX_STOPS + 1], opacity: vec![(0.5, 1.0, 0.5); grd::MAX_STOPS + 1] };
    let bytes = grd::write(std::slice::from_ref(&over));
    // Precondition by size: a file carrying one stop more than the cap must be larger than the
    // same file at the cap. Counting `Lctn` windows would not work — it appears in both the colour
    // and the transparency list — and would keep passing even if the writer started capping,
    // which is exactly what this must catch. Each list is driven on its own, since they are
    // capped independently.
    let capped_stops = over.stops[..grd::MAX_STOPS].to_vec();
    let capped_opacity = over.opacity[..grd::MAX_STOPS].to_vec();
    let colour_only_over = GrdGradient { opacity: Vec::new(), ..over.clone() };
    let colour_only_capped = GrdGradient { stops: capped_stops, opacity: Vec::new(), ..over.clone() };
    assert!(grd::write(&[colour_only_over]).len() > grd::write(&[colour_only_capped]).len(), "colour stops over the cap are not larger than at the cap");
    let trans_only_over = GrdGradient { stops: Vec::new(), ..over.clone() };
    let trans_only_capped = GrdGradient { stops: Vec::new(), opacity: capped_opacity, ..over };
    assert!(grd::write(&[trans_only_over]).len() > grd::write(&[trans_only_capped]).len(), "transparency stops over the cap are not larger than at the cap");
    let f = grd::parse(&bytes).expect("a file over the stop cap still parses");
    assert_eq!(f[0].stops.len(), grd::MAX_STOPS, "colour stops must be capped at MAX_STOPS");
    assert_eq!(f[0].opacity.len(), grd::MAX_STOPS, "transparency stops must be capped at MAX_STOPS");
}

/// `Opct` is a unit float (`UntF` + a 4-byte unit tag), not a `doub`, so patching it needs a
/// different offset from the RGB components. Every numeric field now passes through `finite()`
/// before any clamp, so this pins that a non-finite value is repaired first — saturated, or for a
/// NaN replaced by the field default — rather than reaching `f32::clamp`, which propagates NaN and
/// would hand it straight back out of the parser. Every non-finite shape is tried at both places.
#[test]
fn grd_non_finite_opacity_and_midpoint_do_not_survive() {
    // 3.0e38 is finite as both f64 and f32, so it is not caught by the finiteness check and must
    // be brought into range by the saturate-then-clamp path; it belongs here to keep that honest.
    let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300, 3.0e38];
    // `doub`: key + 4-byte OSType + 8-byte value. `UntF`: key + OSType + 4-byte unit tag + value.
    // `Opct` is the interesting one: a unit float at a different offset from the RGB components, so
    // both the `doub` and `UntF` shapes are patched. `Mdpn` is written as an integer here, which
    // cannot carry a non-finite value at all, so it is skipped rather than pretended to be covered;
    // a hostile file storing `Mdpn` as a double is still guarded by `finite()` and by the fuzz
    // target's range assertion.
    for key in ["Opct", "Rd  ", "Grn ", "Bl  "] {
        for v in bad {
            let mut b = sample_grd();
            let at = b.windows(4).position(|w| w == key.as_bytes()).unwrap_or_else(|| panic!("the sample stores {key:?}"));
            let skip = match &b[at + 4..at + 8] {
                b"doub" => 8usize,
                b"UntF" => 12usize,
                other => panic!("{key} is stored as {other:?}, which this test cannot patch"),
            };
            b[at + skip..at + skip + 8].copy_from_slice(&v.to_be_bytes());
            let f = grd::parse(&b).unwrap_or_else(|e| panic!("{key} = {v} still parses: {e:?}"));
            for g in &f {
                assert_gradient_sane(g);
            }
        }
    }
}

/// `num` returns the raw f64 and `finite()` is the single guard every number passes through. This
/// pins that one policy end to end: only a NaN is truly absent and falls back to the documented
/// default (opacity 100 %, midpoint centred); every other shape — including `+/-inf` and a value
/// finite as `f64` but above `f32::MAX` — saturates and then clamps, so a large positive lands at
/// 1.0 and a large negative at 0.0, monotonic, never a snap back to the default.
/// `Mdpn` is written as a `long` by this crate's own writer, so no fixture can smuggle a
/// non-finite value into it and the guard inside `midpoint()` would never be exercised. The file
/// format allows a `double` there, so the descriptor is built by hand with `Mdpn` as a `doub`:
/// otherwise removing that guard changes nothing any test can see.
#[test]
fn grd_midpoint_stored_as_a_double_is_still_sanitised() {
    use photocraft_psd::descriptor::{Descriptor, Value, VersionedDescriptor};
    for v in [f64::NAN, f64::INFINITY, 1e300, 3.0e38] {
        // Driven through `Trns`, where the reader keeps the midpoint, rather than through `Clrs`,
        // where a colour stop also needs a colour this descriptor does not carry.
        let trns = Descriptor::new("TrnS").with("Mdpn", Value::Double(v));
        let grad = Descriptor::new("Grdn").with("Trns", Value::List(vec![Value::Descriptor(trns)]));
        let outer = Descriptor::new("Grdn").with("Grad", Value::Descriptor(grad));
        let file = Descriptor::new("null").with("GrdL", Value::List(vec![Value::Descriptor(outer)]));
        let mut g = b"8BGR".to_vec();
        g.extend_from_slice(&5u16.to_be_bytes());
        g.extend_from_slice(&VersionedDescriptor::new(file).to_bytes());

        let parsed = grd::parse(&g).unwrap_or_else(|e| panic!("Mdpn = {v} still parses: {e:?}"));
        let Some((_, _, m)) = parsed[0].opacity.first() else { panic!("the transparency stop was dropped") };
        // Pin the exact value, not just the range. Only a NaN is truly absent, so only it falls
        // back to the centred default; every other shape saturates and then clamps, so a large
        // positive lands at 1.0 and a large negative at 0.0 — monotonic, never a snap to 0.5.
        let expected = if v.is_nan() {
            0.5
        } else if v > 0.0 {
            1.0
        } else {
            0.0
        };
        assert_eq!(*m, expected, "a midpoint of {v} must be {expected}, got {m}");
    }
}

/// `.aco` stores CMYK as *ink* amounts in memory and the complement on disk: the file holds 0 for
/// 100 % ink and the reader adds the back. Nothing pinned that, so flipping either side would
/// silently invert every CMYK swatch.
#[test]
fn aco_cmyk_is_stored_inverted_and_reads_back_as_ink() {
    let swatches = vec![AcoSwatch { name: "k".into(), color: AcoColor::Cmyk([65535, 0, 0, 1234]) }];
    let bytes = aco::write(&swatches).expect("writing one CMYK swatch");

    // Version 1 section: version u16, count u16, then a 10-byte record.
    let space = u16::from_be_bytes([bytes[4], bytes[5]]);
    let stored: Vec<u16> = (0..4).map(|i| u16::from_be_bytes([bytes[6 + i * 2], bytes[7 + i * 2]])).collect();
    assert_eq!(space, aco::space::CMYK, "the record must be tagged CMYK");
    assert_eq!(stored, vec![0, u16::MAX, u16::MAX, u16::MAX - 1234], "the file stores the complement of the ink amounts");
    assert_eq!(aco::parse(&bytes).expect("the file parses").swatches, swatches, "the ink amounts must survive the round trip");
}

#[test]
fn grd_unrepresentable_numbers_follow_the_documented_policy() {
    // The *second* transparency stop carries opacity 0.25; the first is already 1.0, so patching
    // it could not tell "fell back to the default" from "the patch had no effect".
    let patch = |key: &[u8], v: f64| {
        let mut b = sample_grd();
        let at = b.windows(4).rposition(|w| w == key).expect("the sample stores the key");
        let skip = match &b[at + 4..at + 8] {
            b"doub" => 8usize,
            b"UntF" => 12usize,
            other => panic!("stored as {other:?}"),
        };
        b[at + skip..at + skip + 8].copy_from_slice(&v.to_be_bytes());
        grd::parse(&b).expect("still parses")
    };
    // Opacity. A NaN is the only truly absent value, so it alone falls back to the 100 % default.
    // Every other shape saturates then clamps, which is monotonic: a large positive stays opaque,
    // a large negative becomes transparent. Nothing snaps the wrong way as the number grows.
    for (v, expected) in [
        (f64::NAN, 1.0),          // absent -> the field default (opaque)
        (f64::INFINITY, 1.0),     // saturates to the top of the range
        (f64::NEG_INFINITY, 0.0), // saturates to the bottom
        (1e300, 1.0),             // over f32::MAX -> saturates up
        (-1e300, 0.0),            // under -f32::MAX -> saturates down
        (3.0e38, 1.0),            // finite as f32 -> clamps up
        (-3.0e38, 0.0),           // finite as f32 -> clamps down
    ] {
        let got = patch(b"Opct", v)[0].opacity[1].1;
        assert_eq!(got, expected, "Opct = {v} must read back as {expected}, got {got}");
    }
    // A colour channel has its own default (zero) for the absent case, and the same monotonic
    // saturate-then-clamp for everything else. Pin the exact number: a value reaching the clamp
    // from above would come back as 1.0, which is in range and would slip past a range-only check.
    for (v, expected) in [
        (f64::NAN, 0.0),          // absent -> the channel default
        (f64::INFINITY, 1.0),     // saturates up, then clamps to 1.0
        (f64::NEG_INFINITY, 0.0), // saturates down, then clamps to 0.0
        (1e300, 1.0),             // over f32::MAX -> saturates up
        (-1e300, 0.0),            // under -f32::MAX -> saturates down
        (3.0e38, 1.0),            // finite as f32 -> clamps up
        (-3.0e38, 0.0),           // finite as f32 -> clamps down
    ] {
        let f = patch(b"Bl  ", v);
        let GrdColor::Rgb([_, _, b]) = f[0].stops[0].color else { panic!("expected an RGB stop") };
        assert_eq!(b, expected, "a blue component of {v} must be {expected}, got {b}");
    }
}

/// A name long enough that the code-unit cap lands *between* the halves of a surrogate pair must
/// not be cut in half: a lone surrogate is ill-formed UTF-16, and writing it would make the file
/// unreadable to anything strict. Capping has to drop the whole character instead, and the file
/// must then survive its own round trip unchanged.
#[test]
fn aco_and_ase_never_split_a_surrogate_pair() {
    // "a" plus 600 emoji: 1 + 1200 units, so a 1024-unit cap lands mid-pair.
    let name = format!("a{}", "\u{1F600}".repeat(600));
    assert!(name.encode_utf16().count() > ase::MAX_NAME_UNITS, "the name must be longer than the cap");

    let swatches = vec![aco::AcoSwatch { name: name.clone(), color: aco::AcoColor::Gray(5000) }];
    let once = aco::write(&swatches).expect("writing a long non-ASCII name is capped, not refused");
    let back = aco::parse(&once).expect("the written file parses");
    // The exact expected name, not merely "it is valid UTF-16": the cap must drop the whole
    // character that does not fit, leaving "a" plus 511 complete pairs and nothing half-written.
    let expected = format!("a{}", "\u{1F600}".repeat(511));
    assert_eq!(back.swatches[0].name, expected, "the capped name must be whole characters only");
    assert_eq!(aco::write(&back.swatches).unwrap(), once, "the file must be stable across a round trip");

    use photocraft_psd::ase::{AseColor, AseEntry, AseFile, AseKind, AseSwatch, write};
    let file = AseFile { entries: vec![AseEntry::Swatch(AseSwatch { name, color: AseColor::Rgb([1.0, 0.0, 0.0]), kind: AseKind::Process })] };
    let a1 = write(&file).expect("writing a long non-ASCII name is capped, not refused");
    let ab = ase::parse(&a1).expect("the written file parses");
    let ase::AseEntry::Swatch(s) = &ab.entries[0] else { panic!("expected a loose swatch") };
    assert_eq!(s.name, expected, "the capped name must be whole characters only");
    assert_eq!(write(&ab).unwrap(), a1, "the file must be stable across a round trip");
}

/// Locations, midpoints and opacity are documented as 0..=1. A hostile file stores raw integers,
/// so the values are driven well outside that range and must come back clamped.
#[test]
fn grd_out_of_range_numbers_are_clamped() {
    let g = GrdGradient {
        name: "out of range".into(),
        noise: false,
        stops: vec![
            GrdStop { location: 2.0, midpoint: 5.0, color: GrdColor::Rgb([1.0, 0.0, 0.0]) },
            GrdStop { location: -3.0, midpoint: -1.0, color: GrdColor::Rgb([0.0, 1.0, 0.0]) },
        ],
        opacity: vec![(9.0, 5.0, 7.0), (-2.0, -1.0, 42.0)],
    };
    let f = grd::parse(&grd::write(&[g])).expect("out-of-range numbers still parse");
    assert_gradient_sane(&f[0]);
    // Range membership alone would be a tautology once the values are already in range: assert the
    // exact clamped result, so a clamp silently degrading into a pass-through is caught. Positive
    // over-range -> 1.0, negative -> 0.0.
    assert_eq!(f[0].stops[0].location, 1.0, "location 2.0 must clamp to 1.0");
    assert_eq!(f[0].stops[0].midpoint, 1.0, "midpoint 5.0 must clamp to 1.0");
    assert_eq!(f[0].stops[1].location, 0.0, "location -3.0 must clamp to 0.0");
    assert_eq!(f[0].stops[1].midpoint, 0.0, "midpoint -1.0 must clamp to 0.0");
    assert_eq!(f[0].opacity[0], (1.0, 1.0, 1.0), "all-positive over-range opacity must clamp to 1.0");
    assert_eq!(f[0].opacity[1], (0.0, 0.0, 1.0), "location/opacity clamp to 0.0, midpoint 42.0 to 1.0");
}

/// `num` returns the raw f64 unfiltered; `finite()` is the single guard, and a NaN is not the only
/// non-finite f32 it has to handle: a value that is finite as f64 and above `f32::MAX` (1e300)
/// would become an infinity on a bare cast, so `finite()` saturates it. Every numeric field must
/// come back finite, so both doors are checked.
#[test]
fn grd_huge_f64_does_not_overflow_to_infinity() {
    for key in ["Rd  ", "Grn ", "Bl  "] {
        let mut b = sample_grd();
        let at = b.windows(4).position(|w| w == key.as_bytes()).unwrap_or_else(|| panic!("the sample stores {key:?}"));
        assert_eq!(&b[at + 4..at + 8], b"doub", "{key} is stored as a double");
        b[at + 8..at + 16].copy_from_slice(&1e300f64.to_be_bytes());
        let f = grd::parse(&b).expect("a huge but finite component still parses");
        for g in &f {
            assert_gradient_sane(g);
        }
    }
}

/// A hostile file carrying a NaN must have it replaced, not handed on: `Rd  ` in the sample is a
/// double, so patching those eight bytes is enough to try to smuggle a NaN past the reader.
#[test]
fn grd_non_finite_numbers_do_not_survive_the_reader() {
    let mut b = sample_grd();
    let at = b.windows(4).position(|w| w == b"Rd  ").expect("the sample stores an RGB component");
    assert_eq!(&b[at + 4..at + 8], b"doub", "an RGB component is stored as a double");
    b[at + 8..at + 16].copy_from_slice(&f64::NAN.to_be_bytes());
    let f = grd::parse(&b).expect("a NaN component still parses");
    assert!(!f.is_empty());
    for g in &f {
        assert_gradient_sane(g);
    }
}

/// An `.ase` that parses must stay within the swatch cap and re-serialize to the same number of
/// colours, so an import never silently drops or invents entries.
#[test]
fn ase_round_trip_preserves_swatch_count() {
    use photocraft_psd::ase::write;
    let f = ase::parse(&sample_ase()).expect("sample .ase parses");
    // No cap assertion: three swatches cannot exceed a cap of 100 000. `ase_swatch_cap_is_enforced_
    // on_hostile_input` drives a file that does.
    let n = f.swatch_count();
    let out = write(&f).expect("re-writing a parsed .ase");
    let back = ase::parse(&out).expect("the rewritten .ase parses");
    assert_eq!(back.swatch_count(), n, "swatch count changed across a round trip");
}

/// A file with more colour blocks than [`ase::MAX_SWATCHES`] must be refused. Each block here is
/// the smallest legal colour block, so the file is small but the declared count is over the cap.
#[test]
fn ase_swatch_cap_is_enforced_on_hostile_input() {
    let mut body = Vec::new();
    body.extend_from_slice(&0u16.to_be_bytes()); // name length 0
    body.extend_from_slice(b"Gray");
    body.extend_from_slice(&0f32.to_be_bytes()); // gray component
    body.extend_from_slice(&2u16.to_be_bytes()); // colour type: process
    let mut block = Vec::new();
    block.extend_from_slice(&1u16.to_be_bytes()); // block type: colour entry
    block.extend_from_slice(&(body.len() as u32).to_be_bytes());
    block.extend_from_slice(&body);

    let total = ase::MAX_SWATCHES + 1;
    let mut f = b"ASEF".to_vec();
    f.extend_from_slice(&1u16.to_be_bytes()); // major
    f.extend_from_slice(&0u16.to_be_bytes()); // minor
    f.extend_from_slice(&(total as u32).to_be_bytes());
    for _ in 0..total {
        f.extend_from_slice(&block);
    }
    assert!(matches!(ase::parse(&f), Err(PsdError::LimitExceeded(_))), "a file over MAX_SWATCHES must be refused, not silently truncated");
    // The same file one block under the cap is accepted, so the cap is what refuses it.
    f.truncate(f.len() - block.len());
    let trimmed = total - 1;
    // signature (4) + major (2) + minor (2) puts the block count at offset 8.
    f[8..12].copy_from_slice(&(trimmed as u32).to_be_bytes());
    assert!(ase::parse(&f).is_ok(), "a file exactly at MAX_SWATCHES must still parse");
}

/// A name longer than [`ase::MAX_NAME_UNITS`] is truncated on read rather than refused: the file
/// is legal, only the name is long, and `read_name` is the only thing standing between the declared
/// length and the string it builds.
#[test]
fn ase_over_long_name_is_truncated_on_read() {
    let units = ase::MAX_NAME_UNITS + 976; // 2000
    let mut body = Vec::new();
    body.extend_from_slice(&(units as u16).to_be_bytes()); // name length, no trailing null here
    for _ in 0..units {
        body.extend_from_slice(&(b'x' as u16).to_be_bytes());
    }
    body.extend_from_slice(b"Gray");
    body.extend_from_slice(&0f32.to_be_bytes());
    body.extend_from_slice(&2u16.to_be_bytes());

    let mut f = b"ASEF".to_vec();
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&0u16.to_be_bytes());
    f.extend_from_slice(&1u32.to_be_bytes()); // one block
    f.extend_from_slice(&1u16.to_be_bytes()); // colour entry
    f.extend_from_slice(&(body.len() as u32).to_be_bytes());
    f.extend_from_slice(&body);

    let parsed = ase::parse(&f).expect("an over-long name is truncated, not refused");
    let ase::AseEntry::Swatch(s) = &parsed.entries[0] else { panic!("expected a loose swatch") };
    assert_eq!(s.name.encode_utf16().count(), ase::MAX_NAME_UNITS, "name must be capped at MAX_NAME_UNITS");
}

/// An `.aco` that parses must re-serialize to the same swatch count, with no name over the cap.
#[test]
fn aco_round_trip_preserves_swatch_count() {
    use photocraft_psd::aco::write;
    // No cap assertion here. `aco::MAX_SWATCHES` is `u16::MAX` against a `u16` on-disk count, so it
    // is unfalsifiable by construction, and the sample names are far shorter than the name cap;
    // `aco_over_long_name_is_truncated_on_read` is where that cap is actually driven.
    let f = aco::parse(&sample_aco()).expect("sample .aco parses");
    let out = write(&f.swatches).expect("re-writing parsed swatches");
    let back = aco::parse(&out).expect("the rewritten .aco parses");
    assert_eq!(back.swatches.len(), f.swatches.len(), "swatch count changed across a round trip");
}

/// A version 2 section whose name declares more units than [`aco::MAX_NAME_UNITS`] must come back
/// truncated. The count is written by hand because `write` will not emit one.
#[test]
fn aco_over_long_name_is_truncated_on_read() {
    let units = 2000usize;
    let mut f = Vec::new();
    f.extend_from_slice(&2u16.to_be_bytes()); // version 2 section
    f.extend_from_slice(&1u16.to_be_bytes()); // one swatch
    f.extend_from_slice(&0u16.to_be_bytes()); // colour space: RGB
    for v in [1u16, 2, 3, 0] {
        f.extend_from_slice(&v.to_be_bytes()); // the four stored values
    }
    // The count includes the terminating null, as `write` does and as the format requires.
    f.extend_from_slice(&((units + 1) as u32).to_be_bytes());
    for _ in 0..units {
        f.extend_from_slice(&(b'x' as u16).to_be_bytes());
    }
    f.extend_from_slice(&0u16.to_be_bytes()); // terminating null
    let parsed = aco::parse(&f).expect("an over-long name is truncated, not refused");
    assert_eq!(parsed.swatches[0].name.encode_utf16().count(), aco::MAX_NAME_UNITS, "name must be capped at MAX_NAME_UNITS");
}

/// `write` must cap too, or a long name from a caller is written whole. The declared length in
/// the version 2 record counts the terminating null, so it must read back as MAX_NAME_UNITS + 1.
#[test]
fn aco_write_caps_the_name() {
    let swatches = vec![AcoSwatch { name: "x".repeat(5000), color: AcoColor::Gray(5000) }];
    let bytes = aco::write(&swatches).expect("an over-long name is capped on write, not refused");
    // Names go out as UTF-16 big-endian, so 'x' is the byte pair 00 78.
    let run = [0u8, b'x'];
    let mut needle = Vec::new();
    for _ in 0..8 {
        needle.extend_from_slice(&run);
    }
    let at = bytes.windows(needle.len()).position(|w| w == needle.as_slice()).expect("the name is written into the file");
    let declared = u32::from_be_bytes([bytes[at - 4], bytes[at - 3], bytes[at - 2], bytes[at - 1]]);
    assert_eq!(declared as usize, aco::MAX_NAME_UNITS + 1, "write must declare at most MAX_NAME_UNITS units plus the null");
    let back = aco::parse(&bytes).expect("the written file parses");
    assert_eq!(back.swatches[0].name.encode_utf16().count(), aco::MAX_NAME_UNITS);
}

/// `write` must cap an over-long name too, and it must count the terminating null in the u16
/// length field it writes. Without the cap that field would wrap, but only once the name passes
/// 65 535 units — so the case that proves it uses 70 000, not merely "long".
#[test]
fn ase_write_caps_the_name() {
    use photocraft_psd::ase::{AseColor, AseEntry, AseFile, AseKind, AseSwatch, write};
    let sw = AseSwatch { name: "x".repeat(70_000), color: AseColor::Rgb([1.0, 0.0, 0.0]), kind: AseKind::Process };
    let bytes = write(&AseFile { entries: vec![AseEntry::Swatch(sw)] }).expect("an over-long name is capped, not refused");
    let run = [0u8, b'x'];
    let mut needle = Vec::new();
    for _ in 0..8 {
        needle.extend_from_slice(&run);
    }
    let at = bytes.windows(needle.len()).position(|w| w == needle.as_slice()).expect("the name is written into the file");
    let declared = u16::from_be_bytes([bytes[at - 2], bytes[at - 1]]) as usize;
    assert_eq!(declared, ase::MAX_NAME_UNITS + 1, "write must declare at most MAX_NAME_UNITS units plus the null");
    let back = ase::parse(&bytes).expect("the written file parses");
    let ase::AseEntry::Swatch(s) = &back.entries[0] else { panic!("expected a loose swatch") };
    assert_eq!(s.name.encode_utf16().count(), ase::MAX_NAME_UNITS);
}

/// Truncating a `.grd` anywhere must be an error: unlike `.aco` there is no optional second
/// section, so a prefix can never be a complete file.
#[test]
fn truncate_grd_is_always_an_error() {
    let b = sample_grd();
    for cut in 0..b.len() {
        let r = catch_unwind(|| grd::parse(&b[..cut]));
        assert!(r.is_ok(), "grd: parse panicked on a {cut}/{}-byte prefix", b.len());
        assert!(r.unwrap().is_err(), "grd: truncation at {cut}/{} parsed successfully", b.len());
    }
    assert!(grd::parse(&b).is_ok(), "grd: the full file must parse");
}

/// Truncating an `.ase` anywhere must be an error: the block count is fixed up front, so a prefix
/// cannot contain the number of blocks the header claims.
#[test]
fn truncate_ase_is_always_an_error() {
    let b = sample_ase();
    for cut in 0..b.len() {
        let r = catch_unwind(|| ase::parse(&b[..cut]));
        assert!(r.is_ok(), "ase: parse panicked on a {cut}/{}-byte prefix", b.len());
        assert!(r.unwrap().is_err(), "ase: truncation at {cut}/{} parsed successfully", b.len());
    }
    assert!(ase::parse(&b).is_ok(), "ase: the full file must parse");
}

/// Truncating an `.aco` inside its first section must be an error. A cut that lands exactly after a
/// complete first section is *not* an error — the reader stops there, having read the version 1
/// section, and ignores whatever follows — so the sweep stops at that boundary rather than
/// claiming every prefix fails. The boundary is derived from the sample so adding a swatch
/// cannot silently weaken the sweep.
#[test]
fn truncate_aco_inside_first_section_is_an_error() {
    let b = sample_aco();
    // version u16 + count u16, then one 10-byte colour record per swatch.
    let first_section = 4 + sample_aco_swatches().len() * 10;
    assert!(first_section <= b.len(), "the sample must contain a whole first section");
    for cut in 0..first_section {
        let r = catch_unwind(|| aco::parse(&b[..cut]));
        assert!(r.is_ok(), "aco: parse panicked on a {cut}/{first_section}-byte prefix");
        assert!(r.unwrap().is_err(), "aco: truncation inside the first section at {cut} parsed successfully");
    }
    match catch_unwind(|| aco::parse(&b[..first_section])) {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => panic!("aco: a complete first section must parse, got {e:?}"),
        Err(_) => panic!("aco: a complete first section panicked"),
    }
    assert!(aco::parse(&b).is_ok(), "aco: the full file must parse");
}

/// Structural `.ase` cases the block reader has to survive: a group end with no start, a group
/// started inside another, a block length that disagrees with what the body consumes, and an
/// unknown colour model whose components are cut short. Each is a clean error or a clean parse,
/// never a panic and never a read past the block.
#[test]
fn ase_structurally_hostile_blocks() {
    use photocraft_psd::ase::{AseColor, AseEntry};
    let header = |count: u32| {
        let mut f = b"ASEF".to_vec();
        f.extend_from_slice(&1u16.to_be_bytes());
        f.extend_from_slice(&0u16.to_be_bytes());
        f.extend_from_slice(&count.to_be_bytes());
        f
    };
    let block = |kind: u16, body: &[u8]| {
        let mut b = kind.to_be_bytes().to_vec();
        b.extend_from_slice(&(body.len() as u32).to_be_bytes());
        b.extend_from_slice(body);
        b
    };
    let group_start = |name: &str| {
        // The name is a u16 code-unit count followed by the units themselves, not raw bytes.
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut b = (units.len() as u16).to_be_bytes().to_vec();
        for u in units {
            b.extend_from_slice(&u.to_be_bytes());
        }
        block(0xC001, &b)
    };
    let group_end = || block(0xC002, &[]);
    let colour = |model: &[u8], floats: &[u8]| {
        let mut b = 0u16.to_be_bytes().to_vec();
        b.extend_from_slice(model);
        b.extend_from_slice(floats);
        b.extend_from_slice(&2u16.to_be_bytes());
        block(0x0001, &b)
    };

    // A group end with no start: documented as ignored, so this parses and yields nothing.
    let mut f = header(1);
    f.extend_from_slice(&group_end());
    let parsed = ase::parse(&f).expect("an unbalanced group end must parse");
    assert!(parsed.entries.is_empty(), "a stray group end must not invent an entry or an empty group");

    // A group started inside another: the outer one closes, both groups are emitted.
    let mut f = header(3);
    f.extend_from_slice(&group_start("outer"));
    f.extend_from_slice(&group_start("inner"));
    f.extend_from_slice(&colour(b"Gray", &1f32.to_be_bytes()));
    let parsed = ase::parse(&f).expect("a nested group start must parse");
    assert_eq!(parsed.swatch_count(), 1, "the colour must land in exactly one group");
    // The message above is only true if the colour is in the *inner* group; check it.
    let inner_holds_it = parsed.entries.iter().any(|e| matches!(e, AseEntry::Group { name, swatches } if name == "inner" && swatches.len() == 1));
    assert!(inner_holds_it, "the colour must land in the inner group, not the outer one");
    let names: Vec<&str> = parsed
        .entries
        .iter()
        .filter_map(|e| match e {
            AseEntry::Group { name, .. } => Some(name.as_str()),
            AseEntry::Swatch(_) => None,
        })
        .collect();
    assert_eq!(names, vec!["outer", "inner"], "both groups must be closed and emitted");

    // A block length shorter than the body it describes: the colour's floats are cut off.
    // Gray carries exactly one component, so declaring the block one byte shorter leaves that
    // component half-read. It must be refused, not guessed at.
    let mut short = 0u16.to_be_bytes().to_vec();
    short.extend_from_slice(b"Gray");
    short.extend_from_slice(&0f32.to_be_bytes());
    short.extend_from_slice(&2u16.to_be_bytes()); // colour type: process
    let mut f = header(1);
    let mut b = 0x0001u16.to_be_bytes().to_vec();
    b.extend_from_slice(&((short.len() - 4) as u32).to_be_bytes());
    b.extend_from_slice(&short);
    f.extend_from_slice(&b);
    let r = catch_unwind(|| ase::parse(&f));
    assert!(r.is_ok(), "a truncated colour component must not panic");
    assert!(r.unwrap().is_err(), "a colour component cut short must be refused, not guessed at");

    // A block that ends after the components, with no colour type at all, is *accepted*: some
    // writers omit it and the reader defaults it to process. Pinning that, because it is the one
    // place this parser is deliberately lenient.
    let mut f = header(1);
    let mut b = 0x0001u16.to_be_bytes().to_vec();
    b.extend_from_slice(&(short.len().saturating_sub(2) as u32).to_be_bytes());
    b.extend_from_slice(&short[..short.len() - 2]);
    f.extend_from_slice(&b);
    let parsed = ase::parse(&f).expect("a colour block with no colour type is accepted by design");
    assert_eq!(parsed.swatch_count(), 1, "it must still yield exactly one swatch");

    // An unknown colour model keeps whatever bytes follow it verbatim.
    // The model tag is four bytes; give it four, then a known payload, so "kept verbatim" can be
    // checked rather than merely asserted.
    let mut f = header(1);
    let payload = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    f.extend_from_slice(&colour(b"XYZ ", &payload));
    let parsed = ase::parse(&f).expect("an unknown colour model is kept verbatim");
    assert_eq!(parsed.swatch_count(), 1, "an unknown model still yields one swatch");
    let AseEntry::Swatch(sw) = &parsed.entries[0] else { panic!("expected a loose swatch") };
    let AseColor::Other { model, data } = &sw.color else { panic!("an unknown model must be kept as Other, got {:?}", sw.color) };
    assert_eq!(model, b"XYZ ", "the model tag must be preserved verbatim");
    assert_eq!(data.as_slice(), &payload, "the component bytes must be preserved verbatim");
}

/// A `.aco` whose version 1 and version 2 sections disagree about the number of swatches is the
/// realistic hostile shape: Photoshop writes both, and a corrupted length lands in one of them.
/// The version 2 names must still line up with the version 1 colours they belong to.
#[test]
fn aco_mismatched_section_counts_do_not_panic() {
    // version 1 with two colours and no names, then a version 2 section claiming five.
    let mut f = vec![0u8, 1, 0, 2];
    for _ in 0..2 {
        f.extend_from_slice(&[0, 0, 0, 1, 0, 2, 0, 3, 0, 0]);
    }
    f.extend_from_slice(&[0, 2, 0, 5]);
    for _ in 0..5 {
        f.extend_from_slice(&[0, 0, 0, 1, 0, 2, 0, 3, 0, 0]);
        // An .aco record has no colour-type field, so a one-character name is exactly two units:
        // two u16 values, not one.
        f.extend_from_slice(&2u32.to_be_bytes());
        f.extend_from_slice(&[0, 78, 0, 0]); // 'N' and its null
    }
    let r = catch_unwind(|| aco::parse(&f));
    assert!(r.is_ok(), "mismatched section counts must not panic");
    let parsed = r.unwrap().expect("the version 2 section is the one with names");
    assert_eq!(parsed.swatches.len(), 5, "the named section is authoritative when present");
    for sw in &parsed.swatches {
        assert_eq!(sw.name, "N", "each version 2 record must keep its name beside its own colour");
    }
}

/// Deep descriptor nesting is the one hostile shape that aborts rather than returning: a stack
/// overflow is not a panic, so `catch_unwind` cannot catch it and it would take the process down.
/// `descriptor` bounds it at `MAX_DEPTH` and its own unit test drives `Descriptor::from_bytes`
/// directly; this drives the same limit through `grd::parse`, which is the path a user's gradient
/// preset actually takes, and asserts the specific error rather than merely surviving.
///
/// The previous version of this test built `b"null"` repeated 2000 times, which is not a descriptor
/// at all — no item counts, no `Objc` tags — so the parser rejected it on the first field and
/// nothing ever recursed. It passed for the wrong reason.
#[test]
fn grd_deeply_nested_descriptor_hits_the_depth_limit() {
    use photocraft_psd::descriptor::{Descriptor, MAX_DEPTH, Value, VersionedDescriptor};
    let mut d = Descriptor::new("null");
    for _ in 0..(MAX_DEPTH + 5) {
        d = Descriptor::new("null").with("in  ", Value::Descriptor(d));
    }
    let mut g = b"8BGR".to_vec();
    g.extend_from_slice(&5u16.to_be_bytes());
    g.extend_from_slice(&VersionedDescriptor::new(d).to_bytes());
    // This is the real assertion: nesting past the limit is refused, not recursed into.
    assert!(matches!(grd::parse(&g), Err(PsdError::LimitExceeded(_))), "nesting past MAX_DEPTH must be refused through the .grd path, not recursed into");

    // The limit discriminates rather than being a blanket refusal: a shallow nesting is accepted
    // (rejected later, for holding no gradients — a different error), while a deep one is a depth
    // refusal. Each wrapper adds more than one unit of depth, so the exact crossover is an
    // implementation detail; what matters is that normal files pass and runaway ones are stopped.
    let build = |levels: usize| {
        let mut d = Descriptor::new("null");
        for _ in 0..levels {
            d = Descriptor::new("null").with("in  ", Value::Descriptor(d));
        }
        let mut g = b"8BGR".to_vec();
        g.extend_from_slice(&5u16.to_be_bytes());
        g.extend_from_slice(&VersionedDescriptor::new(d).to_bytes());
        g
    };
    assert!(!matches!(grd::parse(&build(4)), Err(PsdError::LimitExceeded(_))), "a shallow descriptor must not be refused for depth");
    assert!(
        matches!(grd::parse(&build(MAX_DEPTH + 1)), Err(PsdError::LimitExceeded(_))),
        "a descriptor past the depth limit must be a depth refusal, not an abort"
    );
}

/// Every colour model's components must come back finite and inside their documented range, not
/// just RGB. The writer only emits RGB and foreground/background stops, so each model is built by
/// hand with an out-of-range or non-representable value in every component, then driven through
/// `grd::parse`. Without this, the clamps on HSB, CMYK, Gray and the L* channel — and the finite
/// guard on the Lab a*/b* channels — have no test and can be removed unnoticed.
#[test]
fn grd_every_colour_model_is_clamped_and_finite() {
    use photocraft_psd::descriptor::{Descriptor, Value, VersionedDescriptor};

    // Build a one-stop gradient whose colour is `class_id` with the given (key, value) components.
    let build = |class_id: &str, comps: &[(&str, f64)]| -> Vec<u8> {
        let mut colour = Descriptor::new(class_id);
        for (k, v) in comps {
            colour = colour.with(k, Value::Double(*v));
        }
        let stop = Descriptor::new("Clrt")
            .with("Clr ", Value::Descriptor(colour))
            .with("Type", Value::Enumerated { type_id: photocraft_psd::descriptor::Id::new("Clry"), value: photocraft_psd::descriptor::Id::new("UsrS") })
            .with("Lctn", Value::Integer(0))
            .with("Mdpn", Value::Integer(50));
        let grad = Descriptor::new("Grdn").with("Clrs", Value::List(vec![Value::Descriptor(stop)]));
        let outer = Descriptor::new("Grdn").with("Grad", Value::Descriptor(grad));
        let file = Descriptor::new("null").with("GrdL", Value::List(vec![Value::Descriptor(outer)]));
        let mut g = b"8BGR".to_vec();
        g.extend_from_slice(&5u16.to_be_bytes());
        g.extend_from_slice(&VersionedDescriptor::new(file).to_bytes());
        g
    };

    let hostile = [1e300, 3.0e38, -3.0e38, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for &x in &hostile {
        // The exact boundary each clamp must land on, so a weaker clamp (e.g. a*/b* to -100..=100
        // instead of -128..=127, or a unit range to 0..=0.5) is caught, not just an out-of-range
        // value. A NaN becomes 0.0 (the finite guard's fallback for `f`), a positive saturates to
        // the top of the range, a negative to the bottom.
        let unit = if x.is_nan() || x < 0.0 { 0.0f32 } else { 1.0 };
        let lab_l = if x.is_nan() || x < 0.0 { 0.0f32 } else { 100.0 };
        let lab_ab = if x.is_nan() {
            0.0f32
        } else if x > 0.0 {
            127.0
        } else {
            -128.0
        };
        // RGB components are 0..=255 before the /255; the result must land in 0..=1.
        let f = grd::parse(&build("RGBC", &[("Rd  ", x), ("Grn ", x), ("Bl  ", x)])).expect("RGB parses");
        let GrdColor::Rgb(rgb) = f[0].stops[0].color else { panic!("expected an RGB stop") };
        for v in rgb {
            assert_eq!(v, unit, "RGB component for input {x} must saturate to {unit}");
        }
        // HSB: hue 0..=360, saturation and brightness 0..=1.
        let f = grd::parse(&build("HSBC", &[("H   ", x), ("Strt", x), ("Brgh", x)])).expect("HSB parses");
        if let GrdColor::Hsb([h, sat, br]) = f[0].stops[0].color {
            assert!(h.is_finite() && (0.0..=360.0).contains(&h), "hue {h} out of range for input {x}");
            assert_eq!(sat, unit, "saturation for input {x} must saturate to {unit}");
            assert_eq!(br, unit, "brightness for input {x} must saturate to {unit}");
        } else {
            panic!("expected an HSB stop");
        }
        // CMYK: every ink 0..=1.
        let f = grd::parse(&build("CMYC", &[("Cyn ", x), ("Mgnt", x), ("Ylw ", x), ("Blck", x)])).expect("CMYK parses");
        if let GrdColor::Cmyk(c) = f[0].stops[0].color {
            for v in c {
                assert_eq!(v, unit, "CMYK ink for input {x} must saturate to {unit}");
            }
        } else {
            panic!("expected a CMYK stop");
        }
        // Lab: L* 0..=100, a*/b* bounded to the CIELAB gamut -128..=127.
        let f = grd::parse(&build("LbCl", &[("Lmnc", x), ("A   ", x), ("B   ", x)])).expect("Lab parses");
        if let GrdColor::Lab([l, a, b]) = f[0].stops[0].color {
            assert_eq!(l, lab_l, "L* for input {x} must saturate to {lab_l}");
            assert_eq!(a, lab_ab, "a* for input {x} must saturate to the gamut edge {lab_ab}");
            assert_eq!(b, lab_ab, "b* for input {x} must saturate to the gamut edge {lab_ab}");
        } else {
            panic!("expected a Lab stop");
        }
        // Gray: 0..=1.
        let f = grd::parse(&build("Grsc", &[("Gry ", x)])).expect("Gray parses");
        if let GrdColor::Gray(g) = f[0].stops[0].color {
            assert_eq!(g, unit, "gray for input {x} must saturate to {unit}");
        } else {
            panic!("expected a Gray stop");
        }
    }
}

/// Hue is circular, so an out-of-range hue must *wrap*, not clamp: 370 degrees is 10, not 360, and
/// -10 is 350, not 0. Clamping would hand back a different colour; this pins the wrap so a
/// regression to `clamp` is caught (both produce in-range values, so a range check alone would not).
#[test]
fn grd_hue_wraps_rather_than_clamping() {
    use photocraft_psd::descriptor::{Descriptor, Id, Value, VersionedDescriptor};
    let build = |hue: f64| {
        let colour = Descriptor::new("HSBC").with("H   ", Value::Double(hue)).with("Strt", Value::Double(50.0)).with("Brgh", Value::Double(50.0));
        let stop = Descriptor::new("Clrt")
            .with("Clr ", Value::Descriptor(colour))
            .with("Type", Value::Enumerated { type_id: Id::new("Clry"), value: Id::new("UsrS") })
            .with("Lctn", Value::Integer(0))
            .with("Mdpn", Value::Integer(50));
        let grad = Descriptor::new("Grdn").with("Clrs", Value::List(vec![Value::Descriptor(stop)]));
        let outer = Descriptor::new("Grdn").with("Grad", Value::Descriptor(grad));
        let file = Descriptor::new("null").with("GrdL", Value::List(vec![Value::Descriptor(outer)]));
        let mut g = b"8BGR".to_vec();
        g.extend_from_slice(&5u16.to_be_bytes());
        g.extend_from_slice(&VersionedDescriptor::new(file).to_bytes());
        g
    };
    let hue_of = |bytes: &[u8]| -> f32 {
        let f = grd::parse(bytes).expect("parses");
        let GrdColor::Hsb([h, _, _]) = f[0].stops[0].color else { panic!("expected an HSB stop") };
        h
    };
    assert!((hue_of(&build(370.0)) - 10.0).abs() < 0.01, "370 degrees must wrap to 10, not clamp to 360");
    assert!((hue_of(&build(-10.0)) - 350.0).abs() < 0.01, "-10 degrees must wrap to 350, not clamp to 0");
    // And a value already in range is unchanged.
    assert!((hue_of(&build(123.0)) - 123.0).abs() < 0.01, "an in-range hue must be unchanged");
    // Exactly 360 is in range and legal, so it is kept, not wrapped to 0 (a different position for
    // anything that interpolates in HSB). Unconditional `rem_euclid(360)` would map it to 0, so this
    // pins the in-range guard that keeps the boundary intact.
    assert_eq!(hue_of(&build(360.0)), 360.0, "an in-range hue of exactly 360 must be preserved, not wrapped to 0");
    assert_eq!(hue_of(&build(0.0)), 0.0, "a hue of 0 must stay 0");
}

/// A signature-only or empty file must be a clean error, never a panic and never an
/// out-of-bounds read.
#[test]
fn empty_and_signature_only_inputs_error() {
    fn no_panic_then_err<T>(r: Result<Result<T, PsdError>, Box<dyn core::any::Any + Send>>, what: &str) {
        match r {
            Ok(Ok(_)) => panic!("{what}: an empty file must be a clean error, not parse"),
            Ok(Err(_)) => {}
            Err(_) => panic!("{what}: parse panicked on an empty file"),
        }
    }
    no_panic_then_err(catch_unwind(|| grd::parse(b"")), "grd");
    no_panic_then_err(catch_unwind(|| ase::parse(b"")), "ase");
    no_panic_then_err(catch_unwind(|| aco::parse(b"")), "aco");
    assert!(matches!(grd::parse(b"8BGR"), Err(PsdError::UnexpectedEof { .. })));
    assert!(matches!(grd::parse(b"8BG"), Err(PsdError::UnexpectedEof { .. })));
    assert!(ase::parse(b"ASEF").is_err());
    assert!(aco::parse(b"").is_err());
}
