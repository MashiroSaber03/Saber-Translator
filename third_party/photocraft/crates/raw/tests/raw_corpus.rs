//! Real-camera corpus tests against `corpus/pixls` (raw.pixls.us, public domain): one file per
//! decode path, oracle-checked against values measured when the corpus was pinned (the
//! `inspect` example prints them). Opt-in like every real-file corpus:
//! `cargo xtask corpus --pixls` fetches, `cargo xtask test-corpus` runs.
#![cfg(feature = "corpus")]

use photocraft_raw::{Limits, RawError, RawFormat};
use std::path::PathBuf;

fn pixls(name: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pixls").join(name);
    if !p.is_file() {
        panic!("{} is missing: run `cargo xtask corpus --pixls` (or `--all`) first", p.display());
    }
    p
}

struct Want<'a> {
    file: &'static str,
    format: RawFormat,
    make: &'static str,
    model: &'static str,
    /// Sensor width and height.
    size: (usize, usize),
    /// The four CFA colours, row-major.
    cfa: [u8; 4],
    black: &'a [f32],
    white: [f32; 3],
}

const WANTS: &[Want] = &[
    Want {
        file: "IMG_4059.CR2",
        format: RawFormat::Cr2,
        make: "Canon",
        model: "Canon PowerShot SX50 HS",
        size: (4176, 3062),
        cfa: [0, 1, 1, 2],
        black: &[128.0; 4],
        white: [4095.0; 3],
    },
    // The same camera after DNG Converter: the sensor data round-trips identically.
    Want {
        file: "CRW_4061.DNG",
        format: RawFormat::Dng,
        make: "Canon",
        model: "Canon PowerShot SX50 HS",
        size: (4176, 3062),
        cfa: [0, 1, 1, 2],
        black: &[127.0],
        white: [4095.0; 3],
    },
    Want {
        file: "JD1_8203.NEF",
        format: RawFormat::Nef,
        make: "NIKON CORPORATION",
        model: "NIKON D3",
        size: (4288, 2844),
        cfa: [0, 1, 1, 2],
        black: &[0.0],
        white: [4095.0; 3],
    },
    // Sony cRAW: the saturation level comes from the compressed code's range.
    Want {
        file: "DSC00009.ARW",
        format: RawFormat::Arw,
        make: "SONY",
        model: "DSC-RX0",
        size: (4832, 3224),
        cfa: [0, 1, 1, 2],
        // No readable BlackLevel tag: the cRAW fallback of 512 (#1902). The file's darkest
        // samples sit near 800, so this body's real level is still to be read.
        black: &[512.0],
        white: [16301.0; 3],
    },
    // The G9's high-resolution mode: a 10480 × 7794 sensor cropped to 10368 × 7776.
    Want {
        file: "P1000475.RW2",
        format: RawFormat::Rw2,
        make: "Panasonic",
        model: "DC-G9",
        size: (10480, 7794),
        cfa: [0, 1, 1, 2],
        black: &[127.0, 128.0, 128.0, 127.0],
        white: [4095.0; 3],
    },
    // Four Thirds Olympus: GRBG instead of the usual RGGB.
    Want {
        file: "E_1__C106743_gredos.ORF",
        format: RawFormat::Orf,
        make: "OLYMPUS CORPORATION",
        model: "E-1",
        size: (2624, 1966),
        cfa: [1, 0, 2, 1],
        black: &[65.0; 4],
        white: [4095.0; 3],
    },
];

#[test]
fn decode_matches_the_measured_oracles() {
    for w in WANTS {
        let bytes = std::fs::read(pixls(w.file)).unwrap_or_else(|e| panic!("{}: {e}", w.file));
        let s = photocraft_raw::decode(&bytes, &Limits::default()).unwrap_or_else(|e| panic!("{}: {e}", w.file));
        assert_eq!(s.format, w.format, "{}", w.file);
        assert_eq!(s.make.as_deref(), Some(w.make), "{}", w.file);
        assert_eq!(s.model.as_deref(), Some(w.model), "{}", w.file);
        assert_eq!((s.width, s.height), w.size, "{}", w.file);
        let cfa = s.cfa.as_ref().unwrap_or_else(|| panic!("{}: no CFA", w.file));
        assert_eq!(cfa.colors, w.cfa, "{}", w.file);
        assert_eq!(s.black.values, w.black, "{}", w.file);
        for ch in 0..3 {
            assert!((s.white[ch] - w.white[ch]).abs() < 0.5, "{} white[{ch}] = {}", w.file, s.white[ch]);
        }
    }
}

#[test]
fn developing_produces_the_crop_in_16bit_rgb() {
    for w in WANTS {
        let bytes = std::fs::read(pixls(w.file)).unwrap_or_else(|e| panic!("{}: {e}", w.file));
        let s = photocraft_raw::decode(&bytes, &Limits::default()).unwrap_or_else(|e| panic!("{}: {e}", w.file));
        let d = photocraft_raw::develop_sensor(&s, &photocraft_raw::DevelopOptions::default()).unwrap_or_else(|e| panic!("{}: {e}", w.file));
        assert_eq!((d.width as usize, d.height as usize), (s.crop.width, s.crop.height), "{}", w.file);
        assert_eq!(d.rgb.len(), d.width as usize * d.height as usize * 3, "{}", w.file);
        // A real exposure covers a sane share of the range and never leaves it.
        let (mut lo, mut hi) = (u16::MAX, 0u16);
        for v in &d.rgb {
            lo = (*v).min(lo);
            hi = (*v).max(hi);
        }
        // A real exposure covers a sane share of the range and never leaves it. The PowerShot
        // CR2 and the RX0 ARW are the documented exceptions: neither carries an as-shot white
        // balance we read, so the grey-world estimate brightens these scenes (the PowerShot's
        // own DNG, with its file WB, develops blacks to ~0 — reading the maker-note WB of
        // Canon and Sony is the follow-up).
        let estimated_wb = matches!(w.file, "IMG_4059.CR2" | "DSC00009.ARW");
        let black_cap: u16 = if estimated_wb { 16384 } else { 4096 };
        assert!(lo < black_cap, "{}: developed blacks at {lo}", w.file);
        assert!(hi > u16::MAX / 4, "{}: developed highlights at {hi}", w.file);
        // Deterministic: a second develop of the same sensor gives the same pixels.
        let d2 = photocraft_raw::develop_sensor(&s, &photocraft_raw::DevelopOptions::default()).unwrap();
        assert_eq!(d.rgb, d2.rgb, "{}", w.file);
    }
}

/// The E-5 file is the documented known-unsupported case: later Olympus bodies store packed
/// (compressed) sensor data. The error must say so, never panic, and the file still previews.
#[test]
fn packed_orf_reports_the_documented_gap() {
    let bytes = std::fs::read(pixls("_7061961_copy.ORF")).unwrap();
    let err = photocraft_raw::decode(&bytes, &Limits::default()).unwrap_err();
    assert!(matches!(err, RawError::Unsupported(ref m) if m.contains("Olympus compressed")), "{err}");
    assert!(photocraft_raw::embedded_preview(&bytes).is_some(), "the camera's JPEG preview is still reachable");
}
