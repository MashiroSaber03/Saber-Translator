//! Synthetic DNG files: every storage layout decodes to the exact sensor
//! samples, and development reproduces known colours.

use photocraft_raw::testgen::{DngSpec, DngStorage, Val, mosaic, scene};
use photocraft_raw::*;

fn sensor(bytes: &[u8]) -> Sensor {
    decode(bytes, &Limits::default()).unwrap()
}

/// Deterministic sensor-like samples below `2^bits`.
fn noise(n: usize, bits: u16) -> Vec<u16> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let max = (1u32 << bits) - 1;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((i as u32 * 13 + (s >> 40) as u32 % 200) % (max + 1)) as u16
        })
        .collect()
}

#[test]
fn identifies_dng() {
    let b = DngSpec::cfa(4, 4, vec![0; 16]).build();
    assert_eq!(identify(&b), Some(RawFormat::Dng));
    assert!(is_raw(&b));
    assert!(!is_raw(b"II*\0\x08\0\0\0\0\0"));
    assert!(!is_raw(b"\x89PNG\r\n\x1a\n"));
}

#[test]
fn storage_layouts_decode_exactly() {
    let (w, h) = (37, 23); // odd sizes exercise partial strips and edge tiles
    let cases: Vec<(&str, u16, DngStorage, bool)> = vec![
        ("16-bit strips", 16, DngStorage::Strips { rows: 5 }, false),
        ("16-bit big-endian", 16, DngStorage::Strips { rows: 23 }, true),
        ("12-bit packed", 12, DngStorage::Strips { rows: 4 }, false),
        ("10-bit packed big-endian", 10, DngStorage::Strips { rows: 7 }, true),
        ("8-bit", 8, DngStorage::Strips { rows: 9 }, false),
        ("LJ92 tiles", 14, DngStorage::Lj92Tiles { width: 16, height: 8 }, false),
        ("LJ92 tiles big-endian", 16, DngStorage::Lj92Tiles { width: 32, height: 32 }, true),
        ("LJ92 strips", 12, DngStorage::Lj92Strips { rows: 6 }, false),
    ];
    for (name, bits, storage, be) in cases {
        let data = noise(w * h, bits);
        let mut spec = DngSpec::cfa(w, h, data.clone());
        spec.bits = bits;
        spec.storage = storage;
        spec.big_endian = be;
        spec.white = (1 << bits) - 1;
        let s = sensor(&spec.build());
        assert_eq!((s.width, s.height, s.samples), (w, h, 1), "{name}");
        assert_eq!(s.data, data, "{name}");
        assert_eq!(s.format, RawFormat::Dng);
    }
}

#[test]
fn linear_raw_decodes_and_develops() {
    let (w, h) = (20, 10);
    let data = noise(w * h * 3, 16);
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.samples = 3;
    spec.storage = DngStorage::Lj92Tiles { width: 16, height: 16 };
    let s = sensor(&spec.build());
    assert_eq!((s.samples, s.cfa.is_none()), (3, true));
    assert_eq!(s.data, data);
    let d = develop_sensor(&s, &DevelopOptions::default()).unwrap();
    assert_eq!((d.width, d.height, d.rgb.len()), (20, 10, 600));
    assert!(d.info.cfa.is_none());
}

/// A horizontal ramp from `lo` to `hi` (inclusive), the same in every row.
fn ramp(w: usize, h: usize, lo: u16, hi: u16) -> Vec<u16> {
    (0..w * h).map(|i| lo + ((u32::from(hi - lo) * (i % w) as u32) / (w as u32 - 1)) as u16).collect()
}

/// Developed RGB of a monochrome DNG: every pixel has R = G = B, and the first row's grey level
/// rises with the ramp. Returns the first row's grey levels.
fn assert_neutral_ramp(d: &Developed, name: &str) -> Vec<u16> {
    let px = d.rgb.as_chunks::<3>().0;
    assert!(px.iter().all(|p| p[0] == p[1] && p[1] == p[2]), "{name}: channels differ");
    let row: Vec<u16> = px[..d.width as usize].iter().map(|p| p[0]).collect();
    assert!(row.windows(2).all(|w| w[1] >= w[0]), "{name}: not monotonic: {row:?}");
    assert!(row[row.len() - 1] > row[0] + 10_000, "{name}: flat: {row:?}");
    row
}

/// #3497: a one-sample LinearRaw DNG (VueScan film scans, Leica Monochrom) was rejected as
/// "LinearRaw data with 1 samples per pixel". It opens as neutral grey from every storage.
#[test]
fn monochrome_linear_raw_develops_to_neutral_grey() {
    let (w, h) = (37, 23);
    let cases: Vec<(&str, u16, DngStorage)> = vec![
        ("16-bit strips", 16, DngStorage::Strips { rows: 5 }),
        ("12-bit packed", 12, DngStorage::Strips { rows: 4 }),
        ("8-bit", 8, DngStorage::Strips { rows: 23 }),
        ("LJ92 strips", 14, DngStorage::Lj92Strips { rows: 6 }),
        ("LJ92 tiles", 16, DngStorage::Lj92Tiles { width: 16, height: 8 }),
        ("deflate predictor 2", 16, DngStorage::DeflateStrips { rows: 7, predictor: 2 }),
        // One whole tile: a partial edge tile is a separate decoder limitation.
        ("lossy JPEG tile", 16, DngStorage::LossyJpegTiles { width: 37, height: 23 }),
    ];
    for (name, bits, storage) in cases {
        let white = ((1u32 << bits) - 1) as u16;
        let data = ramp(w, h, 0, white);
        let mut spec = DngSpec::monochrome(w, h, data.clone());
        spec.bits = bits;
        spec.white = u32::from(white);
        spec.storage = storage;
        let lossy = matches!(storage, DngStorage::LossyJpegTiles { .. });
        let s = sensor(&spec.build());
        assert_eq!((s.samples, s.cfa.is_none(), s.is_monochrome()), (1, true, true), "{name}");
        if !lossy {
            assert_eq!(s.data, data, "{name}");
        }
        let d = develop_sensor(&s, &DevelopOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((d.width, d.height, d.rgb.len()), (37, 23, 37 * 23 * 3), "{name}");
        assert!(d.info.cfa.is_none() && d.info.wb_multipliers == [1.0; 3], "{name}: {:?}", d.info);
        assert!(d.warnings.iter().all(|w| !w.contains("colour") && !w.contains("white balance")), "{name}: {:?}", d.warnings);
        let row = assert_neutral_ramp(&d, name);
        if !lossy {
            assert_eq!((row[0], row[row.len() - 1]), (0, 65535), "{name}");
        }
    }
}

/// Black and white levels of a monochrome DNG scale its one sample; the white-balance
/// settings do not tint it.
#[test]
fn monochrome_linear_raw_levels_and_white_balance() {
    let (w, h) = (32, 4);
    let mut spec = DngSpec::monochrome(w, h, ramp(w, h, 1000, 60000));
    spec.black = vec![1000];
    spec.white = 60000;
    let s = sensor(&spec.build());
    for wb in [WhiteBalance::AsShot, WhiteBalance::Auto, WhiteBalance::Multipliers([2.0, 1.0, 1.5])] {
        let d = develop_sensor(&s, &DevelopOptions { white_balance: wb, ..Default::default() }).unwrap();
        let row = assert_neutral_ramp(&d, &format!("{wb:?}"));
        assert_eq!((row[0], row[w - 1]), (0, 65535), "{wb:?}");
    }
    // Develop exposure still applies.
    let dark = develop_sensor(&s, &DevelopOptions { exposure: -1.0, ..Default::default() }).unwrap();
    let row = assert_neutral_ramp(&dark, "exposure -1");
    assert!(row[w - 1] < 60000, "{row:?}");
}

/// Colour metadata a monochrome DNG may carry, well-formed or not (a 1×3 ColorMatrix, a
/// one-value ColorMatrix or AsShotNeutral, BlackLevel counts that fit no repeat), is ignored
/// or falls back; it never errors or indexes out of bounds.
#[test]
fn monochrome_linear_raw_odd_metadata_develops() {
    let (w, h) = (16, 6);
    let base = DngSpec::monochrome(w, h, ramp(w, h, 100, 4000));
    let rat = |v: &[f64]| Val::SRational(v.iter().map(|x| ((x * 10000.0) as i32, 10000)).collect());
    let variants: Vec<(&str, DngSpec)> = vec![
        ("1x3 ColorMatrix", DngSpec { ifd0_extra: vec![(50721, rat(&[0.3, 0.6, 0.1])), (50778, Val::Short(vec![21]))], ..base.clone() }),
        ("one-value ColorMatrix", DngSpec { ifd0_extra: vec![(50721, rat(&[1.0]))], ..base.clone() }),
        ("one-value AsShotNeutral", DngSpec { ifd0_extra: vec![(50728, Val::Rational(vec![(1, 1)]))], ..base.clone() }),
        ("two BlackLevels", DngSpec { black: vec![100, 120], ..base.clone() }),
        ("three BlackLevels", DngSpec { black: vec![100, 110, 120], ..base.clone() }),
        ("2x2 BlackLevel repeat", DngSpec { black: vec![100, 101, 102, 103], black_repeat: (2, 2), ..base.clone() }),
        ("2x2 repeat, one value", DngSpec { black: vec![100], black_repeat: (2, 2), ..base.clone() }),
        ("3x3 ColorMatrix", DngSpec { color_matrix1: Some((21, [3.2, -1.5, -0.5, -1.0, 1.9, 0.04, 0.06, -0.2, 1.06])), ..base.clone() }),
        ("3-value AsShotNeutral", DngSpec { as_shot_neutral: Some([0.5, 1.0, 0.7]), ..base.clone() }),
    ];
    for (name, spec) in variants {
        let d = develop(&spec.build(), &DevelopOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_neutral_ramp(&d, name);
    }
}

/// Broken monochrome DNGs fail with an error, not a panic.
#[test]
fn broken_monochrome_linear_raw_errors() {
    let (w, h) = (16, 6);
    let spec = DngSpec::monochrome(w, h, ramp(w, h, 100, 4000));
    // Every truncation decodes to an error or develops; none panics, and a cut into the
    // strip data is an error.
    let b = spec.build();
    for n in 0..b.len() {
        if let Ok(s) = decode(&b[..n], &Limits::default()) {
            let _ = develop_sensor(&s, &DevelopOptions::default());
        }
    }
    let first: Vec<u8> = spec.data[..4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let strip = b.windows(first.len()).position(|win| win == first.as_slice()).expect("strip data");
    assert!(decode(&b[..strip + w], &Limits::default()).is_err(), "truncated strip");
    // Two samples per pixel is no LinearRaw layout.
    let two = DngSpec { samples: 2, data: ramp(2 * w, h, 100, 4000), ..spec.clone() };
    let err = decode(&two.build(), &Limits::default()).unwrap_err().to_string();
    assert!(err.contains("2 samples per pixel"), "{err}");
    // A monochrome sensor holding less data than its size.
    let mut s = sensor(&spec.build());
    s.data.truncate(10);
    assert!(develop_sensor(&s, &DevelopOptions::default()).is_err());
}

/// `spec`'s file with its Orientation entry (a SHORT `from`) rewritten as a LONG `to`.
fn with_long_orientation(spec: &DngSpec, from: u16, to: u32) -> Vec<u8> {
    let mut b = spec.build();
    let short = [&[0x12, 0x01, 3, 0, 1, 0, 0, 0][..], &from.to_le_bytes(), &[0, 0]].concat();
    let at: Vec<usize> = b.windows(short.len()).enumerate().filter(|(_, w)| *w == short.as_slice()).map(|(i, _)| i).collect();
    assert_eq!(at.len(), 1, "one Orientation entry");
    let long = [&[0x12, 0x01, 4, 0, 1, 0, 0, 0][..], &to.to_le_bytes()].concat();
    b[at[0]..at[0] + long.len()].copy_from_slice(&long);
    b
}

#[test]
fn an_orientation_past_u16_is_not_truncated_into_range() {
    // #1817: 65538 truncated to 2 and rotated the image; it is out of range, so 1.
    let mut spec = DngSpec::cfa(8, 6, vec![500; 48]);
    spec.orientation = 2;
    assert_eq!(sensor(&with_long_orientation(&spec, 2, 65_538)).orientation, 1);
    // Control: a LONG orientation in range is read.
    assert_eq!(sensor(&with_long_orientation(&spec, 2, 6)).orientation, 6);
}

#[test]
fn metadata_is_read() {
    let (w, h) = (40, 30);
    let mut spec = DngSpec::cfa(w, h, vec![1000; w * h]);
    spec.bits = 14;
    spec.white = 15000;
    spec.black = vec![100, 101, 102, 103];
    spec.black_repeat = (2, 2);
    spec.active_area = Some([2, 4, 28, 36]); // top, left, bottom, right
    spec.default_crop = Some(([2, 1], [20, 16]));
    spec.cfa = [1, 0, 2, 1]; // GRBG at the active-area origin
    spec.as_shot_neutral = Some([0.5, 1.0, 0.8]);
    spec.baseline_exposure = Some(0.5);
    spec.orientation = 6;
    spec.linearization = Some((0..4096u32).map(|v| (v * 3).min(65535) as u16).collect());
    let s = sensor(&spec.build());
    assert_eq!(s.white, [15000.0; 3]);
    assert_eq!(s.black.values, vec![100.0, 101.0, 102.0, 103.0]);
    assert_eq!(s.active, Rect::new(4, 2, 32, 26));
    assert_eq!(s.crop, Rect::new(6, 3, 20, 16));
    // Pattern anchored at the active area origin (4, 2): GRBG there.
    assert_eq!(s.cfa.as_ref().unwrap().phase(4, 2), [1, 0, 2, 1]);
    assert_eq!(s.color.as_shot_neutral, Some([0.5, 1.0, 0.8]));
    assert!((s.baseline_exposure - 0.5).abs() < 1e-6);
    assert_eq!(s.orientation, 6);
    assert_eq!(s.linearization.as_ref().map(Vec::len), Some(4096));
    let d = develop_sensor(&s, &DevelopOptions::default()).unwrap();
    // Orientation 6 swaps the cropped 20×16 to 16×20.
    assert_eq!((d.width, d.height), (16, 20));
    assert_eq!(d.info.wb_multipliers, [2.0, 1.0, 1.25]);
    let d = develop_sensor(&s, &DevelopOptions { orient: false, ..Default::default() }).unwrap();
    assert_eq!((d.width, d.height), (20, 16));
}

#[test]
fn enormous_double_crop_origin_is_rejected() {
    let (w, h) = (16, 12);
    let mut spec = DngSpec::cfa(w, h, vec![1000; w * h]);
    spec.active_area = Some([2, 3, 10, 14]);
    spec.default_crop = Some(([0, 0], [11, 8]));
    spec.default_crop_origin_double = Some([1e300, 0.0]);

    assert!(matches!(decode(&spec.build(), &Limits::default()), Err(RawError::Malformed(_))));
}

#[test]
fn out_of_range_default_crop_falls_back_to_active_area_with_warning() {
    let (w, h) = (16, 12);
    let mut spec = DngSpec::cfa(w, h, vec![1000; w * h]);
    spec.active_area = Some([2, 3, 10, 14]);
    spec.default_crop = Some(([10, 0], [5, 4]));

    let s = sensor(&spec.build());
    assert_eq!(s.crop, s.active);
    assert!(s.warnings.iter().any(|warning| warning.contains("out-of-range") && warning.contains("default crop")), "{:?}", s.warnings);
}

/// DefaultCropSize defaults to the whole image, which can't fit once the origin moves in.
#[test]
fn default_crop_origin_without_size_falls_back_to_active_area_with_warning() {
    let (w, h) = (16, 12);
    let mut spec = DngSpec::cfa(w, h, vec![1000; w * h]);
    spec.active_area = Some([2, 3, 10, 14]);
    spec.default_crop_origin_double = Some([1.0, 1.0]);

    let s = sensor(&spec.build());
    assert_eq!(s.crop, s.active);
    assert!(s.warnings.iter().any(|warning| warning.contains("out-of-range") && warning.contains("default crop")), "{:?}", s.warnings);
}

/// #950: DefaultCropOrigin defaults to (0, 0), so a size-only crop starts at the active area's
/// top-left corner.
#[test]
fn default_crop_size_without_origin_crops_from_the_active_area_origin() {
    let (w, h) = (8, 8);
    let mut spec = DngSpec::cfa(w, h, (0..w * h).map(|i| (i as u16) * 37).collect());
    spec.active_area = Some([0, 0, 8, 8]);
    spec.default_crop_size_only = Some([4, 4]);
    let bytes = spec.build();

    let s = sensor(&bytes);
    assert_eq!(s.crop, Rect::new(0, 0, 4, 4));
    assert!(s.warnings.iter().all(|warning| !warning.contains("default crop")), "{:?}", s.warnings);
    let d = develop(&bytes, &DevelopOptions::default()).unwrap();
    assert_eq!((d.width, d.height), (4, 4));

    // Inside an offset active area the crop starts at that area's corner.
    let (w, h) = (16, 12);
    let mut spec = DngSpec::cfa(w, h, vec![1000; w * h]);
    spec.active_area = Some([2, 3, 10, 14]);
    spec.default_crop_size_only = Some([6, 4]);
    let s = sensor(&spec.build());
    assert_eq!(s.crop, Rect::new(3, 2, 6, 4));
}

/// Linear sRGB (D65) → XYZ, from the sRGB primaries (IEC 61966-2-1).
const SRGB_TO_XYZ: [[f64; 3]; 3] = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];

fn invert(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
}

fn ap(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// The DNG model computed independently: camera XYZ of `camera` under an
/// illuminant whose camera neutral is `neutral`, Bradford-adapted from that
/// white (normalized to Y = 1) to D50, then XYZ D50 → linear ProPhoto.
fn reference_prophoto(camera: [f64; 3], neutral: [f64; 3]) -> [f64; 3] {
    const BRADFORD: [[f64; 3]; 3] = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
    const D50: [f64; 3] = [0.9642, 1.0, 0.8249];
    const XYZ_D50_TO_PROPHOTO: [[f64; 3]; 3] = [[1.3459433, -0.2556075, -0.0511118], [-0.5445989, 1.5081673, 0.0205351], [0.0, 0.0, 1.2118128]];
    let white = ap(&SRGB_TO_XYZ, neutral);
    let white = white.map(|v| v / white[1]);
    let xyz = ap(&SRGB_TO_XYZ, camera).map(|v| v / ap(&SRGB_TO_XYZ, neutral)[1]);
    let (s, d) = (ap(&BRADFORD, white), ap(&BRADFORD, D50));
    let cone = ap(&BRADFORD, xyz);
    let adapted = ap(&invert(BRADFORD), [cone[0] * d[0] / s[0], cone[1] * d[1] / s[1], cone[2] * d[2] / s[2]]);
    ap(&XYZ_D50_TO_PROPHOTO, adapted)
}

#[test]
fn colour_matrix_development_matches_reference() {
    // A camera whose native RGB is linear sRGB: ColorMatrix = XYZ → sRGB at D65.
    // The scene is captured with per-channel gains (the "white balance" the camera
    // must undo), recorded as AsShotNeutral.
    let (w, h) = (48, 32);
    let gains = [0.5f32, 1.0, 0.7];
    let patches: [[f32; 3]; 4] = [[0.18, 0.18, 0.18], [0.40, 0.10, 0.05], [0.05, 0.30, 0.10], [0.10, 0.12, 0.45]];
    let rgb: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let p = patches[((i % w) / (w / 2)) + 2 * ((i / w) / (h / 2))];
            [p[0] * gains[0], p[1] * gains[1], p[2] * gains[2]]
        })
        .collect();
    let mut spec = DngSpec::cfa(w, h, mosaic(&rgb, w, [0, 1, 1, 2], 256, 16383));
    spec.bits = 14;
    spec.black = vec![256];
    spec.white = 16383;
    let cm = invert(SRGB_TO_XYZ);
    spec.color_matrix1 = Some((21, [cm[0][0], cm[0][1], cm[0][2], cm[1][0], cm[1][1], cm[1][2], cm[2][0], cm[2][1], cm[2][2]]));
    spec.as_shot_neutral = Some(gains.map(f64::from));
    for storage in [DngStorage::Strips { rows: 32 }, DngStorage::Lj92Tiles { width: 16, height: 16 }] {
        spec.storage = storage;
        for method in [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd] {
            let d = develop(&spec.build(), &DevelopOptions { demosaic: method, ..Default::default() }).unwrap();
            for (pi, p) in patches.iter().enumerate() {
                // Centre of each patch.
                let (cx, cy) = ((pi % 2) * (w / 2) + w / 4, (pi / 2) * (h / 2) + h / 4);
                let o = &d.rgb[(cy * w + cx) * 3..(cy * w + cx) * 3 + 3];
                let g = gains.map(f64::from);
                let want = reference_prophoto([0, 1, 2].map(|c| f64::from(p[c]) * g[c]), g);
                for c in 0..3 {
                    let got = (f64::from(o[c]) / 65535.0).powf(1.8);
                    assert!((got - want[c]).abs() < 0.006, "{storage:?} {method:?} patch {pi} c {c}: got {got:.4} want {:.4}", want[c]);
                }
            }
        }
    }
}

#[test]
fn forward_matrix_path_maps_neutral_to_white() {
    let (w, h) = (16, 16);
    let rgb = vec![[0.3f32, 0.6, 0.45]; w * h]; // a grey under a green-ish light
    let mut spec = DngSpec::cfa(w, h, mosaic(&rgb, w, [0, 1, 1, 2], 0, 65535));
    let cm = invert(SRGB_TO_XYZ);
    let flat = |m: [[f64; 3]; 3]| [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]];
    spec.color_matrix1 = Some((17, flat(cm)));
    spec.color_matrix2 = Some((21, flat(cm)));
    // ForwardMatrix: balanced camera → XYZ D50 (sRGB adapted to D50; white → D50).
    let fm = [[0.4360747, 0.3850649, 0.1430804], [0.2225045, 0.7168786, 0.0606169], [0.0139322, 0.0971045, 0.7141733]];
    spec.forward_matrix1 = Some(flat(fm));
    spec.forward_matrix2 = Some(flat(fm));
    spec.as_shot_neutral = Some([0.5, 1.0, 0.75]);
    let d = develop(&spec.build(), &DevelopOptions::default()).unwrap();
    let o = &d.rgb[(8 * w + 8) * 3..(8 * w + 8) * 3 + 3];
    // Balanced grey 0.6 → ProPhoto (0.6, 0.6, 0.6).
    for (c, v) in o.iter().enumerate() {
        let got = (f64::from(*v) / 65535.0).powf(1.8);
        assert!((got - 0.6).abs() < 0.004, "c {c}: {got}");
    }
}

#[test]
fn highlights_clip_to_white_and_exposure_applies() {
    let (w, h) = (8, 8);
    let mut spec = DngSpec::cfa(w, h, vec![65535; w * h]);
    spec.as_shot_neutral = Some([0.5, 1.0, 0.6]);
    let d = develop(&spec.build(), &DevelopOptions::default()).unwrap();
    assert!(d.rgb.iter().all(|&v| v >= 65500), "clipped sensor must render white");
    let mut spec = DngSpec::cfa(w, h, vec![16384; w * h]); // 0.25 linear grey
    spec.as_shot_neutral = Some([1.0, 1.0, 1.0]);
    let b = spec.build();
    let base = develop(&b, &DevelopOptions::default()).unwrap().rgb[27];
    let up = develop(&b, &DevelopOptions { exposure: 1.0, ..Default::default() }).unwrap().rgb[27];
    let lin = |v: u16| (f64::from(v) / 65535.0).powf(1.8);
    assert!((lin(up) / lin(base) - 2.0).abs() < 0.02, "{} vs {}", lin(up), lin(base));
}

/// storytold/photocraft#4348 (as storytold/lightcraft#548): a 12-bit DNG that states WhiteLevel 4095
/// but whose sensor saturates at 3567 in every colour. Under the camera's white balance (red x2.46,
/// blue x1.59) the stuck red and blue samples clip at 1 while green stays at 0.85, so a blown wall
/// rendered magenta. The wall must come out neutral, and the unclipped ramp beside it unchanged.
#[test]
fn highlights_that_clip_below_the_stated_white_level_stay_neutral() {
    let (w, h) = (64, 32);
    let data: Vec<u16> = (0..w * h).map(|i| if i % w < w / 2 { 528 + ((i % w) * 90 + (i / w) * 3) as u16 } else { 3567 }).collect();
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.bits = 12;
    spec.black = vec![528];
    spec.white = 4095;
    spec.as_shot_neutral = Some([1.0 / 2.46, 1.0, 1.0 / 1.59]);
    let b = spec.build();
    let d = develop(&b, &DevelopOptions::default()).unwrap();
    let px = &d.rgb[(16 * w + 48) * 3..(16 * w + 48) * 3 + 3];
    let lin = |v: u16| (f64::from(v) / 65535.0).powf(1.8);
    let tint = (lin(px[0]) / lin(px[1]) - 1.0).abs().max((lin(px[2]) / lin(px[1]) - 1.0).abs());
    assert!(tint < 0.01, "the clipped wall is tinted: {px:?}");
    let s = sensor(&b);
    for (i, (&got, &was)) in s.data.iter().zip(&data).enumerate() {
        assert_eq!(got, if was == 3567 { 4095 } else { was }, "sample {i}");
    }
}

#[test]
fn gain_map_opcode_is_applied() {
    // Uniform 0.25 grey; a gain map that doubles everything over the whole image.
    let (w, h) = (16, 12);
    let mut spec = DngSpec::cfa(w, h, vec![16384; w * h]);
    spec.as_shot_neutral = Some([1.0; 3]);
    let lin = |v: u16| (f64::from(v) / 65535.0).powf(1.8);
    let base = develop(&spec.build(), &DevelopOptions::default()).unwrap().rgb[(6 * w + 8) * 3 + 1];
    spec.opcode_list2 = Some(photocraft_raw::testgen::gain_map_opcode_list([0, 0, h as u32, w as u32], 1, [[2.0, 2.0], [2.0, 2.0]]));
    let d = develop(&spec.build(), &DevelopOptions::default()).unwrap();
    assert!(!d.warnings.iter().any(|w| w.contains("Opcode")), "{:?}", d.warnings);
    let gained = d.rgb[(6 * w + 8) * 3 + 1];
    assert!((lin(gained) / lin(base) - 2.0).abs() < 0.02, "{} vs {}", lin(gained), lin(base));
}

#[test]
fn white_balance_options() {
    let (w, h) = (16, 16);
    let rgb = vec![[0.2f32, 0.4, 0.3]; w * h];
    let spec = DngSpec::cfa(w, h, mosaic(&rgb, w, [0, 1, 1, 2], 0, 65535));
    let b = spec.build();
    // No AsShotNeutral: grey world → neutral output.
    let d = develop(&b, &DevelopOptions::default()).unwrap();
    assert!(d.warnings.iter().any(|w| w.contains("grey world")));
    let o = &d.rgb[(8 * w + 8) * 3..(8 * w + 8) * 3 + 3];
    assert!((i32::from(o[0]) - i32::from(o[1])).abs() < 300 && (i32::from(o[2]) - i32::from(o[1])).abs() < 300, "{o:?}");
    let d = develop(&b, &DevelopOptions { white_balance: WhiteBalance::Multipliers([2.0, 1.0, 4.0 / 3.0]), ..Default::default() }).unwrap();
    assert_eq!(d.info.wb_multipliers.map(|v| (v * 1000.0).round()), [2000.0, 1000.0, 1333.0]);
    assert!(develop(&b, &DevelopOptions { white_balance: WhiteBalance::Multipliers([0.0, 1.0, 1.0]), ..Default::default() }).is_err());
}

#[test]
fn smooth_scene_develops_with_every_method() {
    let (w, h) = (64, 48);
    let rgb = scene(w, h);
    let spec = DngSpec::cfa(w, h, mosaic(&rgb, w, [2, 1, 1, 0], 0, 65535));
    let mut spec = spec;
    spec.cfa = [2, 1, 1, 0];
    spec.as_shot_neutral = Some([1.0; 3]);
    let b = spec.build();
    let outs: Vec<Vec<u16>> = [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd]
        .iter()
        .map(|&m| develop(&b, &DevelopOptions { demosaic: m, ..Default::default() }).unwrap().rgb)
        .collect();
    // Interior pixels agree closely across methods on a smooth scene.
    for y in 4..h - 4 {
        for x in 4..w - 4 {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                let (a, b, d) = (i32::from(outs[0][i]), i32::from(outs[1][i]), i32::from(outs[2][i]));
                assert!((a - b).abs() < 800 && (a - d).abs() < 800, "({x},{y}) c{c}: {a} {b} {d}");
            }
        }
    }
}

#[test]
fn unsupported_variants_say_so() {
    let mut spec = DngSpec::cfa(8, 8, vec![0; 64]);
    spec.cfa = [0, 1, 2, 1]; // not Bayer
    assert!(matches!(decode(&spec.build(), &Limits::default()), Err(RawError::Unsupported(_))));
}

#[test]
fn limits_are_enforced_before_allocating() {
    let spec = DngSpec::cfa(64, 64, vec![0; 64 * 64]);
    let b = spec.build();
    let tiny = Limits { max_width: 32, ..Limits::default() };
    assert!(matches!(decode(&b, &tiny), Err(RawError::LimitExceeded(_))));
    let tiny = Limits { max_alloc: 1000, ..Limits::default() };
    assert!(matches!(decode(&b, &tiny), Err(RawError::LimitExceeded(_))));
}

/// Adobe DNG compression 8 (zlib): with and without the horizontal-differencing predictor,
/// with and without 12-bit packing, and both byte orders — each decodes to the exact samples.
#[test]
fn deflate_strips_round_trip() {
    let (w, h) = (48, 40);
    for (name, bits, storage, be) in [
        ("deflate 16-bit", 16, DngStorage::DeflateStrips { rows: 16, predictor: 1 }, false),
        ("deflate 16-bit predictor 2", 16, DngStorage::DeflateStrips { rows: 16, predictor: 2 }, false),
        ("deflate 12-bit predictor 2", 12, DngStorage::DeflateStrips { rows: 12, predictor: 2 }, false),
        ("deflate 16-bit predictor 2 BE", 16, DngStorage::DeflateStrips { rows: 16, predictor: 2 }, true),
        ("deflate LinearRaw predictor 2", 16, DngStorage::DeflateStrips { rows: 40, predictor: 2 }, false),
    ] {
        let samples = if name.contains("LinearRaw") { 3 } else { 1 };
        let data = noise(w * h * samples, bits);
        let mut spec = DngSpec::cfa(w, h, data.clone());
        spec.samples = samples;
        spec.bits = bits;
        spec.storage = storage;
        spec.big_endian = be;
        spec.white = (1 << bits) - 1;
        let s = sensor(&spec.build());
        assert_eq!((s.width, s.height, s.samples), (w, h, samples), "{name}");
        assert_eq!(s.data, data, "{name}");
    }
}

/// A zlib stream cut short must fail cleanly (decompression comes up short or errors), and a
/// declared-but-absent giant tile must trip the limits instead of allocating.
#[test]
fn deflate_hostile_fails_cleanly() {
    let (w, h) = (32, 32);
    let data = noise(w * h, 16);
    let mut spec = DngSpec::cfa(w, h, data);
    spec.storage = DngStorage::DeflateStrips { rows: 32, predictor: 2 };
    let bytes = spec.build();
    assert_eq!(sensor(&bytes).data.len(), w * h);
    // Locate the zlib stream (78 9c header of the single strip) and cut it mid-stream.
    let at = bytes.windows(2).position(|w| w == [0x78, 0x9c]).expect("zlib header") + 2;
    let cut = &bytes[..at + (bytes.len() - at) / 2];
    assert!(decode(cut, &Limits::default()).is_err(), "a cut zlib stream must not decode");
}

/// Predictor 2 on 8-bit samples wraps at 8 bits, and 12-bit samples bit-packed inside the zlib
/// stream (not in 16-bit containers) decode too.
#[test]
fn deflate_8_bit_predictor_and_packed_12_bit() {
    use std::io::Write as _;
    let (w, h) = (16usize, 4usize);
    let zlib = |bytes: &[u8]| {
        let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(bytes).unwrap();
        e.finish().unwrap()
    };
    // 8-bit, predictor 2: a row falling from 250 to 10 has deltas that wrap mod 256.
    let data8: Vec<u16> = (0..w * h).map(|i| if i % w < w / 2 { 250 } else { 10 }).collect();
    let mut spec = DngSpec::cfa(w, h, data8.clone());
    spec.bits = 8;
    spec.white = 255;
    spec.storage = DngStorage::DeflateStrips { rows: h, predictor: 2 };
    assert_eq!(sensor(&spec.build()).data, data8, "8-bit predictor 2");
    // 12-bit packed: two samples in three bytes, MSB first. Swap the testgen's 16-bit
    // containers for a packed stream of the same samples.
    let data12: Vec<u16> = (0..w * h).map(|i| (i as u16 * 61) & 0xFFF).collect();
    let mut spec = DngSpec::cfa(w, h, data12.clone());
    spec.bits = 12;
    spec.white = 4095;
    spec.storage = DngStorage::DeflateStrips { rows: h, predictor: 1 };
    let mut bytes = spec.build();
    let containers: Vec<u8> = data12.iter().flat_map(|v| v.to_le_bytes()).collect();
    let old = zlib(&containers);
    let packed: Vec<u8> = data12.chunks(2).flat_map(|p| [(p[0] >> 4) as u8, ((p[0] & 0xF) << 4 | p[1] >> 8) as u8, p[1] as u8]).collect();
    let mut new = zlib(&packed);
    let at = bytes.windows(old.len()).position(|c| c == old.as_slice()).expect("the strip");
    assert!(new.len() <= old.len());
    new.resize(old.len(), 0); // trailing bytes after the zlib stream are ignored
    bytes[at..at + old.len()].copy_from_slice(&new);
    assert_eq!(sensor(&bytes).data, data12, "12-bit packed");
}

// ---------------------------------------------------------------- JPEG XL (DNG 1.7)

/// Losslessly encodes `w × h` samples (1 = grey, 3 = RGB) at 8 or 16 bits as a JPEG XL codestream.
fn jxl(samples: &[u16], w: usize, h: usize, channels: usize, bits: u16) -> Vec<u8> {
    use zune_core::{bit_depth::BitDepth, colorspace::ColorSpace, options::EncoderOptions};
    let (depth, bytes): (BitDepth, Vec<u8>) = match bits {
        8 => (BitDepth::Eight, samples.iter().map(|&v| v as u8).collect()),
        _ => (BitDepth::Sixteen, samples.iter().flat_map(|v| v.to_ne_bytes()).collect()),
    };
    let space = if channels == 1 { ColorSpace::Luma } else { ColorSpace::RGB };
    let mut out = Vec::new();
    zune_jpegxl::JxlSimpleEncoder::new(&bytes, EncoderOptions::new(w, h, space, depth)).encode(&mut out).unwrap();
    out
}

/// `data` (`w × h × spp`) cut into `tw × th` tiles, edge tiles padded by repeating the last row / column.
fn tiles(data: &[u16], w: usize, h: usize, spp: usize, tw: usize, th: usize) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    for ty in (0..h).step_by(th) {
        for tx in (0..w).step_by(tw) {
            let mut t = Vec::with_capacity(tw * th * spp);
            for y in 0..th {
                for x in 0..tw {
                    let at = ((ty + y).min(h - 1) * w + (tx + x).min(w - 1)) * spp;
                    t.extend_from_slice(&data[at..at + spp]);
                }
            }
            out.push(t);
        }
    }
    out
}

#[test]
fn jpeg_xl_layouts_decode_exactly() {
    let (w, h) = (37, 23); // odd sizes: a short last strip and partial edge tiles
    for (name, spp, bits, tiled) in [
        ("LinearRaw 16-bit strips", 3, 16, false),
        ("LinearRaw 16-bit tiles", 3, 16, true),
        ("CFA 16-bit strips", 1, 16, false),
        ("CFA 16-bit tiles", 1, 16, true),
        ("LinearRaw 8-bit strips", 3, 8, false),
        ("CFA 8-bit tiles", 1, 8, true),
    ] {
        let data = noise(w * h * spp, bits);
        let mut spec = DngSpec::cfa(w, h, data.clone());
        spec.samples = spp;
        spec.bits = bits;
        spec.white = (1u32 << bits) - 1;
        if tiled {
            let (tw, th) = (16, 16);
            spec.storage = DngStorage::JxlTiles { width: tw, height: th };
            spec.jxl_segments = tiles(&data, w, h, spp, tw, th).iter().map(|t| jxl(t, tw, th, spp, bits)).collect();
        } else {
            let rows = 10;
            spec.storage = DngStorage::JxlStrips { rows };
            spec.jxl_segments =
                (0..h).step_by(rows).map(|y| (y, (y + rows).min(h) - y)).map(|(y, r)| jxl(&data[y * w * spp..(y + r) * w * spp], w, r, spp, bits)).collect();
        }
        let s = sensor(&spec.build());
        assert_eq!((s.width, s.height, s.samples), (w, h, spp), "{name}");
        assert_eq!(s.data, data, "{name}");
    }
}

#[test]
fn jpeg_xl_linear_raw_develops() {
    // A Samsung Expert RAW-like file: one 16-bit LinearRaw JPEG XL strip with a baseline exposure.
    let (w, h) = (24, 16);
    let data: Vec<u16> = scene(w, h).iter().flat_map(|p| p.map(|v| (v * 20000.0) as u16)).collect();
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.samples = 3;
    spec.storage = DngStorage::JxlStrips { rows: h };
    spec.jxl_segments = vec![jxl(&data, w, h, 3, 16)];
    spec.baseline_exposure = Some(1.0);
    let b = spec.build();
    let d = develop(&b, &DevelopOptions::default()).unwrap();
    assert_eq!((d.width as usize, d.height as usize, d.rgb.len()), (w, h, w * h * 3));
    // Same pixels stored uncompressed develop identically.
    spec.storage = DngStorage::Strips { rows: h };
    assert_eq!(develop(&spec.build(), &DevelopOptions::default()).unwrap().rgb, d.rgb);
}

#[test]
fn jpeg_xl_mismatches_are_errors() {
    let (w, h) = (16, 8);
    let data = noise(w * h * 3, 16);
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.samples = 3;
    spec.storage = DngStorage::JxlStrips { rows: h };
    // Codestream smaller than the strip.
    spec.jxl_segments = vec![jxl(&data[..w * 4 * 3], w, 4, 3, 16)];
    assert!(matches!(decode(&spec.build(), &Limits::default()), Err(RawError::Malformed(_))));
    // Grey codestream for RGB samples.
    spec.jxl_segments = vec![jxl(&data[..w * h], w, h, 1, 16)];
    assert!(matches!(decode(&spec.build(), &Limits::default()), Err(RawError::Unsupported(_))));
    // Not a codestream at all.
    spec.jxl_segments = vec![vec![0xff, 0x0a, 1, 2, 3, 4, 5]];
    assert!(matches!(decode(&spec.build(), &Limits::default()), Err(RawError::Malformed(_))));
}

#[test]
fn profile_gain_table_map_is_reported() {
    let gain_table = |s: &Sensor| s.warnings.iter().any(|w| w.contains("ProfileGainTableMap"));
    let mut spec = DngSpec::cfa(8, 8, vec![500; 64]);
    assert!(!gain_table(&sensor(&spec.build())));
    spec.profile_gain_table_map = Some(vec![0; 64]);
    assert!(gain_table(&sensor(&spec.build())));
}

/// Adobe DNG compression 34892 (lossy baseline JPEG, 8-bit luma tiles): decodes to the tile
/// samples widened to 16 bits, within the JPEG quality band (Q60 artifacts are mild but real).
#[test]
fn lossy_jpeg_tiles_decode() {
    let (w, h) = (32, 32);
    // A smooth ramp: baseline JPEG at Q60 keeps it within a few quantisation steps.
    let data: Vec<u16> = (0..w * h).map(|i| 4096 + (i / w) as u16 * 16).collect();
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.storage = DngStorage::LossyJpegTiles { width: 16, height: 16 };
    let s = sensor(&spec.build());
    assert_eq!((s.width, s.height, s.samples), (w, h, 1));
    let mut worst = 0u32;
    for (got, want) in s.data.iter().zip(&data) {
        // The tile was written 8-bit (>>8), so the quantised expectation is (want>>8)<<8.
        let quant = u32::from(*want >> 8) << 8;
        worst = worst.max(u32::from(got.abs_diff(quant as u16)));
    }
    assert!(worst <= 4 * 256, "worst deviation from the 8-bit quantisation: {worst}");
}

/// A lossy JPEG tile cut mid-stream fails cleanly.
#[test]
fn lossy_jpeg_tiles_hostile() {
    let (w, h) = (32, 32);
    let data: Vec<u16> = (0..w * h).map(|_i| 8192u16).collect();
    let mut spec = DngSpec::cfa(w, h, data);
    spec.storage = DngStorage::LossyJpegTiles { width: 32, height: 32 };
    let bytes = spec.build();
    let at = bytes.windows(2).position(|w| w == [0xFF, 0xD8]).expect("jpeg SOI") + 2;
    let cut = &bytes[..at + (bytes.len() - at) / 2];
    assert!(decode(cut, &Limits::default()).is_err(), "a cut JPEG tile must not decode");
}

/// A tile JPEG whose header lies about its size fails the header check, before any pixel
/// decode (a hostile SOF must not be able to size the decoder's buffers first).
#[test]
fn lossy_jpeg_tile_dimension_mismatch_errors() {
    use image::ImageEncoder as _;
    let (w, h) = (32, 32);
    let data: Vec<u16> = (0..w * h).map(|i| 4096 + (i / w) as u16 * 16).collect();
    let mut spec = DngSpec::cfa(w, h, data);
    spec.storage = DngStorage::LossyJpegTiles { width: 32, height: 32 };
    let mut bytes = spec.build();
    // Swap the 32x32 tile JPEG for a 16x16 one, inside the same segment (the trailing bytes
    // after its EOI stay; the JPEG decoder stops at EOI).
    let small = {
        let img = image::GrayImage::from_fn(16, 16, |x, _| image::Luma([(x as u16 * 16) as u8]));
        let mut j = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut j, 60);
        enc.write_image(img.as_raw(), 16, 16, image::ExtendedColorType::L8).unwrap();
        j
    };
    let soi = bytes.windows(3).position(|w| w == [0xFF, 0xD8, 0xFF]).expect("SOI");
    let eoi = soi + bytes[soi..].windows(2).position(|w| w == [0xFF, 0xD9]).expect("EOI") + 2;
    bytes.splice(soi..eoi, small);
    let err = decode(&bytes, &Limits::default()).unwrap_err().to_string();
    assert!(err.contains("16x16") && err.contains("32x32"), "{err}");
}

/// A 3-sample LinearRaw lossy DNG (YCbCr on disk, as real camera files are) decodes to RGB
/// tiles widened to 16 bits, within the JPEG quality band; a LinearizationTable maps the
/// decoded samples in the develop stage, so the sensor sees the raw widened values.
#[test]
fn lossy_jpeg_rgb_linearraw_decodes() {
    let (w, h) = (32, 32);
    // Channel offsets of 64 8-bit steps survive JPEG chroma quantisation at Q60.
    let data: Vec<u16> = (0..w * h * 3).map(|i| 4096 + ((i / (w * 3)) as u16 * 16) + (i % 3) as u16 * 64 * 256).collect();
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.samples = 3;
    spec.storage = DngStorage::LossyJpegTiles { width: 16, height: 16 };
    let s = sensor(&spec.build());
    assert_eq!((s.width, s.height, s.samples), (w, h, 3));
    let mut worst = 0u32;
    for (got, want) in s.data.iter().zip(&data) {
        let quant = u32::from(*want >> 8) << 8;
        worst = worst.max(u32::from(got.abs_diff(quant as u16)));
    }
    assert!(worst <= 8 * 256, "worst deviation from the 8-bit quantisation: {worst}");
    // Each channel carries its own level: R and B stay apart everywhere.
    for p in 0..w * h {
        let (r, b) = (s.data[p * 3], s.data[p * 3 + 2]);
        assert!(r.abs_diff(b) >= 64 * 256 - 8 * 256, "channels collapsed at pixel {p}");
    }
}
