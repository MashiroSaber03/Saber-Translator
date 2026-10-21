//! Synthetic Fujifilm RAF files: every sample storage, X-Trans and Bayer
//! layouts, metadata, and the unsupported variants.

use photocraft_raw::testgen::{RafPacking, RafSpec, XTRANS, mosaic_cfa, scene};
use photocraft_raw::*;

const RGGB: [u8; 4] = [0, 1, 1, 2];

/// A 30×18 X-Trans RAF with a 2-pixel masked border.
fn xtrans(packing: RafPacking, bits: u32) -> RafSpec {
    let (w, h) = (30, 18);
    let white = ((1u32 << bits) - 1) as u16;
    let mut s = RafSpec::new(w, h, mosaic_cfa(&scene(w, h), w, &XTRANS, 6, 256, white), XTRANS.to_vec());
    s.bits = bits;
    s.packing = packing;
    s.black = vec![256; 36];
    s.crop = Some([2, 2, 14, 26]);
    s
}

#[test]
fn every_storage_decodes_exactly() {
    for (packing, bits) in [(RafPacking::U16 { le: true }, 14), (RafPacking::U16 { le: false }, 16), (RafPacking::Lsb12, 12), (RafPacking::Words14, 14)] {
        let spec = xtrans(packing, bits);
        let b = spec.build();
        assert_eq!(identify(&b), Some(RawFormat::Raf));
        let s = decode(&b, &Limits::default()).unwrap_or_else(|e| panic!("{packing:?}: {e}"));
        assert_eq!((s.width, s.height), (30, 18));
        assert_eq!(s.data, spec.data, "{packing:?}");
    }
}

#[test]
fn xtrans_metadata() {
    let mut spec = xtrans(RafPacking::U16 { le: true }, 14);
    spec.exposure_bias = Some((-72, 100));
    spec.orientation = 6;
    let s = decode(&spec.build(), &Limits::default()).unwrap();
    let cfa = s.cfa.as_ref().unwrap();
    assert_eq!((cfa.width, cfa.height), (6, 6));
    for (i, &c) in XTRANS.iter().enumerate() {
        assert_eq!(cfa.color(i % 6, i / 6), c, "position {i}");
    }
    assert!(!cfa.is_bayer());
    assert_eq!(s.crop, Rect::new(2, 2, 26, 14));
    assert_eq!(s.black, BlackLevels::uniform(256.0));
    assert_eq!(s.camera_wb, Some([2.0, 1.0, 1.5]));
    assert!((s.baseline_exposure - 0.72).abs() < 1e-9);
    assert_eq!(s.orientation, 6);
    assert_eq!((s.make.as_deref(), s.model.as_deref()), (Some("FUJIFILM"), Some("X-Synthetic")));
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
}

#[test]
fn xtrans_develops() {
    let mut spec = xtrans(RafPacking::U16 { le: true }, 14);
    spec.orientation = 6;
    let b = spec.build();
    for m in [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd] {
        let d = develop(&b, &DevelopOptions { demosaic: m, ..Default::default() }).unwrap();
        // Cropped to 26×14, then rotated.
        assert_eq!((d.width, d.height), (14, 26));
        assert_eq!(d.info.format, RawFormat::Raf);
        // The repeat seen from the crop origin (2, 2).
        assert_eq!(d.info.cfa.as_deref(), Some("GRBGBR/BGGRGG/RGGBGG/GBRGRB/RGGBGG/BGGRGG"));
        assert_eq!(d.info.wb_multipliers, [2.0, 1.0, 1.5]);
    }
}

#[test]
fn xtrans_white_balance_is_per_site() {
    // A neutral grey recorded through WB gains R ×2, B ×1.5: developed with the
    // as-shot balance it must come out neutral at every X-Trans site (a gain
    // applied per column pair, as for Bayer, would tint it).
    let (w, h) = (36, 24);
    let rgb: Vec<[f32; 3]> = vec![[0.4 / 2.0, 0.4, 0.4 / 1.5]; w * h];
    let spec = RafSpec { black: vec![256], ..RafSpec::new(w, h, mosaic_cfa(&rgb, w, &XTRANS, 6, 256, 16383), XTRANS.to_vec()) };
    let d = develop(&spec.build(), &DevelopOptions::default()).unwrap();
    for p in d.rgb.as_chunks::<3>().0 {
        assert!(p[0].abs_diff(p[1]) <= 2 && p[2].abs_diff(p[1]) <= 2, "{p:?}");
    }
}

#[test]
fn bayer_layouts() {
    let (w, h) = (24, 16);
    for cfa in [RGGB, [2, 1, 1, 0]] {
        for (packing, bits) in [(RafPacking::Lsb12, 12), (RafPacking::Words14, 14), (RafPacking::U16 { le: true }, 14)] {
            let white = ((1u32 << bits) - 1) as u16;
            let data = mosaic_cfa(&scene(w, h), w, &cfa, 2, 64, white);
            let spec = RafSpec { bits, packing, black: vec![64, 64, 64, 64], ..RafSpec::new(w, h, data, cfa.to_vec()) };
            let s = decode(&spec.build(), &Limits::default()).unwrap();
            assert_eq!(s.data, spec.data);
            assert_eq!(s.cfa.as_ref().unwrap().phase(0, 0), cfa, "{packing:?}");
            let d = develop_sensor(&s, &DevelopOptions::default()).unwrap();
            assert_eq!(d.info.cfa.as_deref(), Some(if cfa == RGGB { "RGGB" } else { "BGGR" }));
        }
    }
}

#[test]
fn unsupported_variants_report_and_keep_the_preview() {
    let mut spec = xtrans(RafPacking::Compressed, 14);
    let b = spec.build();
    match decode(&b, &Limits::default()) {
        Err(RawError::Unsupported(m)) => assert!(m.contains("compressed"), "{m}"),
        other => panic!("expected unsupported, got {other:?}"),
    }
    let p = embedded_preview(&b).unwrap();
    assert_eq!((p.width, p.height), (32, 24));

    spec.packing = RafPacking::U16 { le: true };
    spec.cfa_tiff = false;
    match decode(&spec.build(), &Limits::default()) {
        Err(RawError::Unsupported(m)) => assert!(m.contains("FinePix"), "{m}"),
        other => panic!("expected unsupported, got {other:?}"),
    }
}

#[test]
fn bad_layouts_and_sizes_fail_cleanly() {
    // A Bayer FujiLayout that is not a Bayer pattern.
    let spec = RafSpec::new(8, 4, vec![100; 32], vec![0, 0, 2, 2]);
    assert!(matches!(decode(&spec.build(), &Limits::default()), Err(RawError::Unsupported(_))));
    // Dimensions beyond the limits are rejected before allocating.
    let spec = xtrans(RafPacking::U16 { le: true }, 14);
    let limits = Limits { max_pixels: 100, ..Limits::default() };
    assert!(matches!(decode(&spec.build(), &limits), Err(RawError::LimitExceeded(_))));
    // A truncated strip.
    let b = spec.build();
    assert!(matches!(decode(&b[..b.len() - 7], &Limits::default()), Err(RawError::Malformed(_))));
}
