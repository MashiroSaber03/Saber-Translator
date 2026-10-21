//! ICC profiles through import/export: byte-exact PSD round trips with real profiles, PNG/JPEG
//! embedding and extraction, colour-managed CMYK → RGB for formats without CMYK.

mod common;

use std::sync::Arc;

use common::*;
use photocraft_cms::{Builtin, ColorSpace, Profile};
use photocraft_color::{ColorMode, SampleType};
use photocraft_io::*;

fn single(mode: ColorMode, depth: SampleType, alpha: bool) -> photocraft_doc::Document {
    let mut d = photocraft_doc::Document::new("s", photocraft_geom::Size::new(9, 6), mode, depth);
    let fmt = d.pixel_format();
    d.layers.push(raster("Background", fmt, d.bounds(), 3, alpha));
    d
}

fn real_profiles() -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = Builtin::ALL.iter().map(|b| b.profile().to_bytes().to_vec()).collect();
    for p in ["/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc", "/System/Library/ColorSync/Profiles/AdobeRGB1998.icc"] {
        if let Ok(b) = std::fs::read(p) {
            v.push(b);
        }
    }
    v
}

#[test]
fn psd_icc_roundtrip_byte_exact() {
    for icc in real_profiles() {
        let p = Profile::parse(&icc).unwrap();
        let mode = match p.color_space {
            ColorSpace::Cmyk => ColorMode::Cmyk,
            ColorSpace::Gray => ColorMode::Grayscale,
            ColorSpace::Lab => ColorMode::Lab,
            _ => ColorMode::Rgb,
        };
        let mut d = gen_doc(mode, SampleType::U8, Features::PIXELS);
        d.icc_profile = Some(Arc::new(icc.clone()));
        let r = export(&d, "x.psd", &ExportOptions::default()).unwrap();
        let back = import("x.psd", &r.bytes).unwrap().document;
        assert_eq!(back.icc_profile.as_deref(), Some(&icc), "{}", p.description);
    }
}

#[test]
fn png_and_jpeg_embed_and_extract_icc() {
    for b in [Builtin::Srgb, Builtin::DisplayP3, Builtin::AdobeRgbCompat, Builtin::ProPhotoCompat] {
        let icc = b.profile().to_bytes().to_vec();
        for (name, alpha) in [("x.png", true), ("x.jpg", false)] {
            let mut d = single(ColorMode::Rgb, SampleType::U8, alpha);
            d.icc_profile = Some(Arc::new(icc.clone()));
            let r = export(&d, name, &ExportOptions::default()).unwrap();
            let back = import(name, &r.bytes).unwrap().document;
            assert_eq!(back.icc_profile.as_deref(), Some(&icc), "{b:?} {name}");
            assert_eq!(Profile::parse(back.icc_profile.as_ref().unwrap()).unwrap().description, b.description());
        }
    }
    // Gray PNG with a gray profile.
    let icc = Builtin::GrayGamma22.profile().to_bytes().to_vec();
    let mut d = single(ColorMode::Grayscale, SampleType::U16, false);
    d.icc_profile = Some(Arc::new(icc.clone()));
    let r = export(&d, "g.png", &ExportOptions::default()).unwrap();
    assert_eq!(import("g.png", &r.bytes).unwrap().document.icc_profile.as_deref(), Some(&icc));
}

#[test]
fn cmyk_to_png_is_colour_managed_and_tagged_srgb() {
    // Native single-layer CMYK document written to PNG: converted through the CMYK profile.
    let mut d = single(ColorMode::Cmyk, SampleType::U8, false);
    d.icc_profile = Some(Builtin::CoatedCmyk.profile().to_bytes());
    let r = export(&d, "c.png", &ExportOptions::default()).unwrap();
    assert!(r.warnings.iter().any(|w| w.contains("sRGB")), "{:?}", r.warnings);
    let back = import("c.png", &r.bytes).unwrap().document;
    assert_eq!(back.mode, ColorMode::Rgb);
    let icc = back.icc_profile.expect("tagged");
    assert_eq!(Profile::parse(&icc).unwrap().description, Builtin::Srgb.profile().description);
    // Lab documents never write Lab numbers as RGB.
    let d = single(ColorMode::Lab, SampleType::U8, false);
    let r = export(&d, "l.png", &ExportOptions::default()).unwrap();
    let back = import("l.png", &r.bytes).unwrap().document;
    assert_eq!(Profile::parse(back.icc_profile.as_ref().unwrap()).unwrap().color_space, ColorSpace::Rgb);
}

#[test]
fn cmyk_fills_survive_psd_roundtrip_at_every_depth() {
    use photocraft_doc::{Color, Document, Fill, GradientStyle, Layer, LayerContent};
    let icc = photocraft_cms::synth::cmyk_profile(&photocraft_cms::synth::CmykParams {
        tvi: [0.26, 0.26, 0.26, 0.3],
        grid_a2b: 5,
        grid_b2a: 9,
        ..Default::default()
    })
    .to_bytes();
    let a = Color { mode: ColorMode::Cmyk, c: [0.2, 0.4, 0.6, 0.3], alpha: 0.7 };
    let b = Color { mode: ColorMode::Cmyk, c: [0.6, 0.2, 0.3, 0.1], alpha: 0.4 };
    for depth in SampleType::ALL {
        for profile in [None, Some(icc.clone())] {
            for style in [
                None,
                Some(GradientStyle::Linear),
                Some(GradientStyle::Radial),
                Some(GradientStyle::Angle),
                Some(GradientStyle::Reflected),
                Some(GradientStyle::Diamond),
            ] {
                let mut fill = style.map_or(Fill::Solid(a), |s| Fill::gradient(vec![(1.0, b), (0.0, a)], 35.0, 0.8, s, true));
                if let Fill::Gradient { midpoints, opacity_stops, dither, offset, .. } = &mut fill {
                    *midpoints = vec![0.25];
                    *opacity_stops = vec![(0.0, 0.6), (1.0, 0.9)];
                    *offset = (0.1, -0.2);
                    *dither = true;
                }
                let mut doc = Document::new("CMYK fill", photocraft_geom::Size::new(9, 6), ColorMode::Cmyk, depth);
                doc.icc_profile = profile.clone();
                doc.layers.push(Layer::new("Fill", LayerContent::Fill(fill)));
                let expected = photocraft_compose::flatten(&doc).px;
                assert!(max_diff(&expected, &photocraft_compose::render_tiled(&doc, doc.bounds(), 3).px) <= 1e-6);
                let bytes = export(&doc, "x.psd", &ExportOptions::default()).unwrap().bytes;
                let back = import("x.psd", &bytes).unwrap().document;
                let actual = photocraft_compose::flatten(&back).px;
                let diff = max_diff(&expected, &actual);
                assert!(diff <= 1e-6, "{depth:?} {style:?}: round-trip error {diff}");
                assert_eq!(back.icc_profile, doc.icc_profile);
                if style.is_none() {
                    let native = photocraft_raster::Surface::with_default(doc.pixel_format(), &[a.c[0], a.c[1], a.c[2], a.c[3], a.alpha]);
                    let actual = back.layers[0].fill_cache.as_ref().unwrap().surface.read_region(doc.bounds());
                    for (actual, expected) in actual.iter().zip(native.read_region(doc.bounds())) {
                        let tolerance = if depth == SampleType::F32 { 1e-6 } else { 0.0 };
                        assert!((actual - expected).abs() <= tolerance, "{depth:?}: {actual} vs {expected}");
                    }
                }
            }
        }
    }
}

#[cfg(feature = "corpus")]
#[test]
fn uncached_cmyk_gradient_corpus_roundtrip() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd-tools/colormodes/4x4_16bit_cmyk.psd");
    let bytes = std::fs::read(path).expect("run cargo xtask corpus --all");
    let doc = import("x.psd", &bytes).unwrap().document;
    assert!(doc.layers.iter().any(|l| matches!(l.content, photocraft_doc::LayerContent::Fill(_)) && l.fill_cache.is_none()));
    let expected = photocraft_compose::flatten(&doc).px;
    let saved = export(&doc, "x.psd", &ExportOptions::default()).unwrap().bytes;
    let back = import("x.psd", &saved).unwrap().document;
    let diff = max_diff(&expected, &photocraft_compose::flatten(&back).px);
    eprintln!("uncached 16-bit CMYK gradient round-trip max error: {diff:.8}");
    assert!(diff <= 1e-6, "round-trip error {diff}");
    assert_eq!(back.icc_profile, doc.icc_profile);
}

#[test]
fn rgb_fill_colours_separate_through_the_document_cmyk_profile() {
    use photocraft_cms::{Intent, Transform, synth};
    use photocraft_doc::{Color, Document, Fill, Layer, LayerContent};
    let profile = synth::cmyk_profile(&synth::CmykParams { tvi: [0.26, 0.26, 0.26, 0.3], grid_a2b: 5, grid_b2a: 9, ..Default::default() });
    let profile = Profile::parse(&profile.to_bytes()).unwrap();
    let rgb = [0.2, 0.4, 0.6];
    let mut expected = [0.0; 4];
    Transform::new(Builtin::Srgb.profile(), &profile, Intent::RelativeColorimetric, true).unwrap().eval_fast(&rgb, &mut expected);
    for depth in SampleType::ALL {
        let mut doc = Document::new("RGB fill", photocraft_geom::Size::new(1, 1), ColorMode::Cmyk, depth);
        doc.icc_profile = Some(profile.to_bytes());
        doc.layers.push(Layer::new("Fill", LayerContent::Fill(Fill::Solid(Color::rgb(rgb[0], rgb[1], rgb[2])))));
        let bytes = export(&doc, "x.psd", &ExportOptions::default()).unwrap().bytes;
        let back = import("x.psd", &bytes).unwrap().document;
        let samples = back.layers[0].fill_cache.as_ref().unwrap().surface.read_region(back.bounds());
        let tolerance = match depth {
            SampleType::U8 => 1.0 / 255.0,
            SampleType::U16 => 1.0 / 65535.0,
            SampleType::F32 => 1e-6,
        };
        for (actual, expected) in samples.iter().zip(expected) {
            assert!((actual - expected).abs() <= tolerance, "{depth:?}: {actual} vs {expected}");
        }
    }
}

/// A two-stop version 1 `grdm` block whose stops are the PSD colour structures `a` and `b`.
fn grdm_block(a: (u16, [u16; 4]), b: (u16, [u16; 4])) -> Vec<u8> {
    let mut v = vec![0, 1, 0, 0];
    v.extend_from_slice(&1u32.to_be_bytes()); // name: just the terminator
    v.extend_from_slice(&[0, 0]);
    v.extend_from_slice(&2u16.to_be_bytes());
    for (t, (space, c)) in [(0u32, a), (4096, b)] {
        v.extend_from_slice(&t.to_be_bytes());
        v.extend_from_slice(&50u32.to_be_bytes());
        v.extend_from_slice(&space.to_be_bytes());
        for x in c {
            v.extend_from_slice(&x.to_be_bytes());
        }
        v.extend_from_slice(&[0, 0]);
    }
    v.extend_from_slice(&2u16.to_be_bytes());
    for t in [0u32, 4096] {
        v.extend_from_slice(&t.to_be_bytes());
        v.extend_from_slice(&50u32.to_be_bytes());
        v.extend_from_slice(&255u16.to_be_bytes());
    }
    v.extend_from_slice(&2u16.to_be_bytes()); // expansion count
    v.extend_from_slice(&0u16.to_be_bytes()); // smoothness
    v
}

fn saved_grdm(bytes: &[u8]) -> Vec<u8> {
    let f = photocraft_psd::PsdFile::from_bytes(bytes).unwrap();
    f.layers().iter().find_map(|l| l.block(b"grdm")).map(|b| b.data.clone()).expect("grdm block")
}

#[test]
fn gradient_maps_with_cmyk_lab_gray_stops_import_and_keep_their_space() {
    use photocraft_cms::{Intent, Transform, synth};
    use photocraft_doc::{Adjustment, Document, Layer, LayerContent};
    let profile = synth::cmyk_profile(&synth::CmykParams { tvi: [0.26, 0.26, 0.26, 0.3], grid_a2b: 5, grid_b2a: 9, ..Default::default() });
    let profile = Profile::parse(&profile.to_bytes()).unwrap();
    let to_srgb = Transform::new(&profile, Builtin::Srgb.profile(), Intent::RelativeColorimetric, true).unwrap();
    // First stop 60 % cyan + 40 % black (stored inverted), second paper white.
    let cmyk_stop = (2u16, [65535 - 39321, 65535, 65535, 65535 - 26214]);
    let paper = (2u16, [65535; 4]);
    let lab_dark = (7u16, [2000, 0, 0, 0]);
    let lab_red = (7u16, [5000, 6000u16, 2000, 0]);
    let gray_black = (8u16, [10000, 0, 0, 0]);
    let gray_white = (8u16, [0, 0, 0, 0]);
    for depth in SampleType::ALL {
        for (mode, stops) in [(ColorMode::Cmyk, (cmyk_stop, paper)), (ColorMode::Lab, (lab_dark, lab_red)), (ColorMode::Grayscale, (gray_white, gray_black))] {
            let block = grdm_block(stops.0, stops.1);
            let mut doc = Document::new("map", photocraft_geom::Size::new(4, 2), mode, depth);
            if mode == ColorMode::Cmyk {
                doc.icc_profile = Some(profile.to_bytes());
            }
            let fmt = doc.pixel_format();
            doc.layers.push(raster("Background", fmt, doc.bounds(), 3, false));
            doc.layers.push(Layer::new("Map", LayerContent::Adjustment(Adjustment::Unsupported { psd_key: "grdm".into(), raw: block.clone() })));
            let bytes = export(&doc, "x.psd", &ExportOptions::default()).unwrap().bytes;
            let mut back = import("x.psd", &bytes).unwrap().document;
            let LayerContent::Adjustment(Adjustment::GradientMap { stops: got, .. }) = &back.layers[1].content else {
                panic!("{mode:?} {depth:?}: {:?}", back.layers[1].content);
            };
            let (first, last) = (got[0].1, got[got.len() - 1].1);
            match mode {
                ColorMode::Cmyk => {
                    // Through the document's own profile, not the built-in coated CMYK.
                    let mut expected = [0.0f32; 3];
                    to_srgb.eval_fast(&[0.6, 0.0, 0.0, 0.4], &mut expected);
                    assert!(first.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-4), "{depth:?}: {first:?} vs {expected:?}");
                    let builtin = photocraft_color::convert::cmyk_to_rgb([0.6, 0.0, 0.0, 0.4]);
                    assert!(first.iter().zip(builtin).any(|(a, b)| (a - b).abs() > 1e-3), "the document profile is used");
                }
                ColorMode::Lab => {
                    assert!(first.iter().all(|v| (v - first[0]).abs() < 1e-3) && first[0] < 0.3, "{first:?}");
                    assert!(last[0] > last[1] + 0.2, "a = +60 is reddish: {last:?}");
                }
                // An inverting map, white to black.
                _ => assert_eq!((first, last), ([1.0; 3], [0.0; 3])),
            }
            // The layer now renders: the composite differs from the background alone.
            let with_map = photocraft_compose::flatten(&back).px;
            let mut hidden = back.clone();
            hidden.layers[1].visible = false;
            assert!(max_diff(&with_map, &photocraft_compose::flatten(&hidden).px) > 0.01, "{mode:?} {depth:?}");
            // Saving again keeps the block (and so the stops' colour space) byte for byte...
            let again = export(&back, "x.psd", &ExportOptions::default()).unwrap().bytes;
            assert_eq!(saved_grdm(&again), block, "{mode:?} {depth:?}");
            // ...and so does toggling Reverse and Dither, which only flip their flags.
            if let LayerContent::Adjustment(Adjustment::GradientMap { reverse, dither, .. }) = &mut back.layers[1].content {
                *reverse = true;
                *dither = true;
            }
            let flipped_file = export(&back, "x.psd", &ExportOptions::default()).unwrap().bytes;
            let flipped = saved_grdm(&flipped_file);
            assert_eq!(&flipped[2..4], &[1, 1]);
            assert_eq!(flipped[4..], block[4..], "{mode:?} {depth:?}: stops keep their colour space");
            let reopened = import("x.psd", &flipped_file).unwrap().document;
            assert_eq!(reopened.layers[1].content, back.layers[1].content);
        }
    }
}
