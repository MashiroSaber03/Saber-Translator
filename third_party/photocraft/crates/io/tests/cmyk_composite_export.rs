use photocraft_cms::{Builtin, Intent, Profile, Transform};
use photocraft_codecs::{ChannelLayout, SampleType as CodecSample, decode};
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::Size;
use photocraft_io::{ExportOptions, document_to_image, export};
use photocraft_raster::Surface;

fn document(depth: SampleType, alpha: f32) -> Document {
    let mut doc = Document::new("CMYK composite", Size::new(8, 8), ColorMode::Cmyk, depth);
    doc.icc_profile = Some(Builtin::CoatedCmyk.profile().to_bytes());
    let surface = Surface::with_default(doc.pixel_format(), &[0.3, 0.6, 0.1, 0.2, alpha]);
    doc.layers.push(Layer::new("Ink", LayerContent::Raster(surface)));
    doc.layers.push(Layer::new("Empty", LayerContent::Raster(Surface::new(doc.pixel_format()))));
    doc
}

fn tag(bytes: &[u8], id: u16) -> Option<u16> {
    assert_eq!(&bytes[..2], b"II");
    let start = usize::try_from(u32::from_le_bytes(bytes[4..8].try_into().unwrap())).unwrap();
    let count = usize::from(u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()));
    bytes[start + 2..]
        .as_chunks::<12>()
        .0
        .iter()
        .take(count)
        .find_map(|entry| (u16::from_le_bytes(entry[..2].try_into().unwrap()) == id).then(|| u16::from_le_bytes(entry[8..10].try_into().unwrap())))
}

#[test]
fn tiff_keeps_cmyk_profile_depth_and_straight_alpha_when_composited() {
    for (depth, sample) in [(SampleType::U8, CodecSample::U8), (SampleType::U16, CodecSample::U16), (SampleType::F32, CodecSample::F32)] {
        for alpha in [1.0, 0.5] {
            let doc = document(depth, alpha);

            let result = export(&doc, "tiff", &ExportOptions::default()).unwrap();

            let image = decode(&result.bytes).unwrap();
            assert_eq!(image.layout(), if alpha == 1.0 { ChannelLayout::Cmyk } else { ChannelLayout::CmykA });
            assert_eq!(image.sample_type(), sample);
            assert_eq!(image.icc.as_deref(), doc.icc_profile.as_deref().map(Vec::as_slice));
            assert_eq!(tag(&result.bytes, 262), Some(5));
            assert_eq!(tag(&result.bytes, 277), Some(if alpha == 1.0 { 4 } else { 5 }));
            assert_eq!(tag(&result.bytes, 338), if alpha == 1.0 { None } else { Some(2) });
            if alpha < 1.0 {
                assert!(image.to_normalized().as_chunks::<5>().0.iter().all(|px| (px[4] - alpha).abs() < 0.004));
                let opaque = document_to_image(&document(depth, 1.0), &mut Vec::new()).unwrap().to_normalized();
                for (px, color) in image.to_normalized().as_chunks::<5>().0.iter().zip(opaque.as_chunks::<4>().0) {
                    assert!(px[..4].iter().zip(color).all(|(a, b)| (a - b).abs() < 0.004));
                }
            }
        }
    }
}

#[test]
fn tiff_keeps_true_zero_alpha_when_composited() {
    for (depth, sample) in [(SampleType::U8, CodecSample::U8), (SampleType::U16, CodecSample::U16), (SampleType::F32, CodecSample::F32)] {
        let doc = document(depth, 0.0);

        let result = export(&doc, "tiff", &ExportOptions::default()).unwrap();

        let image = decode(&result.bytes).unwrap();
        assert_eq!(image.layout(), ChannelLayout::CmykA);
        assert_eq!(image.sample_type(), sample);
        assert_eq!(image.icc.as_deref(), doc.icc_profile.as_deref().map(Vec::as_slice));
        assert_eq!(tag(&result.bytes, 262), Some(5));
        assert_eq!(tag(&result.bytes, 277), Some(5));
        assert_eq!(tag(&result.bytes, 338), Some(2));
        assert!(image.to_normalized().as_chunks::<5>().0.iter().all(|px| px[4].abs() < f32::EPSILON));
    }
}

#[test]
fn psd_keeps_cmyk_mode_depth_and_profile_when_composited() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let doc = document(depth, 0.5);

        let result = export(&doc, "psd", &ExportOptions::default()).unwrap();

        let back = photocraft_io::import("composite.psd", &result.bytes).unwrap().document;
        assert_eq!(back.mode, ColorMode::Cmyk);
        assert_eq!(back.depth, depth);
        assert_eq!(back.icc_profile, doc.icc_profile);
        assert_eq!(back.layer_count(), doc.layer_count());
    }
}

#[test]
fn jpeg_keeps_four_components_profile_and_zero_ink_matte_when_composited() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for alpha in [1.0, 0.5, 0.0] {
            let doc = document(depth, alpha);
            let mut opts = ExportOptions::default();
            opts.encode.jpeg_quality = 100;

            let result = export(&doc, "jpg", &opts).unwrap();

            let image = decode(&result.bytes).unwrap();
            assert_eq!(image.layout(), ChannelLayout::Cmyk);
            assert_eq!(image.icc.as_deref(), doc.icc_profile.as_deref().map(Vec::as_slice));
            let adobe = result.bytes.windows(5).position(|bytes| bytes == b"Adobe").unwrap();
            assert_eq!(result.bytes[adobe + 11], 0);
            let mut marker = 2;
            while result.bytes[marker + 1] != 192 {
                let length = usize::from(u16::from_be_bytes(result.bytes[marker + 2..marker + 4].try_into().unwrap()));
                marker += length + 2;
            }
            assert_eq!(result.bytes[marker + 9], 4);
            let expected = document_to_image(&doc, &mut Vec::new()).unwrap();
            for (px, source) in image.to_normalized().as_chunks::<4>().0.iter().zip(expected.to_normalized().chunks_exact(expected.layout().channels())) {
                for channel in 0..4 {
                    let ink = source[channel] * if expected.layout().has_alpha() { source[4] } else { 1.0 };
                    assert!((px[channel] - ink).abs() < 0.015, "{px:?} vs {source:?}");
                }
            }
            if depth != SampleType::U8 {
                assert!(result.warnings.iter().any(|warning| warning.contains("bit")), "{:?}", result.warnings);
            }
        }
    }
}

#[test]
fn composite_uses_distinctive_document_profile_on_every_worker() {
    let params = photocraft_cms::synth::CmykParams {
        description: "Export synthetic uncoated".into(),
        tvi: [0.26, 0.26, 0.26, 0.3],
        grid_a2b: 5,
        grid_b2a: 9,
        ..Default::default()
    };
    let profile = photocraft_cms::synth::cmyk_profile(&params);
    let mut doc = document(SampleType::F32, 0.5);
    doc.size = Size::new(64, 64);
    doc.icc_profile = Some(profile.to_bytes());
    let rgb = photocraft_compose::flatten(&doc);
    let transform = Transform::new(Builtin::Srgb.profile(), &profile, Intent::RelativeColorimetric, true).unwrap();
    let fallback = Transform::new(Builtin::Srgb.profile(), Builtin::CoatedCmyk.profile(), Intent::RelativeColorimetric, true).unwrap();
    let mut expected = [0.0; 4];
    transform.eval_fast(&rgb.px[0][..3], &mut expected);
    let mut wrong = [0.0; 4];
    fallback.eval_fast(&rgb.px[0][..3], &mut wrong);
    assert!(expected.iter().zip(wrong).any(|(a, b)| (a - b).abs() > 0.02));
    assert!(photocraft_io::pixels::bands(rgb.px.len()).len() > 1);

    let image = document_to_image(&doc, &mut Vec::new()).unwrap();

    assert_eq!(image.layout(), ChannelLayout::CmykA);
    for px in image.to_normalized().as_chunks::<5>().0 {
        assert!(px[..4].iter().zip(expected).all(|(a, b)| (a - b).abs() < 0.00001), "{px:?} vs {expected:?}");
    }
    for extension in ["tiff", "jpg"] {
        let encoded = export(&doc, extension, &ExportOptions::default()).unwrap();
        assert_eq!(decode(&encoded.bytes).unwrap().icc.as_deref(), doc.icc_profile.as_deref().map(Vec::as_slice));
    }
}

#[test]
fn cmyk_composite_rejects_invalid_non_cmyk_and_unusable_profiles() {
    let without_tags = |prefix: &[u8]| {
        let mut bytes = Builtin::CoatedCmyk.profile().to_bytes().to_vec();
        let count = usize::try_from(u32::from_be_bytes(bytes[128..132].try_into().unwrap())).unwrap();
        for entry in bytes[132..].as_chunks_mut::<12>().0.iter_mut().take(count) {
            if &entry[..3] == prefix {
                entry[..4].copy_from_slice(b"test");
            }
        }
        std::sync::Arc::new(bytes)
    };
    for bytes in [std::sync::Arc::new(vec![1, 2, 3]), Builtin::Srgb.profile().to_bytes(), without_tags(b"B2A"), without_tags(b"A2B")] {
        let mut doc = document(SampleType::U8, 1.0);
        doc.icc_profile = Some(bytes);

        for extension in ["tiff", "jpg"] {
            assert!(export(&doc, extension, &ExportOptions::default()).is_err(), "accepted invalid profile for {extension}");
        }
        // RGB-only targets never write CMYK, so they still export (through the built-in profile).
        assert!(export(&doc, "png", &ExportOptions::default()).is_ok(), "rejected invalid profile for png");
    }
}

#[test]
fn untagged_composite_embeds_coated_cmyk() {
    let mut doc = document(SampleType::U8, 1.0);
    doc.icc_profile = None;

    let image = document_to_image(&doc, &mut Vec::new()).unwrap();

    assert_eq!(image.layout(), ChannelLayout::Cmyk);
    assert_eq!(image.icc.unwrap(), Builtin::CoatedCmyk.profile().to_bytes().as_slice());
}

#[test]
fn native_samples_are_preserved_without_composite_roundtrip() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut doc = document(depth, 0.5);
        doc.layers.pop();
        let LayerContent::Raster(surface) = &doc.layers[0].content else { panic!() };
        let expected = surface.to_interleaved(doc.bounds());

        let image = document_to_image(&doc, &mut Vec::new()).unwrap();

        assert_eq!(image.data(), expected);
    }
}

#[test]
fn rgb_only_export_uses_direct_composite_not_reseparated_cmyk() {
    let doc = document(SampleType::U16, 0.5);
    let rgb = photocraft_compose::flatten(&doc);
    let expected =
        photocraft_codecs::Image::from_normalized(8, 8, ChannelLayout::Rgba, CodecSample::U16, &rgb.px.into_iter().flatten().collect::<Vec<_>>()).unwrap();

    let result = export(&doc, "png", &ExportOptions::default()).unwrap();

    let image = decode(&result.bytes).unwrap();
    assert_eq!(image.data(), expected.data());
    assert_eq!(Profile::parse(image.icc.as_ref().unwrap()).unwrap().color_space, photocraft_cms::ColorSpace::Rgb);
}

#[test]
fn layered_tiff_keeps_its_existing_cmyk_psd_composite_path() {
    let doc = document(SampleType::U16, 0.5);
    let opts = ExportOptions { tiff_layers: true, ..Default::default() };

    let result = export(&doc, "tiff", &opts).unwrap();

    let image = decode(&result.bytes).unwrap();
    assert_eq!(image.layout(), ChannelLayout::CmykA);
    assert_eq!(image.icc.as_deref(), doc.icc_profile.as_deref().map(Vec::as_slice));
    assert!(image.meta.photoshop_layers.is_some());
    assert_eq!(tag(&result.bytes, 338), Some(2));
    assert!(image.to_normalized().as_chunks::<5>().0.iter().all(|px| (px[4] - 0.5).abs() < 0.0001));
}

/// A tagged CMYK file's merged image reads through its own profile, as the compositor reads the
/// layers, so the two are compared in the same colours (the oracle tests rely on it: psd-tools'
/// tagged CMYK files were 95–100% off before).
#[test]
fn merged_composite_of_a_tagged_cmyk_file_reads_through_its_profile() {
    use photocraft_cms::synth::{CmykParams, cmyk_profile};
    use photocraft_color::convert::{CmykSpace, cmyk_to_rgb, with_cmyk_space};
    let mut doc = Document::new("tagged", Size::new(6, 4), ColorMode::Cmyk, SampleType::U8);
    let profile =
        cmyk_profile(&CmykParams { description: "Merged Test CMYK".into(), tvi: [0.26, 0.26, 0.26, 0.3], grid_a2b: 5, grid_b2a: 9, ..Default::default() });
    doc.icc_profile = Some(profile.to_bytes());
    let mut surface = Surface::new(photocraft_color::PixelFormat::new(ColorMode::Cmyk, SampleType::U8, false));
    for x in 0..6 {
        for y in 0..4 {
            surface.write_pixel(x, y, &[x as f32 / 5.0, y as f32 / 3.0, 0.4, (x * y) as f32 / 15.0]);
        }
    }
    doc.layers.push(Layer::new("Background", LayerContent::Raster(surface)));
    let psd = export(&doc, "psd", &ExportOptions::default()).unwrap();
    let file = photocraft_psd::PsdFile::from_bytes(&psd.bytes).unwrap();
    let merged = photocraft_io::merged_composite(&file).unwrap();
    // The merged image's own inks (planar, 255 = no ink), shown through the file's profile.
    let raw = file.decode_merged().unwrap();
    let plane = 6 * 4;
    let space = CmykSpace::for_profile(doc.icc_profile.as_ref());
    assert!(space.is_some(), "a profile other than the built-in coated CMYK");
    let shown: Vec<[f32; 3]> =
        with_cmyk_space(space.as_ref(), || (0..plane).map(|i| cmyk_to_rgb(std::array::from_fn(|c| 1.0 - f32::from(raw[c * plane + i]) / 255.0))).collect());
    let builtin: Vec<[f32; 3]> = (0..plane).map(|i| cmyk_to_rgb(std::array::from_fn(|c| 1.0 - f32::from(raw[c * plane + i]) / 255.0))).collect();
    let apart = |a: &[[f32; 3]]| merged.iter().zip(a).flat_map(|(m, s)| (0..3).map(move |c| (m[c] - s[c]).abs())).fold(0.0f32, f32::max);
    assert!(apart(&shown) < 1e-4, "merged image read through its profile: {}", apart(&shown));
    assert!(apart(&builtin) > 0.01, "the built-in profile would show it differently");
}
