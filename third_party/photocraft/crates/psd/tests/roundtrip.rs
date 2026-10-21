//! Round-trip and byte-stability tests over generated files.

use photocraft_psd::testgen::{self, MODES, mode_channels, mode_depths, pattern_plane};
use photocraft_psd::*;

fn assert_stable(file: &PsdFile, name: &str) {
    let bytes = file.to_bytes().unwrap_or_else(|e| panic!("{name}: write failed: {e}"));
    let parsed = PsdFile::from_bytes(&bytes).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
    assert_eq!(&parsed, file, "{name}: model round trip");
    let again = parsed.to_bytes().unwrap();
    assert_eq!(again, bytes, "{name}: byte stability");
}

#[test]
fn all_generated_cases_roundtrip() {
    let cases = testgen::all_cases();
    assert!(cases.len() > 100, "only {} cases", cases.len());
    for c in &cases {
        assert_stable(&c.file, &c.name);
    }
}

#[test]
fn all_generated_channels_decode() {
    for c in testgen::all_cases() {
        let f = &c.file;
        let merged = f.decode_merged().unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let (w, h) = (f.header.width as usize, f.header.height as usize);
        assert_eq!(merged.len(), f.header.row_bytes() * h * usize::from(f.header.channels));
        for (i, layer) in f.layers().iter().enumerate() {
            for ch in &layer.channels {
                layer.decode_channel(ch.id, f.header.depth, f.header.version).unwrap_or_else(|e| panic!("{} layer {i} ch {}: {e}", c.name, ch.id));
            }
        }
        let _ = (w, h);
    }
}

#[test]
fn merged_only_decodes_to_pattern() {
    for mode in MODES {
        for &depth in mode_depths(mode) {
            for c in Compression::ALL {
                let f = testgen::merged_only(Version::Psd, mode, depth, c, 9, 5);
                let dec = f.decode_merged().unwrap();
                let mut expect = Vec::new();
                for ch in 0..mode_channels(mode) {
                    expect.extend(pattern_plane(9, 5, depth, u32::from(ch)));
                }
                assert_eq!(dec, expect, "{mode:?} {depth} {c:?}");
            }
        }
    }
}

macro_rules! layered_case {
    ($name:ident, $ver:expr, $mode:expr, $depth:expr, $comp:expr) => {
        #[test]
        fn $name() {
            let f = testgen::layered($ver, $mode, $depth, $comp);
            assert_stable(&f, stringify!($name));
            assert!(f.layers().len() > 30);
            let expect_placement = $depth != 8;
            assert_eq!(matches!(f.layer_info_placement, LayerInfoPlacement::GlobalBlock { .. }), expect_placement);
        }
    };
}

layered_case!(layered_psd_rgb8_raw, Version::Psd, ColorMode::Rgb, 8, Compression::Raw);
layered_case!(layered_psd_rgb8_rle, Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
layered_case!(layered_psd_rgb8_zip, Version::Psd, ColorMode::Rgb, 8, Compression::Zip);
layered_case!(layered_psd_rgb8_zipp, Version::Psd, ColorMode::Rgb, 8, Compression::ZipPrediction);
layered_case!(layered_psd_rgb16_raw, Version::Psd, ColorMode::Rgb, 16, Compression::Raw);
layered_case!(layered_psd_rgb16_rle, Version::Psd, ColorMode::Rgb, 16, Compression::Rle);
layered_case!(layered_psd_rgb16_zip, Version::Psd, ColorMode::Rgb, 16, Compression::Zip);
layered_case!(layered_psd_rgb16_zipp, Version::Psd, ColorMode::Rgb, 16, Compression::ZipPrediction);
layered_case!(layered_psd_rgb32_raw, Version::Psd, ColorMode::Rgb, 32, Compression::Raw);
layered_case!(layered_psd_rgb32_rle, Version::Psd, ColorMode::Rgb, 32, Compression::Rle);
layered_case!(layered_psd_rgb32_zip, Version::Psd, ColorMode::Rgb, 32, Compression::Zip);
layered_case!(layered_psd_rgb32_zipp, Version::Psd, ColorMode::Rgb, 32, Compression::ZipPrediction);
layered_case!(layered_psb_rgb8_raw, Version::Psb, ColorMode::Rgb, 8, Compression::Raw);
layered_case!(layered_psb_rgb8_rle, Version::Psb, ColorMode::Rgb, 8, Compression::Rle);
layered_case!(layered_psb_rgb8_zip, Version::Psb, ColorMode::Rgb, 8, Compression::Zip);
layered_case!(layered_psb_rgb8_zipp, Version::Psb, ColorMode::Rgb, 8, Compression::ZipPrediction);
layered_case!(layered_psb_rgb16_rle, Version::Psb, ColorMode::Rgb, 16, Compression::Rle);
layered_case!(layered_psb_rgb32_zipp, Version::Psb, ColorMode::Rgb, 32, Compression::ZipPrediction);
layered_case!(layered_psd_gray8_rle, Version::Psd, ColorMode::Grayscale, 8, Compression::Rle);
layered_case!(layered_psd_gray16_zipp, Version::Psd, ColorMode::Grayscale, 16, Compression::ZipPrediction);
layered_case!(layered_psd_gray32_zip, Version::Psd, ColorMode::Grayscale, 32, Compression::Zip);
layered_case!(layered_psd_cmyk8_rle, Version::Psd, ColorMode::Cmyk, 8, Compression::Rle);
layered_case!(layered_psd_cmyk16_raw, Version::Psd, ColorMode::Cmyk, 16, Compression::Raw);
layered_case!(layered_psb_cmyk8_zip, Version::Psb, ColorMode::Cmyk, 8, Compression::Zip);
layered_case!(layered_psd_lab8_rle, Version::Psd, ColorMode::Lab, 8, Compression::Rle);
layered_case!(layered_psb_lab16_zipp, Version::Psb, ColorMode::Lab, 16, Compression::ZipPrediction);

macro_rules! merged_case {
    ($name:ident, $mode:expr, $depth:expr) => {
        #[test]
        fn $name() {
            for v in [Version::Psd, Version::Psb] {
                for c in Compression::ALL {
                    let f = testgen::merged_only(v, $mode, $depth, c, 31, 3);
                    assert_stable(&f, stringify!($name));
                    assert!(f.layer_info.is_none());
                }
            }
        }
    };
}

merged_case!(merged_bitmap_1, ColorMode::Bitmap, 1);
merged_case!(merged_gray_8, ColorMode::Grayscale, 8);
merged_case!(merged_gray_16, ColorMode::Grayscale, 16);
merged_case!(merged_gray_32, ColorMode::Grayscale, 32);
merged_case!(merged_indexed_8, ColorMode::Indexed, 8);
merged_case!(merged_rgb_8, ColorMode::Rgb, 8);
merged_case!(merged_rgb_16, ColorMode::Rgb, 16);
merged_case!(merged_rgb_32, ColorMode::Rgb, 32);
merged_case!(merged_cmyk_8, ColorMode::Cmyk, 8);
merged_case!(merged_cmyk_16, ColorMode::Cmyk, 16);
merged_case!(merged_lab_8, ColorMode::Lab, 8);
merged_case!(merged_lab_16, ColorMode::Lab, 16);
merged_case!(merged_multichannel_8, ColorMode::Multichannel, 8);
merged_case!(merged_multichannel_16, ColorMode::Multichannel, 16);
merged_case!(merged_duotone_8, ColorMode::Duotone, 8);

#[test]
fn psb_uses_long_section_lengths() {
    let psd = testgen::small(Version::Psd, Compression::Raw).to_bytes().unwrap();
    let psb = testgen::small(Version::Psb, Compression::Raw).to_bytes().unwrap();
    // Layer & mask length (+4), layer info length (+4), 4 layers × channels × 4.
    let f = testgen::small(Version::Psd, Compression::Raw);
    let nch: usize = f.layers().iter().map(|l| l.channels.len()).sum();
    assert_eq!(psb.len(), psd.len() + 8 + nch * 4);
    assert_eq!(psb[5], 2);
}

#[test]
fn psb_rle_counts_are_u32() {
    let psd = testgen::merged_only(Version::Psd, ColorMode::Grayscale, 8, Compression::Rle, 5, 4);
    let psb = testgen::merged_only(Version::Psb, ColorMode::Grayscale, 8, Compression::Rle, 5, 4);
    assert_eq!(psb.image_data.data.len(), psd.image_data.data.len() + 4 * 2);
}

#[test]
fn psb_long_key_global_block() {
    let mut f = testgen::small(Version::Psb, Compression::Rle);
    f.global_blocks.push(TaggedBlock::new(*b"FMsk", vec![1, 2, 3, 4]));
    f.global_blocks.push(TaggedBlock::new(*b"Alph", vec![5, 6]));
    let b = f.to_bytes().unwrap();
    let p = PsdFile::from_bytes(&b).unwrap();
    assert_eq!(p, f);
    // Reading the PSB bytes while pretending PSD must not silently succeed
    // with the same structure.
    let mut as_psd = b.clone();
    as_psd[5] = 1;
    if let Ok(q) = PsdFile::from_bytes(&as_psd) {
        assert_ne!(q, f);
    }
}

#[test]
fn edits_are_reencoded() {
    let mut f = testgen::small(Version::Psd, Compression::Rle);
    let v = f.header.version;
    let layer = &mut f.layers_mut()[0];
    let (w, h) = layer.rect.size().unwrap();
    let new = vec![7u8; w * h];
    let ch = layer.channels.iter_mut().find(|c| c.id == 0).unwrap();
    ch.set_decoded(Compression::ZipPrediction, &new, w, h, 8, v).unwrap();
    let b = f.to_bytes().unwrap();
    let p = PsdFile::from_bytes(&b).unwrap();
    assert_eq!(p.layers()[0].decode_channel(0, 8, v).unwrap(), new);
    assert_eq!(p.layers()[0].channel(0).unwrap().compression, Some(Compression::ZipPrediction));
}

#[test]
fn renaming_changes_luni_and_roundtrips() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let l = &mut f.layers_mut()[0];
    *l.block_mut(b"luni").unwrap() = TaggedBlock::unicode_name("Renamed \u{263a}");
    let p = PsdFile::from_bytes(&f.to_bytes().unwrap()).unwrap();
    assert_eq!(p.layers()[0].name(), "Renamed \u{263a}");
}

#[test]
fn unknown_blocks_and_resources_pass_through() {
    let mut f = testgen::small(Version::Psd, Compression::Rle);
    f.resources.push(ImageResource::new(9999, vec![1, 2, 3, 4, 5]));
    f.global_blocks.push(TaggedBlock::new(*b"Q3Dx", (0..=255).collect()));
    f.layers_mut()[0].blocks.push(TaggedBlock::new(*b"Vid!", vec![0xab; 7]));
    f.layers_mut()[0].blend_mode = BlendMode::Unknown(*b"new!");
    assert_stable(&f, "unknown passthrough");
    let p = PsdFile::from_bytes(&f.to_bytes().unwrap()).unwrap();
    assert_eq!(p.layers()[0].blend_mode, BlendMode::Unknown(*b"new!"));
    assert_eq!(p.global_block(b"Q3Dx").unwrap().data.len(), 256);
}

#[test]
fn unknown_color_mode_passthrough() {
    let mut f = testgen::merged_only(Version::Psd, ColorMode::Grayscale, 8, Compression::Raw, 2, 2);
    f.header.color_mode = ColorMode::Unknown(5);
    assert_stable(&f, "unknown mode");
}

#[test]
fn unknown_compression_passthrough() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layers_mut()[0].channels[0].compression = Some(Compression::Unknown(9));
    assert_stable(&f, "unknown channel compression");
    assert!(f.layers()[0].decode_channel(f.layers()[0].channels[0].id, 8, Version::Psd).is_err());
}

#[test]
fn trailing_section_bytes_preserved() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layer_mask_trailing = vec![0; 6];
    assert_stable(&f, "layer mask trailing");
    f.layers_mut()[1].extra_trailing = vec![0; 5];
    assert_stable(&f, "record trailing");
    f.layer_info.as_mut().unwrap().padding = Some(vec![0, 0, 0, 0]);
    f.layers_mut()[1].extra_trailing.clear();
    assert_stable(&f, "layer info padding");
}

#[test]
fn global_mask_accessors() {
    let f = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Raw);
    let gm = f.global_layer_mask.as_ref().unwrap();
    assert_eq!(gm.overlay_color_space(), Some(0));
    assert_eq!(gm.color_components(), Some([0xffff, 0, 0, 0]));
    assert_eq!(gm.opacity(), Some(50));
    assert_eq!(gm.kind(), Some(128));
    assert_eq!(GlobalLayerMask::default().kind(), None);
}

#[test]
fn file_accessors() {
    let f = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
    assert!((f.resolution().unwrap().h_res() - 72.0).abs() < 1e-9);
    assert_eq!(f.icc_profile().unwrap().len(), 13);
    assert_eq!(f.has_real_merged_data(), Some(true));
    assert!(f.merged_has_alpha());
    assert!(f.validate().is_ok());
    let lfx = f.layers().iter().find_map(|l| l.block(b"lfx2")).expect("lfx2");
    let d = descriptor::VersionedDescriptor::from_bytes(&lfx.data[4..]).unwrap();
    assert_eq!(d, testgen::sample_descriptor());
}

#[test]
fn validate_rejects_bad_models() {
    let mut f = testgen::merged_only(Version::Psd, ColorMode::Indexed, 8, Compression::Raw, 2, 2);
    assert!(f.validate().is_ok());
    f.color_mode_data.pop();
    assert!(f.validate().is_err());
    let mut g = testgen::merged_only(Version::Psd, ColorMode::Rgb, 8, Compression::Raw, 2, 2);
    g.header.width = 40_000;
    assert!(g.validate().is_err());
    g.header.version = Version::Psb;
    assert!(g.validate().is_ok());
}

#[test]
fn psd_too_large_for_32bit_errors_on_write() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    // A resource larger than u32 can't be constructed cheaply; instead check
    // RLE rows > 65535 bytes are rejected for PSD but allowed for PSB.
    let l = photocraft_psd::compression::PlaneLayout { planes: 1, width: 70_000, height: 1, depth: 8, version: Version::Psd };
    let row: Vec<u8> = (0..70_000u32).map(|i| (i % 251) as u8).collect();
    assert!(compression::encode_planes(Compression::Rle, &row, &l).is_err());
    let l = compression::PlaneLayout { version: Version::Psb, ..l };
    assert!(compression::encode_planes(Compression::Rle, &row, &l).is_ok());
    f.header.version = Version::Psd;
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn save_and_open() {
    let f = testgen::small(Version::Psd, Compression::Rle);
    let dir = std::env::temp_dir().join(format!("photocraft-psd-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("x.psd");
    f.save(&p).unwrap();
    let g = PsdFile::open(&p).unwrap().unwrap();
    assert_eq!(g, f);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A density-only mask record written in the spec's order (parameters, then the real-mask
/// fields) used to be read back in Photoshop's order, which moved the density into the real
/// mask's background and dropped it from the parameters (#949). The model must survive the
/// round trip, not only the bytes.
#[test]
fn spec_order_density_only_mask_round_trips_as_a_model() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let layer = &mut f.layers_mut()[0];
    layer.mask = MaskData::Mask(LayerMask {
        rect: Rect::from_xywh(0, 0, 1, 1),
        default_color: 255,
        flags: LayerMask::FLAG_PARAMETERS,
        parameters: Some(MaskParameters { flags: 1, user_density: Some(64), ..Default::default() }),
        real: Some(RealMask { flags: 0, background: 0, rect: Rect::from_xywh(0, 0, 1, 1) }),
        trailing: vec![],
        real_first: false,
    });
    // The generated 2 x 2 mask channel is replaced by one that matches the 1 x 1 rectangle.
    let mask_channel = layer.channels.iter_mut().find(|c| c.id == -2).expect("the generated layer has a mask channel");
    *mask_channel = ChannelData::encode(-2, Compression::Raw, &[64], 1, 1, 8, Version::Psd).unwrap();
    let b = f.to_bytes().unwrap();
    let g = PsdFile::from_bytes(&b).unwrap();
    assert_eq!(g, f, "model round trip");
    assert_eq!(g.layers()[0].decode_channel(-2, 8, Version::Psd).unwrap(), vec![64]);
    assert_eq!(g.to_bytes().unwrap(), b, "byte stability");
}
