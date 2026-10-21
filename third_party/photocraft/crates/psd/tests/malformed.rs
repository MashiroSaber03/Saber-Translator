//! Malformed input handling: every error path must return `Err`, never panic.

use photocraft_psd::testgen;
use photocraft_psd::*;
use proptest::prelude::*;

fn truncation_sweep(bytes: &[u8], name: &str) {
    for cut in 0..bytes.len() {
        let r = PsdFile::from_bytes(&bytes[..cut]);
        assert!(r.is_err(), "{name}: truncation at {cut}/{} parsed successfully", bytes.len());
    }
    assert!(PsdFile::from_bytes(bytes).is_ok(), "{name}: full file must parse");
}

macro_rules! trunc {
    ($name:ident, $ver:expr, $comp:expr) => {
        #[test]
        fn $name() {
            let b = testgen::small($ver, $comp).to_bytes().unwrap();
            truncation_sweep(&b, stringify!($name));
        }
    };
}

trunc!(truncate_small_psd_raw, Version::Psd, Compression::Raw);
trunc!(truncate_small_psd_rle, Version::Psd, Compression::Rle);
trunc!(truncate_small_psd_zip, Version::Psd, Compression::Zip);
trunc!(truncate_small_psd_zipp, Version::Psd, Compression::ZipPrediction);
trunc!(truncate_small_psb_raw, Version::Psb, Compression::Raw);
trunc!(truncate_small_psb_rle, Version::Psb, Compression::Rle);
trunc!(truncate_small_psb_zip, Version::Psb, Compression::Zip);
trunc!(truncate_small_psb_zipp, Version::Psb, Compression::ZipPrediction);

#[test]
fn truncate_merged_only_all_modes() {
    for mode in testgen::MODES {
        for &depth in testgen::mode_depths(mode) {
            let b = testgen::merged_only(Version::Psd, mode, depth, Compression::Rle, 5, 3).to_bytes().unwrap();
            truncation_sweep(&b, &format!("{mode:?} {depth}"));
        }
    }
}

#[test]
fn truncate_builder_16bit_lr16() {
    let mut b = PsdBuilder::new(3, 2).depth(16);
    b.push_layer(LayerSpec::new("x", 0, 0, 3, 2, PixelData::Rgba16(vec![1000; 24])));
    let bytes = b.to_bytes().unwrap();
    truncation_sweep(&bytes, "builder 16");
}

#[test]
fn empty_input() {
    assert!(matches!(PsdFile::from_bytes(&[]), Err(PsdError::UnexpectedEof { .. })));
}

#[test]
fn bad_file_signature() {
    let mut b = testgen::small(Version::Psd, Compression::Raw).to_bytes().unwrap();
    b[..4].copy_from_slice(b"8BPX");
    assert!(matches!(PsdFile::from_bytes(&b), Err(PsdError::InvalidSignature { .. })));
}

#[test]
fn bad_version() {
    let mut b = testgen::small(Version::Psd, Compression::Raw).to_bytes().unwrap();
    for v in [0u16, 3, 0xffff] {
        b[4..6].copy_from_slice(&v.to_be_bytes());
        assert_eq!(PsdFile::from_bytes(&b), Err(PsdError::UnsupportedVersion(v)));
    }
}

#[test]
fn bad_depth_and_channels() {
    let good = testgen::small(Version::Psd, Compression::Raw).to_bytes().unwrap();
    let mut b = good.clone();
    b[22..24].copy_from_slice(&7u16.to_be_bytes());
    assert!(PsdFile::from_bytes(&b).is_err());
    let mut b = good.clone();
    b[12..14].copy_from_slice(&0u16.to_be_bytes());
    assert!(PsdFile::from_bytes(&b).is_err());
    let mut b = good;
    b[12..14].copy_from_slice(&57u16.to_be_bytes());
    assert!(PsdFile::from_bytes(&b).is_err());
}

#[test]
fn oversized_dimensions_rejected() {
    let mut b = testgen::small(Version::Psb, Compression::Raw).to_bytes().unwrap();
    b[14..18].copy_from_slice(&300_001u32.to_be_bytes());
    assert!(matches!(PsdFile::from_bytes(&b), Err(PsdError::LimitExceeded(_))));
}

#[test]
fn zero_dimensions_rejected() {
    // Height is bytes 14..18 and width 18..22; the spec range starts at 1.
    let good = testgen::small(Version::Psd, Compression::Raw).to_bytes().unwrap();
    for range in [14..18, 18..22, 14..22] {
        let mut b = good.clone();
        b[range.clone()].fill(0);
        let err = PsdFile::from_bytes(&b).unwrap_err();
        assert!(err.to_string().contains("at least 1x1"), "{range:?}: {err}");
    }
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.header.width = 0;
    assert!(f.validate().is_err(), "a zero-sized header must not pass the writer's validation");
}

#[test]
fn huge_declared_dimensions_small_data() {
    // 300000² RLE image with 16 bytes of data must fail fast, not allocate.
    let mut f = testgen::merged_only(Version::Psb, ColorMode::Grayscale, 8, Compression::Rle, 1, 1);
    f.header.width = 300_000;
    f.header.height = 300_000;
    let b = f.to_bytes().unwrap();
    assert!(PsdFile::from_bytes(&b).is_err());
    // ZIP over `MAX_DECODED_BYTES` is not inflated at parse time (#2926); decoding refuses it.
    f.image_data.compression = Compression::Zip;
    f.image_data.data = compression::zip_compress(&[0; 100]);
    let b = f.to_bytes().unwrap();
    assert!(matches!(PsdFile::from_bytes(&b).and_then(|f| f.decode_merged()), Err(PsdError::LimitExceeded(_))));
}

#[test]
fn bad_resource_signature() {
    let f = testgen::small(Version::Psd, Compression::Raw);
    let mut b = f.to_bytes().unwrap();
    // header 26 + color mode len 4 + resources len 4 → first resource sig.
    b[34..38].copy_from_slice(b"XXXX");
    assert!(matches!(PsdFile::from_bytes(&b), Err(PsdError::InvalidSignature { .. })));
}

#[test]
fn bad_layer_blend_signature() {
    let f = testgen::small(Version::Psd, Compression::Raw);
    let b = f.to_bytes().unwrap();
    let pos = b.windows(8).position(|w| w == b"8BIMnorm").expect("blend sig");
    let mut b2 = b.clone();
    b2[pos..pos + 4].copy_from_slice(b"XBIM");
    assert!(PsdFile::from_bytes(&b2).is_err());
}

#[test]
fn oversized_section_lengths() {
    let f = testgen::small(Version::Psd, Compression::Raw);
    let b = f.to_bytes().unwrap();
    // Color mode data length.
    let mut b2 = b.clone();
    b2[26..30].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(PsdFile::from_bytes(&b2).is_err());
    // Resource section length.
    let mut b3 = b.clone();
    b3[30..34].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(PsdFile::from_bytes(&b3).is_err());
}

#[test]
fn corrupt_zip_merged_rejected() {
    let mut f = testgen::small(Version::Psd, Compression::Zip);
    let n = f.image_data.data.len();
    f.image_data.data[n / 2] ^= 0xff;
    f.image_data.data[n / 2 + 1] ^= 0x55;
    let b = f.to_bytes().unwrap();
    assert!(PsdFile::from_bytes(&b).is_err());
}

#[test]
fn corrupt_layer_channel_is_lazy_error() {
    let mut f = testgen::small(Version::Psd, Compression::Rle);
    let ch = &mut f.layers_mut()[0].channels[1];
    let id = ch.id;
    for b in ch.data.iter_mut() {
        *b = 0x7f;
    }
    let bytes = f.to_bytes().unwrap();
    let p = PsdFile::from_bytes(&bytes).expect("lazy: parse succeeds");
    assert!(p.layers()[0].decode_channel(id, 8, Version::Psd).is_err());
    assert!(p.layer(0).unwrap().rgba8().is_err());
}

#[test]
fn negative_layer_rect_decode_errors() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layers_mut()[0].rect = Rect { top: 10, left: 0, bottom: 0, right: 4 };
    let b = f.to_bytes().unwrap();
    let p = PsdFile::from_bytes(&b).unwrap();
    assert!(p.layers()[0].decode_channel(0, 8, Version::Psd).is_err());
    assert!(p.layer(0).unwrap().rgba8().is_err());
}

#[test]
fn too_many_layer_channels() {
    let f = testgen::small(Version::Psd, Compression::Raw);
    let b = f.to_bytes().unwrap();
    let li = b.windows(4).position(|w| w == [0, 0xff, 0xff, 0xff]).is_some();
    let _ = li;
    // Construct directly: layer count 1, rect, channel count 0xffff.
    let mut f2 = testgen::merged_only(Version::Psd, ColorMode::Rgb, 8, Compression::Raw, 1, 1);
    f2.layer_info = Some(LayerInfo::default());
    let mut b2 = f2.to_bytes().unwrap();
    // Find layer info: after resources. Locate the 2-byte count (0) and
    // replace with a malformed record.
    let res_len = u32::from_be_bytes(b2[30..34].try_into().unwrap()) as usize;
    let lm = 34 + res_len;
    let mut rec = vec![0, 1];
    rec.extend_from_slice(&[0; 16]);
    rec.extend_from_slice(&0xffffu16.to_be_bytes());
    let li_len = rec.len() as u32;
    let mut section = li_len.to_be_bytes().to_vec();
    section.extend_from_slice(&rec);
    let mut out = b2[..lm].to_vec();
    out.extend_from_slice(&(section.len() as u32).to_be_bytes());
    out.extend_from_slice(&section);
    let old_lm_len = u32::from_be_bytes(b2[lm..lm + 4].try_into().unwrap()) as usize;
    out.extend_from_slice(&b2[lm + 4 + old_lm_len..]);
    b2 = out;
    assert!(matches!(PsdFile::from_bytes(&b2), Err(PsdError::LimitExceeded(_)) | Err(PsdError::UnexpectedEof { .. })));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn random_bytes_never_panic(data in proptest::collection::vec(any::<u8>(), 0..512)) {
        let _ = PsdFile::from_bytes(&data);
    }

    #[test]
    fn random_bytes_with_valid_header_never_panic(tail in proptest::collection::vec(any::<u8>(), 0..512)) {
        let mut data = testgen::small(Version::Psd, Compression::Raw).to_bytes().unwrap()[..26].to_vec();
        data.extend(tail);
        let _ = PsdFile::from_bytes(&data);
    }

    #[test]
    fn mutations_never_panic(
        idx in proptest::collection::vec(any::<prop::sample::Index>(), 1..8),
        vals in proptest::collection::vec(any::<u8>(), 8),
        psb in any::<bool>(),
        comp in 0usize..4,
    ) {
        let v = if psb { Version::Psb } else { Version::Psd };
        let mut b = testgen::small(v, Compression::ALL[comp]).to_bytes().unwrap();
        for (i, ix) in idx.iter().enumerate() {
            let at = ix.index(b.len());
            b[at] = vals[i];
        }
        if let Ok(f) = PsdFile::from_bytes(&b) {
            // Anything that parses must also be writable and decodable
            // without panicking.
            let _ = f.to_bytes();
            let _ = f.decode_merged();
            let _ = f.composite_rgba8();
            let _ = f.layer_tree();
            for l in f.iter_layers() {
                let _ = l.rgba8();
                let _ = l.user_mask();
                let _ = l.name();
                for bl in &l.record.blocks {
                    let _ = bl.parsed();
                }
            }
            for r in &f.resources {
                let _ = r.parsed();
            }
        }
    }

    #[test]
    fn parsed_mutants_are_byte_stable(
        idx in any::<prop::sample::Index>(),
        val in any::<u8>(),
    ) {
        let mut b = testgen::small(Version::Psd, Compression::Rle).to_bytes().unwrap();
        let at = idx.index(b.len());
        b[at] = val;
        if let Ok(f) = PsdFile::from_bytes(&b) {
            let out = f.to_bytes().unwrap();
            let g = PsdFile::from_bytes(&out).unwrap();
            prop_assert_eq!(g, f);
        }
    }
}
