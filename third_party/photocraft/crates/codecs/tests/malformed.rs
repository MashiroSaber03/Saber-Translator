//! Malformed input must produce errors, never panics or huge allocations.

mod common;
use common::*;
use photocraft_codecs::*;
use proptest::prelude::*;

fn tight() -> DecodeOptions {
    DecodeOptions { limits: Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 22, max_alloc: 64 << 20 }, ..Default::default() }
}

fn samples() -> Vec<(Format, Vec<u8>)> {
    let mut v = Vec::new();
    for f in rw_formats() {
        for (l, s) in [(ChannelLayout::Rgba, SampleType::U8), (ChannelLayout::Gray, SampleType::U16), (ChannelLayout::Rgb, SampleType::F32)] {
            let mut img = synth(19, 13, l, s, 11, 0.3);
            img.icc = Some(sample_icc(300));
            img.meta.exif = Some(sample_exif());
            img.meta.xmp = Some(SAMPLE_XMP.into());
            v.push((f, encode(&img, f, &EncodeOptions::default()).unwrap()));
        }
    }
    // Read-only formats: Apple-encoded HEIC from `corpus/heif` (`cargo xtask corpus --heif`), a
    // single picture and a grid of tiles.
    #[cfg(all(feature = "corpus", feature = "heif"))]
    for name in ["rgb-strips-96.heic", "checker-1024.heic"] {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/heif/heic-rs").join(name);
        let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}: run `cargo xtask corpus --all`", p.display()));
        v.push((Format::Heif, bytes));
    }
    // Extra variants: interlaced PNG, ASCII PNM, CMYK JPEG.
    let img = synth(19, 13, ChannelLayout::Rgb, SampleType::U8, 1, 0.3);
    v.push((Format::Png, encode(&img, Format::Png, &EncodeOptions { png_interlaced: true, ..Default::default() }).unwrap()));
    v.push((Format::Pnm, b"P3\n2 2\n255\n0 1 2 3 4 5 6 7 8 9 10 11\n".to_vec()));
    v.push((Format::Jpeg, encode(&synth(19, 13, ChannelLayout::Cmyk, SampleType::U8, 1, 0.3), Format::Jpeg, &EncodeOptions::default()).unwrap()));
    v
}

fn no_panic(f: Format, bytes: &[u8]) {
    let _ = decode_with(bytes, &tight());
    let _ = decode_as_with(f, bytes, &tight());
}

#[test]
fn truncated_at_many_offsets() {
    for (f, bytes) in samples() {
        let n = bytes.len();
        let step = (n / 97).max(1);
        let mut cuts: Vec<usize> = (0..n.min(80)).collect();
        cuts.extend((0..n).step_by(step));
        cuts.extend(n.saturating_sub(40)..n);
        for cut in cuts {
            no_panic(f, &bytes[..cut]);
        }
    }
}

#[test]
fn truncated_files_mostly_error() {
    // Chopping off the second half must not silently succeed for formats
    // with length-checked rasters.
    for (f, bytes) in samples() {
        if matches!(f, Format::Png | Format::Pnm | Format::Fits | Format::Dpx | Format::Qoi | Format::OpenExr | Format::Tiff | Format::Bmp) {
            let r = decode_as_with(f, &bytes[..bytes.len() / 2], &tight());
            assert!(r.is_err(), "{f:?} decoded half a file");
        }
        // JPEG decodes leniently (the missing part grey) but never silently (#518).
        if f == Format::Jpeg {
            match decode_as_with(f, &bytes[..bytes.len() / 2], &tight()) {
                Ok(img) => assert_eq!(img.warnings, [DecodeWarning::Truncated { format: f }]),
                Err(e) => assert!(e.to_string().contains("before any image data"), "{e}"),
            }
        }
    }
}

#[test]
fn bit_flips_never_panic() {
    let mut rng = common::Rng::new(42);
    for (f, bytes) in samples() {
        for _ in 0..150 {
            let mut b = bytes.clone();
            let flips = 1 + (rng.next_u64() % 4) as usize;
            for _ in 0..flips {
                let i = (rng.next_u64() as usize) % b.len();
                b[i] ^= 1 << (rng.next_u64() % 8);
            }
            no_panic(f, &b);
        }
    }
}

#[test]
fn byte_overwrites_in_header_never_panic() {
    for (f, bytes) in samples() {
        for i in 0..bytes.len().min(64) {
            for v in [0x00, 0x7F, 0xFF] {
                let mut b = bytes.clone();
                b[i] = v;
                no_panic(f, &b);
            }
        }
    }
}

#[test]
fn empty_input_errors_for_every_format() {
    for f in Format::ALL {
        assert!(decode_as(f, &[]).is_err(), "{f:?}");
    }
}

#[test]
fn pnm_specific_malformations() {
    let cases: &[&[u8]] = &[
        b"P6\n",
        b"P6\n2 2\n",
        b"P6\n2 2\n255",
        b"P6\n2 2\n0\n",
        b"P6\n2 2\n70000\n",
        b"P6\n-2 2\n255\n",
        b"P6\n2 2\n255\n\x01\x02",
        b"P5\n2 2\n65535\n\x00\x00\x00",
        b"P2\n2 2\n10\n1 2 3 11\n",
        b"P1\n2 2\n0 1 2 0\n",
        b"P4\n9 2\n\x00",
        b"P7\nWIDTH 2\nHEIGHT 2\nDEPTH 9\nMAXVAL 255\nENDHDR\n",
        b"P7\nWIDTH 2\nHEIGHT 2\nMAXVAL 255\nENDHDR\n",
        b"P7\nWIDTH 2\nBOGUS 3\nENDHDR\n",
        b"PF\n2 2\n0.0\n",
        b"PF\n2 2\nnan\n",
        b"PF\n2 2\n-1.0\n\0\0\0\0",
        b"P6 4294967295 4294967295 255\n",
    ];
    for c in cases {
        assert!(decode(c).is_err(), "{:?}", String::from_utf8_lossy(c));
    }
}

#[test]
fn pnm_valid_variants() {
    let img = decode(b"P1\n# comment\n3 2\n1 0 1\n0 1 0\n").unwrap();
    assert_eq!(img.data(), &[0, 255, 0, 255, 0, 255]);
    let img = decode(b"P1\n3 2\n101010").unwrap();
    assert_eq!(img.data(), &[0, 255, 0, 255, 0, 255]);
    let img = decode(b"P4\n3 2\n\xA0\x40").unwrap();
    assert_eq!(img.data(), &[0, 255, 0, 255, 0, 255]);
    let img = decode(b"P2\n2 1\n15\n0 15\n").unwrap();
    assert_eq!(img.data(), &[0, 255]);
    let img = decode(b"P2\n2 1\n1023\n0 1023\n").unwrap();
    assert_eq!(img.sample_type(), SampleType::U16);
    assert_eq!(img.to_u16_samples().unwrap(), vec![0, 65535]);
    let img = decode(b"P3\n1 1\n255\n1 2 3\n").unwrap();
    assert_eq!((img.layout(), img.data()), (ChannelLayout::Rgb, &[1u8, 2, 3][..]));
    let img = decode(b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 2\nMAXVAL 255\nTUPLTYPE GRAYSCALE_ALPHA\nENDHDR\n\x10\x20").unwrap();
    assert_eq!((img.layout(), img.data()), (ChannelLayout::GrayA, &[0x10u8, 0x20][..]));
    let mut pfm = b"PF\n1 2\n1.0\n".to_vec(); // big-endian, bottom-up
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        pfm.extend_from_slice(&v.to_be_bytes());
    }
    let img = decode(&pfm).unwrap();
    assert_eq!(img.to_f32_samples().unwrap(), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn jpeg_marker_garbage() {
    let cases: &[&[u8]] = &[
        &[0xFF, 0xD8, 0xFF],
        &[0xFF, 0xD8, 0xFF, 0xE0, 0x00],
        &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01],
        &[0xFF, 0xD8, 0xFF, 0xE2, 0x00, 0x10, b'I', b'C', b'C', b'_', b'P', b'R', b'O', b'F', b'I', b'L', b'E', 0, 1, 1],
        &[0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 8, 0xFF, 0xFF, 0xFF, 0xFF, 3, 0, 0, 0],
        &[0xFF, 0xD8, 0xFF, 0xD9],
    ];
    for c in cases {
        assert!(decode(c).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, .. ProptestConfig::default() })]

    #[test]
    fn random_bytes_never_panic(data in proptest::collection::vec(any::<u8>(), 0..2048)) {
        let _ = decode_with(&data, &tight());
    }

    #[test]
    fn random_bytes_with_magic_never_panic(idx in 0usize..14, data in proptest::collection::vec(any::<u8>(), 0..1024)) {
        let magics: [&[u8]; 14] = [
            b"\x89PNG\r\n\x1a\n", &[0xFF, 0xD8, 0xFF], b"II*\0", b"RIFF\0\0\0\0WEBP", b"GIF89a", b"BM",
            &[0, 0, 1, 0, 1, 0], b"P6\n", b"qoif", &[0x76, 0x2F, 0x31, 0x01], b"#?RADIANCE\n", b"P7\n", b"PF\n",
            b"\0\0\0\x18ftypheic\0\0\0\0mif1heic",
        ];
        let mut b = magics[idx].to_vec();
        b.extend_from_slice(&data);
        let _ = decode_with(&b, &tight());
    }

    #[test]
    fn random_bytes_each_format_never_panic(idx in 0usize..Format::ALL.len(), data in proptest::collection::vec(any::<u8>(), 0..512)) {
        let _ = decode_as_with(Format::ALL[idx], &data, &tight());
    }

    #[test]
    fn mutated_valid_files_never_panic(which in 0usize..40, pos in any::<usize>(), val in any::<u8>(), cut in any::<usize>()) {
        let s = samples_cached();
        let (f, bytes) = &s[which % s.len()];
        let mut b = bytes.clone();
        let p = pos % b.len();
        b[p] = val;
        let c = cut % (b.len() + 1);
        let _ = decode_as_with(*f, &b[..c.max(p + 1).min(b.len())], &tight());
    }
}

fn samples_cached() -> &'static [(Format, Vec<u8>)] {
    static S: std::sync::OnceLock<Vec<(Format, Vec<u8>)>> = std::sync::OnceLock::new();
    S.get_or_init(samples)
}

/// A little-endian TIFF with one 1x1 8-bit grey strip and the given extra IFD entries
/// (tag, type, count, value), sorted into place.
fn tiny_tiff(extra: &[(u16, u16, u32, u32)]) -> Vec<u8> {
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    let mut entries = vec![
        (256, SHORT, 1, 1), // ImageWidth
        (257, SHORT, 1, 1), // ImageLength
        (258, SHORT, 1, 8), // BitsPerSample
        (259, SHORT, 1, 1), // Compression: none
        (262, SHORT, 1, 1), // Photometric: BlackIsZero
        (273, LONG, 1, 0),  // StripOffsets (patched below)
        (277, SHORT, 1, 1), // SamplesPerPixel
        (278, SHORT, 1, 1), // RowsPerStrip
        (279, LONG, 1, 1),  // StripByteCounts
    ];
    entries.extend_from_slice(extra);
    entries.sort_by_key(|e| e.0);
    let ifd_len = 2 + 12 * entries.len() + 4;
    let strip = 8 + ifd_len as u32;
    let mut b = b"II*\0".to_vec();
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, value) in entries {
        let value = if tag == 273 { strip } else { value };
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    b.push(0x80);
    b
}

#[test]
fn tiff_fixture_decodes() {
    let img = decode_as_with(Format::Tiff, &tiny_tiff(&[]), &tight()).unwrap();
    assert_eq!((img.width(), img.height()), (1, 1));
}

/// Regression: an empty SampleFormat tag made the tiff 0.10 decoder index an empty list
/// (`index out of bounds: the len is 0`). Found by mutation fuzzing; fixed by tiff 0.11.
#[test]
fn tiff_empty_sample_format_is_an_error() {
    let bytes = tiny_tiff(&[(339, 3, 0, 0)]);
    assert!(decode_as_with(Format::Tiff, &bytes, &tight()).is_err());
    assert!(decode_with(&bytes, &tight()).is_err());
}
