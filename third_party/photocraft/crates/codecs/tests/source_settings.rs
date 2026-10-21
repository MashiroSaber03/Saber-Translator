//! Reading back the settings a lossy file was saved with: a JPEG's quality from its
//! quantisation tables, a WebP's lossy or lossless encoding (and its quality).

mod common;
use common::*;
use photocraft_codecs::*;

fn photo(w: u32, h: u32, layout: ChannelLayout) -> Image {
    synth(w, h, layout, SampleType::U8, 5, 0.03)
}

#[test]
fn jpeg_quality_is_estimated_within_two() {
    for layout in [ChannelLayout::Rgb, ChannelLayout::Gray] {
        let img = photo(40, 24, layout);
        for q in [1u8, 5, 10, 25, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 100] {
            let bytes = encode(&img, Format::Jpeg, &EncodeOptions { jpeg_quality: q, ..Default::default() }).unwrap();
            let est = jpeg_quality(&bytes).unwrap_or_else(|| panic!("{layout:?} q{q}: no estimate"));
            assert!(est.abs_diff(q) <= 2, "{layout:?}: saved at {q}, estimated {est}");
        }
    }
}

#[test]
fn jpeg_quality_of_hand_built_tables() {
    // A minimal JPEG prefix: SOI, one DQT holding the Annex K luminance table (quality 50) in
    // zig-zag order, then SOS.
    let luma: [u8; 64] = [
        16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49, 35, 37, 29, 40, 58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78,
        64, 68, 87, 69, 55, 56, 80, 109, 81, 87, 95, 98, 103, 104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99,
    ];
    let mut b = vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 67, 0x00];
    b.extend_from_slice(&luma);
    b.extend_from_slice(&[0xFF, 0xDA, 0, 2]);
    assert_eq!(jpeg_quality(&b), Some(50));
}

#[test]
fn hostile_jpeg_tables_give_no_estimate_and_never_panic() {
    let img = photo(16, 16, ChannelLayout::Rgb);
    let good = encode(&img, Format::Jpeg, &EncodeOptions { jpeg_quality: 70, ..Default::default() }).unwrap();
    // Every truncation of a real file.
    for n in 0..good.len().min(800) {
        let _ = jpeg_quality(&good[..n]);
    }
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0xFF],
        vec![0xFF, 0xD8],
        // A DQT whose length runs past the end.
        vec![0xFF, 0xD8, 0xFF, 0xDB, 0xFF, 0xFF, 0x00, 1, 2, 3],
        // A DQT of length 0 and 1 (less than its own length field).
        vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x00],
        vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x01],
        // A table cut short, a bad precision, a slot past 3, a 16-bit table cut mid-entry.
        [vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 12, 0x00], vec![9; 9]].concat(),
        [vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 67, 0x70], vec![9; 64]].concat(),
        [vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 67, 0x0F], vec![9; 64]].concat(),
        [vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 68, 0x10], vec![0xFF; 65]].concat(),
        // All-zero and all-0xFF tables.
        [vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 67, 0x00], vec![0; 64]].concat(),
        [vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 131, 0x10], vec![0xFF; 128]].concat(),
        vec![0xFF; 64],
    ];
    for c in &cases {
        let _ = jpeg_quality(c);
    }
    assert_eq!(jpeg_quality(&[0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x00]), None);
    assert_eq!(jpeg_quality(b"not a jpeg at all"), None);
    let png = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    assert_eq!(jpeg_quality(&png), None);
}

#[test]
fn webp_lossless_and_lossy_are_told_apart() {
    let img = photo(33, 21, ChannelLayout::Rgb);
    let lossless = encode(&img, Format::WebP, &EncodeOptions::default()).unwrap();
    assert_eq!(webp_encoding(&lossless), Some(WebpEncoding::Lossless));
    for q in [10u8, 30, 50, 75, 80, 90, 100] {
        let bytes = encode(&img, Format::WebP, &EncodeOptions { webp_lossless: false, webp_quality: q, ..Default::default() }).unwrap();
        let Some(WebpEncoding::Lossy { quality: Some(est) }) = webp_encoding(&bytes) else { panic!("q{q}: {:?}", webp_encoding(&bytes)) };
        assert!(est.abs_diff(q) <= 2, "lossy WebP saved at {q}, estimated {est}");
    }
    // With alpha and metadata (the extended container).
    let rgba = photo(20, 20, ChannelLayout::Rgba);
    let bytes = encode(&rgba, Format::WebP, &EncodeOptions { webp_lossless: false, webp_quality: 60, ..Default::default() }).unwrap();
    assert!(matches!(webp_encoding(&bytes), Some(WebpEncoding::Lossy { quality: Some(58..=62) })), "{:?}", webp_encoding(&bytes));
}

#[test]
fn hostile_webp_never_panics() {
    let img = photo(24, 24, ChannelLayout::Rgb);
    for opts in [EncodeOptions::default(), EncodeOptions { webp_lossless: false, ..Default::default() }] {
        let good = encode(&img, Format::WebP, &opts).unwrap();
        for n in 0..good.len().min(400) {
            let _ = webp_encoding(&good[..n]);
        }
        let mut flipped = good.clone();
        for i in 12..flipped.len().min(64) {
            flipped[i] ^= 0xA5;
            let _ = webp_encoding(&flipped);
        }
    }
    let mut huge = b"RIFF\0\0\0\0WEBPVP8 ".to_vec();
    huge.extend_from_slice(&u32::MAX.to_le_bytes());
    let _ = webp_encoding(&huge);
    let mut anmf = b"RIFF\0\0\0\0WEBPANMF".to_vec();
    anmf.extend_from_slice(&3u32.to_le_bytes());
    let _ = webp_encoding(&anmf);
    assert_eq!(webp_encoding(b"RIFF\0\0\0\0WAVE"), None);
}
