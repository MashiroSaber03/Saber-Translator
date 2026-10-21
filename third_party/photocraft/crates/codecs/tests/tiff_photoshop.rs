//! Photoshop's private TIFF tags (34377 image resources, 37724 ImageSourceData) travel through
//! `Metadata` as opaque bytes: written when present, read from either byte order and from
//! BigTIFF, and never a panic or an allocation on malformed input.

mod common;
use common::*;
use photocraft_codecs::*;

fn with_tags(layout: ChannelLayout, sample: SampleType) -> Image {
    let mut img = synth(9, 7, layout, sample, 5, 0.3);
    img.meta.photoshop_resources = Some(b"8BIM\x03\xed\x00\x00\x00\x00\x00\x10\x00\x48\x00\x00\x00\x01\x00\x01\x00\x48\x00\x00\x00\x01\x00\x01".to_vec());
    let mut layers = b"Adobe Photoshop Document Data Block\0".to_vec();
    layers.extend_from_slice(b"MIB8ryaL");
    layers.extend_from_slice(&4u32.to_le_bytes());
    layers.extend_from_slice(&[0, 0, 0, 0]);
    img.meta.photoshop_layers = Some(layers);
    img
}

#[test]
fn tags_round_trip_through_the_encoder() {
    for (layout, sample) in [
        (ChannelLayout::Rgb, SampleType::U8),
        (ChannelLayout::Rgba, SampleType::U16),
        (ChannelLayout::Gray, SampleType::F32),
        (ChannelLayout::CmykA, SampleType::U8),
    ] {
        for compression in [TiffCompression::None, TiffCompression::Deflate, TiffCompression::Lzw, TiffCompression::PackBits] {
            let img = with_tags(layout, sample);
            let bytes = encode(&img, Format::Tiff, &EncodeOptions { tiff_compression: compression, ..Default::default() }).unwrap();
            // The encoder writes the running target's byte order.
            assert_eq!(bytes.starts_with(b"II"), tiff_writes_little_endian());
            let (res, layers) = tiff_photoshop_tags(&bytes);
            assert_eq!(res, img.meta.photoshop_resources.as_deref());
            assert_eq!(layers, img.meta.photoshop_layers.as_deref());
            let back = decode(&bytes).unwrap();
            assert_eq!(back.meta.photoshop_resources, img.meta.photoshop_resources, "{layout:?} {sample:?} {compression:?}");
            assert_eq!(back.meta.photoshop_layers, img.meta.photoshop_layers);
            assert_eq!(back.data(), img.data());
        }
    }
}

#[test]
fn absent_and_empty_tags_are_not_written() {
    let mut img = synth(4, 4, ChannelLayout::Rgb, SampleType::U8, 1, 0.0);
    let bytes = encode(&img, Format::Tiff, &EncodeOptions::default()).unwrap();
    assert_eq!(tiff_photoshop_tags(&bytes), (None, None));
    assert!(decode(&bytes).unwrap().meta.photoshop_layers.is_none());
    img.meta.photoshop_layers = Some(Vec::new());
    img.meta.photoshop_resources = Some(Vec::new());
    let bytes = encode(&img, Format::Tiff, &EncodeOptions::default()).unwrap();
    assert_eq!(tiff_photoshop_tags(&bytes), (None, None));
}

/// A minimal uncompressed 2×1 gray TIFF with a 37724 tag, in either byte order, classic or
/// BigTIFF: `payload` is stored inline when it fits the value field, at an offset otherwise.
fn hand_made(little: bool, big: bool, payload: &[u8]) -> Vec<u8> {
    let w16 = |v: u16| if little { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let w32 = |v: u32| if little { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let w64 = |v: u64| if little { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let mut f = Vec::new();
    f.extend_from_slice(if little { b"II" } else { b"MM" });
    let header_len = if big { 16 } else { 8 };
    let pixels_at = header_len;
    let ifd_at = pixels_at + 2;
    f.extend(w16(if big { 43 } else { 42 }));
    if big {
        f.extend(w16(8));
        f.extend(w16(0));
        f.extend(w64(ifd_at as u64));
    } else {
        f.extend(w32(ifd_at as u32));
    }
    f.extend_from_slice(&[0x40, 0xc0]);
    // Entries: width, length, bits, compression, photometric, strip offsets, samples, rows per
    // strip, strip byte counts, 37724.
    let tags: [(u16, u16, u64, u64); 10] = [
        (256, 3, 1, 2),
        (257, 3, 1, 1),
        (258, 3, 1, 8),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, pixels_at as u64),
        (277, 3, 1, 1),
        (278, 3, 1, 1),
        (279, 4, 1, 2),
        (37724, 7, payload.len() as u64, 0),
    ];
    let entry_len = if big { 20 } else { 12 };
    let inline = if big { 8 } else { 4 };
    let count_len = if big { 8 } else { 2 };
    let next_len = if big { 8 } else { 4 };
    let data_at = ifd_at + count_len + tags.len() * entry_len + next_len;
    let mut entries = Vec::new();
    for (tag, ty, count, value) in tags {
        entries.extend(w16(tag));
        entries.extend(w16(ty));
        entries.extend(if big { w64(count) } else { w32(count as u32) });
        let mut v = Vec::new();
        if tag == 37724 {
            if payload.len() <= inline {
                v.extend_from_slice(payload);
            } else {
                v.extend(if big { w64(data_at as u64) } else { w32(data_at as u32) });
            }
        } else if ty == 3 {
            v.extend(w16(value as u16));
        } else {
            v.extend(if big { w64(value) } else { w32(value as u32) });
        }
        v.resize(inline, 0);
        entries.extend(v);
    }
    f.extend(if big { w64(tags.len() as u64) } else { w16(tags.len() as u16) });
    f.extend(entries);
    f.extend(if big { w64(0) } else { w32(0) });
    assert_eq!(f.len(), data_at);
    if payload.len() > inline {
        f.extend_from_slice(payload);
    }
    f
}

#[test]
fn hand_made_files_in_every_layout() {
    let long = b"Adobe Photoshop Document Data Block\0MIB8".to_vec();
    for little in [false, true] {
        for big in [false, true] {
            for payload in [&b"ab"[..], &b"abcd"[..], &b"abcdefgh"[..], &long[..]] {
                let f = hand_made(little, big, payload);
                let (res, layers) = tiff_photoshop_tags(&f);
                assert_eq!(res, None);
                assert_eq!(layers, Some(payload), "little {little} big {big} len {}", payload.len());
                let img = decode(&f).unwrap();
                assert_eq!(img.dimensions(), (2, 1));
                assert_eq!(img.meta.photoshop_layers.as_deref(), Some(payload));
            }
        }
    }
}

#[test]
fn malformed_tags_read_as_absent_without_panicking() {
    let long = vec![7u8; 40];
    for little in [false, true] {
        for big in [false, true] {
            let f = hand_made(little, big, &long);
            // Truncated anywhere: never a panic; the payload is absent once it is cut.
            for cut in 0..f.len() {
                let (_, layers) = tiff_photoshop_tags(&f[..cut]);
                if cut < f.len() {
                    assert!(layers.is_none_or(|l| l.len() == 40));
                }
                let _ = decode(&f[..cut]);
            }
            // Bit flips in the directory.
            for i in 0..f.len() - 40 {
                let mut g = f.clone();
                g[i] ^= 0xff;
                let _ = tiff_photoshop_tags(&g);
                let _ = decode(&g);
            }
        }
    }
    assert_eq!(tiff_photoshop_tags(b""), (None, None));
    assert_eq!(tiff_photoshop_tags(b"II*\0"), (None, None));
    assert_eq!(tiff_photoshop_tags(b"MM\0*\xff\xff\xff\xff"), (None, None));
    // A count that overflows the file.
    let mut f = hand_made(true, false, &long);
    let at = f.windows(2).position(|w| w == 37724u16.to_le_bytes()).unwrap();
    f[at + 4..at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(tiff_photoshop_tags(&f).1, None);
}
