mod common;

use common::tiffgen::*;
use photocraft_codecs::{Image, decode, exif_orientation, upright_exif};

fn classic_white_is_zero() -> Vec<u8> {
    let width = 4u16;
    let height = 1u16;
    let pixels = [0u8, 64, 192, 255];
    let entries: [(u16, u16, u32); 9] = [
        (256, 3, u32::from(width)),
        (257, 3, u32::from(height)),
        (258, 3, 8),
        (259, 3, 1),
        (262, 3, 0),   // WhiteIsZero
        (273, 4, 122), // StripOffsets: immediately after IFD0.
        (277, 3, 1),
        (278, 3, u32::from(height)),
        (279, 4, pixels.len() as u32),
    ];
    let mut bytes = b"II*\0\x08\0\0\0".to_vec();
    bytes.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, field_type, value) in entries {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&field_type.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        if field_type == 3 {
            bytes.extend_from_slice(&(value as u16).to_le_bytes());
            bytes.extend_from_slice(&[0; 2]);
        } else {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&pixels);
    bytes
}

fn big_tiff_gray(stored: &[u8], width: u16, height: u16, orientation: u16) -> Vec<u8> {
    let mut entries = vec![
        (256u16, 3u16, 1u64, u64::from(width)),
        (257, 3, 1, u64::from(height)),
        (258, 3, 1, 8),
        (259, 3, 1, 1),
        (262, 3, 1, 1), // BlackIsZero
        (273, 4, 1, 0), // StripOffsets, patched below.
        (274, 3, 1, u64::from(orientation)),
        (277, 3, 1, 1),
        (278, 3, 1, u64::from(height)),
        (279, 4, 1, stored.len() as u64),
    ];
    entries.sort_by_key(|entry| entry.0);
    let pixel_offset = 16u64 + 8 + entries.len() as u64 * 20 + 8;
    if let Some(strip_offset) = entries.iter_mut().find(|entry| entry.0 == 273) {
        strip_offset.3 = pixel_offset;
    }

    let mut bytes = b"II+\0\x08\0\0\0\x10\0\0\0\0\0\0\0".to_vec();
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (tag, field_type, count, value) in entries {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&field_type.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(stored);
    bytes
}

fn stored_for_orientation_6_or_8(orientation: u16) -> Vec<u8> {
    match orientation {
        6 => vec![3, 6, 2, 5, 1, 4],
        8 => vec![4, 1, 5, 2, 6, 3],
        _ => unreachable!(),
    }
}

#[test]
fn white_is_zero_is_normalized_once_by_the_tiff_decoder() {
    let image = decode(&classic_white_is_zero()).expect("synthetic WhiteIsZero TIFF should decode");

    assert_eq!(image.dimensions(), (4, 1));
    assert_eq!(image.data(), &[255, 191, 63, 0]);
}

#[test]
fn big_tiff_orientation_6_and_8_are_applied_to_decoded_pixels() {
    let upright = [1u8, 2, 3, 4, 5, 6];

    for orientation in [6, 8] {
        let bytes = big_tiff_gray(&stored_for_orientation_6_or_8(orientation), 2, 3, orientation);
        let image: Image = decode(&bytes).expect("synthetic BigTIFF should decode");

        assert_eq!(exif_orientation(&bytes), orientation);
        assert_eq!(exif_orientation(&upright_exif(&bytes)), 1);
        assert_eq!(image.dimensions(), (3, 2), "orientation {orientation}");
        assert_eq!(image.data(), &upright, "orientation {orientation}");
    }
}

fn with_compression(mut dir: Dir, code: u64) -> Dir {
    for entry in &mut dir.entries {
        if entry.0 == 259 {
            entry.2 = vec![V::N(code)];
        }
    }
    dir
}

#[test]
fn jpeg_compressed_tiff_decodes() {
    // A flat mid-grey 16x16 image as a baseline JPEG stored in one strip (Compression = 7).
    let mut jpeg = Vec::new();
    let gray = vec![100u8; 16 * 16];
    jpeg_encoder::Encoder::new(&mut jpeg, 95).encode(&gray, 16, 16, jpeg_encoder::ColorType::Luma).expect("encode the JPEG strip");
    let mut dir = with_compression(strips(16, 16, 8, 1, 1, 16, &[0; 256]), 7);
    dir.chunks = vec![jpeg];
    let bytes = build(true, false, &[dir], &[0]).bytes;

    let image = decode(&bytes).expect("a JPEG-compressed TIFF should decode");
    assert_eq!(image.dimensions(), (16, 16));
    assert!(image.data().iter().all(|&v| v.abs_diff(100) <= 4), "{:?}", image.data().get(..4));
}

#[test]
fn ccitt_group4_tiff_decodes() {
    // 8x3 bilevel: a white row, then two black rows. Bits: V0 | H(white 0, black 8) | V0 V0 | EOFB.
    let bits = "1".to_string() + "001" + "00110101" + "000101" + "11" + "000000000001" + "000000000001";
    let mut data = Vec::new();
    for chunk in bits.as_bytes().chunks(8) {
        let mut byte = 0u8;
        for (i, c) in chunk.iter().enumerate() {
            if *c == b'1' {
                byte |= 0x80 >> i;
            }
        }
        data.push(byte);
    }
    let mut dir = with_compression(strips(8, 3, 1, 1, 0, 3, &[0; 3]), 4);
    dir.chunks = vec![data];
    let bytes = build(true, false, &[dir], &[0]).bytes;

    let image = decode(&bytes).expect("a CCITT G4 TIFF should decode");
    assert_eq!(image.dimensions(), (8, 3));
    let d = image.data();
    assert_eq!(d.len(), 24, "{d:?}");
    assert!(d[..8].iter().all(|&v| v == d[0]), "{d:?}");
    assert!(d[8..].iter().all(|&v| v == d[8]), "{d:?}");
    assert_ne!(d[0], d[8], "{d:?}");
    assert_eq!(d[0], 255, "white row of a WhiteIsZero image");
}

/// #4484: a TIFF saved with the default options must open in software that only reads
/// baseline TIFF (embroidery and plotter tools, older readers). Like Photoshop's default, that
/// means uncompressed strips with no predictor, classic offsets and unassociated alpha.
#[test]
fn default_tiff_is_baseline() {
    use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, SampleType, encode};
    use std::io::Cursor;
    use tiff::tags::Tag;

    for layout in [ChannelLayout::Rgb, ChannelLayout::Rgba] {
        let img = common::synth(33, 21, layout, SampleType::U8, 7, 0.4);
        let bytes = encode(&img, Format::Tiff, &EncodeOptions::default()).expect("encode");
        assert!(bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*"), "{layout:?}: not a classic TIFF");
        let mut dec = tiff::decoder::Decoder::new(Cursor::new(&bytes[..])).expect("open");
        assert_eq!(dec.get_tag_u32(Tag::Compression).ok(), Some(1), "{layout:?}: Compression");
        assert_eq!(dec.find_tag(Tag::Predictor).ok().flatten().and_then(|v| v.into_u16().ok()).unwrap_or(1), 1, "{layout:?}: Predictor");
        assert!(dec.find_tag(Tag::TileWidth).ok().flatten().is_none(), "{layout:?}: tiled");
        assert_eq!(dec.get_tag_u32(Tag::PlanarConfiguration).ok().unwrap_or(1), 1, "{layout:?}: planar");
        let extra = dec.find_tag(Tag::ExtraSamples).ok().flatten().and_then(|v| v.into_u16_vec().ok());
        assert_eq!(extra, if layout == ChannelLayout::Rgba { Some(vec![2]) } else { None }, "{layout:?}: ExtraSamples");
        assert_eq!(decode(&bytes).expect("decode").data(), img.data(), "{layout:?}: round trip");
    }
}
