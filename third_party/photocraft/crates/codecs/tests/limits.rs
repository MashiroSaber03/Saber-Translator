//! Decompression-bomb guards.

mod common;
use common::*;
use photocraft_codecs::*;

fn opts(l: Limits) -> DecodeOptions {
    DecodeOptions { limits: l, ..Default::default() }
}

fn is_limit(r: Result<Image, CodecError>) -> bool {
    matches!(r, Err(CodecError::LimitExceeded(_)))
}

fn encoded(f: Format) -> Vec<u8> {
    let img = synth(64, 48, ChannelLayout::Rgba, SampleType::U8, 1, 0.1);
    encode(&img, f, &EncodeOptions::default()).unwrap()
}

#[test]
fn max_pixels_enforced_for_every_format() {
    for f in rw_formats() {
        let b = encoded(f);
        let l = Limits { max_pixels: 1000, ..Limits::default() };
        assert!(is_limit(decode_with(&b, &opts(l))), "{f:?}");
        let l = Limits { max_pixels: 64 * 48, ..Limits::default() };
        assert!(decode_with(&b, &opts(l)).is_ok(), "{f:?} exact limit should pass");
    }
}

#[test]
fn max_width_enforced_for_every_format() {
    for f in rw_formats() {
        let l = Limits { max_width: 63, ..Limits::default() };
        assert!(is_limit(decode_with(&encoded(f), &opts(l))), "{f:?}");
    }
}

#[test]
fn max_height_enforced_for_every_format() {
    for f in rw_formats() {
        let l = Limits { max_height: 47, ..Limits::default() };
        assert!(is_limit(decode_with(&encoded(f), &opts(l))), "{f:?}");
    }
}

#[test]
fn max_alloc_enforced_for_every_format() {
    for f in rw_formats() {
        let l = Limits { max_alloc: 64 * 48, ..Limits::default() };
        assert!(decode_with(&encoded(f), &opts(l)).is_err(), "{f:?}");
    }
}

#[test]
fn limits_none_accepts() {
    for f in rw_formats() {
        assert!(decode_with(&encoded(f), &opts(Limits::none())).is_ok(), "{f:?}");
    }
}

fn png_chunk(ty: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut v = (data.len() as u32).to_be_bytes().to_vec();
    v.extend_from_slice(ty);
    v.extend_from_slice(data);
    let mut crc_in = ty.to_vec();
    crc_in.extend_from_slice(data);
    v.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    v
}

fn crc32(d: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in d {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

#[test]
fn png_bomb_header_rejected_before_alloc() {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&100_000u32.to_be_bytes());
    ihdr.extend_from_slice(&100_000u32.to_be_bytes());
    ihdr.extend_from_slice(&[16, 6, 0, 0, 0]);
    let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
    b.extend(png_chunk(b"IHDR", &ihdr));
    b.extend(png_chunk(b"IDAT", &[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]));
    b.extend(png_chunk(b"IEND", &[]));
    assert!(is_limit(decode(&b)));
}

#[test]
fn png_header_at_the_default_limit_needs_a_plausible_file_size() {
    // 32768 x 32768 x RGBA16 is exactly the default max_alloc: it passes the dimension limits,
    // but a ~100 byte file cannot hold it (zlib tops out near 1032:1), so it is refused before
    // the 8 GiB output buffer is allocated.
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&32768u32.to_be_bytes());
    ihdr.extend_from_slice(&32768u32.to_be_bytes());
    ihdr.extend_from_slice(&[16, 6, 0, 0, 0]);
    let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
    b.extend(png_chunk(b"IHDR", &ihdr));
    b.extend(png_chunk(b"IDAT", &[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]));
    b.extend(png_chunk(b"IEND", &[]));
    assert!(is_limit(decode(&b)));
}

#[test]
fn png_highly_compressible_image_still_decodes() {
    let img = Image::from_u8(3000, 3000, ChannelLayout::Rgba, vec![255u8; 3000 * 3000 * 4]).unwrap();
    let b = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    assert!(b.len() < 100_000, "{}", b.len());
    assert_eq!(decode(&b).unwrap().width(), 3000);
}

const PNG_TEXT_BUDGET: usize = 64 << 10;

fn png_text_decode(bytes: &[u8], max_alloc: usize) -> Result<Image, CodecError> {
    decode_with(bytes, &opts(Limits { max_alloc: max_alloc as u64, ..Limits::default() }))
}

fn zlib(text: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(text).unwrap();
    encoder.finish().unwrap()
}

// zTXt and iTXt payloads supplied here are already compressed.
fn png_text_stream(ty: &[u8; 4], keyword: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut data = keyword.to_vec();
    data.push(0);
    match ty {
        b"tEXt" => {}
        b"zTXt" => data.push(0),
        b"iTXt" => data.extend_from_slice(&[1, 0, 0, 0]),
        _ => panic!("unexpected text chunk"),
    }
    data.extend_from_slice(payload);
    png_chunk(ty, &data)
}

fn png_text(ty: &[u8; 4], keyword: &[u8], text: &[u8]) -> Vec<u8> {
    if ty == b"tEXt" { png_text_stream(ty, keyword, text) } else { png_text_stream(ty, keyword, &zlib(text)) }
}

fn png_with_text(before_idat: &[Vec<u8>], after_idat: &[Vec<u8>]) -> Vec<u8> {
    let mut image = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut image, 1, 1);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[127]).unwrap();
        writer.finish().unwrap();
    }
    let mut at = 8;
    let mut idat = None;
    let iend = loop {
        let len = u32::from_be_bytes(image[at..at + 4].try_into().unwrap()) as usize;
        let ty = &image[at + 4..at + 8];
        if ty == b"IDAT" && idat.is_none() {
            idat = Some(at);
        }
        if ty == b"IEND" {
            break at;
        }
        at += len + 12;
    };
    let idat = idat.unwrap();
    let mut output = image[..idat].to_vec();
    for chunk in before_idat {
        output.extend_from_slice(chunk);
    }
    output.extend_from_slice(&image[idat..iend]);
    for chunk in after_idat {
        output.extend_from_slice(chunk);
    }
    output.extend_from_slice(&image[iend..]);
    output
}

#[test]
fn png_compressed_text_bombs_obey_limits_before_and_after_idat() {
    let bomb = vec![b'A'; PNG_TEXT_BUDGET * 8];
    for ty in [b"zTXt", b"iTXt"] {
        for after_idat in [false, true] {
            let chunks = [png_text(ty, b"Comment", &bomb)];
            let bytes = if after_idat { png_with_text(&[], &chunks) } else { png_with_text(&chunks, &[]) };
            assert!(bytes.len() < PNG_TEXT_BUDGET / 2, "fixture must fit comfortably within the compressed-input budget");
            assert!(is_limit(png_text_decode(&bytes, PNG_TEXT_BUDGET)), "{ty:?}, after IDAT={after_idat}");
        }
    }
}

#[test]
fn png_text_chunks_share_an_aggregate_budget() {
    for text_len in [(PNG_TEXT_BUDGET - 3) / 3, PNG_TEXT_BUDGET / 3] {
        let text = vec![b'A'; text_len];
        let before = [png_text(b"tEXt", b"A", &text), png_text(b"zTXt", b"B", &text)];
        let after = [png_text(b"iTXt", b"C", &text)];
        let result = png_text_decode(&png_with_text(&before, &after), PNG_TEXT_BUDGET);
        if 3 * (text_len + 1) <= PNG_TEXT_BUDGET {
            let image = result.unwrap();
            assert_eq!(image.meta.text.len(), 3);
            assert!(image.meta.text.iter().all(|(_, value)| value.len() == text_len));
        } else {
            assert!(is_limit(result), "chunks fitting individually must still share a budget");
        }
    }
}

#[test]
fn png_text_accepts_the_exact_utf8_budget() {
    for ty in [b"tEXt", b"zTXt", b"iTXt"] {
        let text = vec![b'A'; PNG_TEXT_BUDGET - 1];
        let bytes = png_with_text(&[png_text(ty, b"K", &text)], &[]);
        let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
        assert_eq!(image.meta.text, vec![("K".into(), String::from_utf8(text).unwrap())]);
        assert!(is_limit(png_text_decode(&bytes, PNG_TEXT_BUDGET - 1)), "{ty:?}: keyword and text both count");
    }
}

#[test]
fn png_plain_itxt_obeys_the_exact_budget_and_validates_fields() {
    let mut payload = b"K\0\0\0en\0Keyword\0".to_vec();
    payload.extend_from_slice(&vec![b'A'; PNG_TEXT_BUDGET - 1]);
    let bytes = png_with_text(&[], &[png_chunk(b"iTXt", &payload)]);
    let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
    assert_eq!(image.meta.text, vec![("K".into(), "A".repeat(PNG_TEXT_BUDGET - 1))]);
    assert!(is_limit(png_text_decode(&bytes, PNG_TEXT_BUDGET - 1)));

    let invalid_fields: &[&[u8]] = &[
        b"K\0\0\0\0\0\xFF",     // Invalid text UTF-8.
        b"K\0\0\0\xFF\0\0text", // Non-ASCII language tag.
        b"K\0\0\0\0\xFF\0text", // Invalid translated-keyword UTF-8.
        b"K\0\x02\0\0\0text",   // Unknown compression flag.
        b"K\0\x01\x01\0\0text", // Unknown compression method for compressed text.
        b"K\0\0",               // Missing compression method.
        b"K\0\0\0\0text",       // Missing translated-keyword separator.
    ];
    for fields in invalid_fields {
        let bytes = png_with_text(&[png_chunk(b"iTXt", fields)], &[png_text(b"tEXt", b"Good", b"retained")]);
        let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
        assert_eq!(image.meta.text, vec![("Good".into(), "retained".into())]);
    }
}

#[test]
fn png_latin1_expansion_counts_keyword_and_text_utf8_bytes() {
    let text_len = (PNG_TEXT_BUDGET - 2) / 2;
    for ty in [b"tEXt", b"zTXt", b"iTXt"] {
        let text = if ty == b"iTXt" { "é".repeat(text_len).into_bytes() } else { vec![0xE9; text_len] };
        let bytes = png_with_text(&[], &[png_text(ty, &[0xE9], &text)]);
        let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
        assert_eq!(image.meta.text, vec![("é".into(), "é".repeat(text_len))]);
        assert!(is_limit(png_text_decode(&bytes, PNG_TEXT_BUDGET - 1)), "{ty:?}: Latin-1 expands when stored as UTF-8");
    }
}

#[test]
fn png_duplicate_xmp_chunks_each_consume_budget() {
    const KEYWORD: &[u8] = b"XML:com.adobe.xmp";
    let text_len = (PNG_TEXT_BUDGET - 2 * KEYWORD.len()) / 2;
    for extra in [0, 1] {
        let first = png_text(b"iTXt", KEYWORD, &vec![b'A'; text_len + extra]);
        let last_text = vec![b'B'; text_len];
        let last = png_text(b"iTXt", KEYWORD, &last_text);
        let result = png_text_decode(&png_with_text(&[first], &[last]), PNG_TEXT_BUDGET);
        if extra == 0 {
            let image = result.unwrap();
            assert_eq!(image.meta.xmp, Some(String::from_utf8(last_text).unwrap()));
            assert!(image.meta.text.is_empty());
        } else {
            assert!(is_limit(result), "replacing an XMP packet must not reset its budget charge");
        }
    }
}

#[test]
fn png_xmp_in_all_text_chunk_types_is_preserved() {
    const XMP: &str = "<x:xmpmeta xmlns:x='adobe:ns:meta/'/>";
    for ty in [b"tEXt", b"zTXt", b"iTXt"] {
        for after_idat in [false, true] {
            let chunks = [png_text(ty, b"XML:com.adobe.xmp", XMP.as_bytes())];
            let bytes = if after_idat { png_with_text(&[], &chunks) } else { png_with_text(&chunks, &[]) };
            let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
            assert_eq!(image.meta.xmp.as_deref(), Some(XMP), "{ty:?}, after IDAT: {after_idat}");
            assert!(image.meta.text.is_empty(), "XMP must not also be retained as ordinary text");
            let saved = encode(&image, Format::Png, &EncodeOptions::default()).unwrap();
            assert_eq!(decode(&saved).unwrap().meta.xmp, image.meta.xmp);
        }
    }
}

#[test]
fn png_mixed_xmp_chunk_types_use_last_packet_and_share_budget() {
    for first_ty in [b"tEXt", b"zTXt", b"iTXt"] {
        for last_ty in [b"tEXt", b"zTXt", b"iTXt"] {
            let first = png_text(first_ty, b"XML:com.adobe.xmp", b"first");
            let last = png_text(last_ty, b"XML:com.adobe.xmp", b"last");
            let bytes = png_with_text(&[first], &[last]);
            let budget = 2 * b"XML:com.adobe.xmp".len() + b"first".len() + b"last".len();
            // Pixel decoding needs a larger limit, so check the shared budget with larger packets.
            let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
            assert_eq!(image.meta.xmp.as_deref(), Some("last"));
            assert!(image.meta.text.is_empty());
            let first = png_text(first_ty, b"XML:com.adobe.xmp", &vec![b'A'; PNG_TEXT_BUDGET - budget]);
            let last = png_text(last_ty, b"XML:com.adobe.xmp", b"last");
            let bytes = png_with_text(&[first], &[last]);
            assert!(png_text_decode(&bytes, PNG_TEXT_BUDGET - b"first".len()).is_ok());
            assert!(is_limit(png_text_decode(&bytes, PNG_TEXT_BUDGET - b"first".len() - 1)));
        }
    }
}

#[test]
fn png_corrupt_text_is_still_skipped() {
    let valid = zlib(b"discard");
    let mut bad_header = valid.clone();
    bad_header[0] = 0;
    let truncated = valid[..valid.len() - 4].to_vec();
    for ty in [b"zTXt", b"iTXt"] {
        for stream in [&bad_header, &truncated] {
            let before = [png_text_stream(ty, b"Bad", stream)];
            let after = [png_text(b"tEXt", b"Good", b"retained")];
            let image = png_text_decode(&png_with_text(&before, &after), PNG_TEXT_BUDGET).unwrap();
            assert_eq!(image.meta.text, vec![("Good".into(), "retained".into())]);
        }
    }
    let invalid_utf8 = png_text(b"iTXt", b"Bad", &[0xFF]);
    let image = png_text_decode(&png_with_text(&[], &[invalid_utf8]), PNG_TEXT_BUDGET).unwrap();
    assert!(image.meta.text.is_empty());
}

#[test]
fn png_text_with_bad_crc_and_text_after_iend_are_ignored() {
    let bomb = vec![b'A'; PNG_TEXT_BUDGET * 8];
    for ty in [b"zTXt", b"iTXt"] {
        let valid = png_text(ty, b"Comment", &bomb);
        let mut bad_crc = valid.clone();
        *bad_crc.last_mut().unwrap() ^= 1;
        for after_idat in [false, true] {
            let bytes = if after_idat { png_with_text(&[], &[bad_crc.clone()]) } else { png_with_text(&[bad_crc.clone()], &[]) };
            let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
            assert!(image.meta.text.is_empty());
        }
        let mut bytes = png_with_text(&[], &[]);
        bytes.extend_from_slice(&valid);
        let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
        assert!(image.meta.text.is_empty());
    }
}

#[test]
fn png_malformed_chunk_tails_remain_tolerated_after_pixels() {
    let base = png_with_text(&[png_text(b"tEXt", b"Good", b"retained")], &[png_chunk(b"vpAg", &[])]);
    let iend = base.len() - 12;
    let tails: &[&[u8]] = &[&[], b"\x00\x00\x00", b"\x00\x00\x00\x10iTXtK\0\0", b"\xFF\xFF\xFF\xFFiTXt"];
    for tail in tails {
        let mut bytes = base[..iend].to_vec();
        bytes.extend_from_slice(tail);
        let image = png_text_decode(&bytes, PNG_TEXT_BUDGET).unwrap();
        assert_eq!(image.dimensions(), (1, 1));
        assert_eq!(image.meta.text, vec![("Good".into(), "retained".into())]);
    }
}

#[test]
fn png_invalid_optional_text_does_not_spend_an_exhausted_budget() {
    let full = png_text(b"tEXt", b"K", &vec![b'A'; PNG_TEXT_BUDGET - 1]);
    let invalid = [png_chunk(b"iTXt", b"Bad\0\0\0\0\0\xFF"), png_text_stream(b"zTXt", b"Bad", b"invalid")];
    let image = png_text_decode(&png_with_text(&[full], &invalid), PNG_TEXT_BUDGET).unwrap();
    assert_eq!(image.meta.text.len(), 1);
    assert_eq!(image.meta.text[0].1.len(), PNG_TEXT_BUDGET - 1);
}

#[test]
fn pnm_bomb_header_rejected() {
    assert!(is_limit(decode(b"P6\n200000 200000\n255\n\0\0\0")));
    assert!(is_limit(decode(b"P7\nWIDTH 99999\nHEIGHT 99999\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n")));
    assert!(is_limit(decode(b"PF\n100000 100000\n-1.0\n")));
}

#[test]
fn pnm_header_alone_reserves_nothing_without_limits() {
    // #1110: ASCII PGM/PPM reserved four bytes per declared sample before reading any. A
    // 2³¹ × 2³¹ header asked for more than isize::MAX ("capacity overflow"), and PBM reserved
    // 2⁶² bytes before finding the raster missing (an abort).
    let none = opts(Limits::none());
    let side = 1u32 << 31;
    for f in [format!("P1\n{side} {side}\n"), format!("P2\n{side} {side}\n255\n"), format!("P3\n{side} {side}\n255\n"), format!("P4\n{side} {side}\n")] {
        assert!(decode_with(f.as_bytes(), &none).is_err(), "{f:?}");
    }
}

#[test]
fn pnm_header_sizes_that_overflow_usize_are_errors_without_limits() {
    let m = u32::MAX;
    let none = opts(Limits::none());
    for f in [
        format!("P3\n{m} {m}\n255\n"),
        format!("P6\n{m} {m}\n255\n"),
        format!("P5\n{m} {m}\n65535\n"),
        format!("PF\n{m} {m}\n-1.0\n"),
        format!("P7\nWIDTH {m}\nHEIGHT {m}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"),
    ] {
        assert!(decode_with(f.as_bytes(), &none).is_err(), "{f:?}");
    }
}

#[test]
fn ascii_pnm_with_the_tightest_layout_still_decodes() {
    // The capacity bound for #1110 must not cut off files with no slack: PBM digits need no
    // separator, and the last ASCII sample needs no trailing whitespace.
    assert_eq!(decode(b"P1\n3 1\n010").unwrap().data(), &[255, 0, 255]);
    assert_eq!(decode(b"P2\n3 1\n255\n1 2 3").unwrap().data(), &[1, 2, 3]);
    assert_eq!(decode(b"P3\n1 1\n255\n9 8 7").unwrap().data(), &[9, 8, 7]);
}

#[test]
fn jpeg_bomb_header_rejected() {
    // SOI + SOF0 claiming 65535x65535x3 with no scan data.
    let b = [0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 8, 0xFF, 0xFF, 0xFF, 0xFF, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0, 0xFF, 0xD9];
    let l = Limits { max_pixels: 1 << 24, ..Limits::default() };
    assert!(is_limit(decode_with(&b, &opts(l))));
}

#[test]
fn tiff_bomb_header_rejected() {
    // Minimal little-endian TIFF claiming 1_000_000 x 1_000_000 8-bit gray.
    let mut b = b"II*\0\x08\0\0\0".to_vec();
    let entries: [(u16, u16, u32, u32); 6] = [(256, 4, 1, 1_000_000), (257, 4, 1, 1_000_000), (258, 3, 1, 8), (262, 3, 1, 1), (273, 4, 1, 0), (279, 4, 1, 0)];
    b.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in entries {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    assert!(is_limit(decode(&b)));
}

#[test]
fn limits_check_api() {
    let l = Limits::default();
    assert!(l.check(100, 100, ChannelLayout::Rgba, SampleType::F32).is_ok());
    assert!(matches!(l.check(0, 1, ChannelLayout::Gray, SampleType::U8), Err(CodecError::InvalidImage(_))));
    assert!(matches!(l.check(u32::MAX, 1, ChannelLayout::Gray, SampleType::U8), Err(CodecError::LimitExceeded(_))));
    let l = Limits { max_alloc: 399, ..Limits::default() };
    assert!(l.check(10, 10, ChannelLayout::Rgba, SampleType::U8).is_err());
    assert!(l.check(10, 10, ChannelLayout::Rgb, SampleType::U8).is_ok());
}
