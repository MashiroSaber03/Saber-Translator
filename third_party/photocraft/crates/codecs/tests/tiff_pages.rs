//! Large-file TIFF support (FILE-215-8): BigTIFF in both byte orders, every IFD of a multi-page
//! file and its SubIFDs, banded decoding of strips and tiles in either planar configuration, the
//! sample formats and interpretations around them, and hostile structures (cycles, counts and
//! offsets past the end, absurd sizes) that must give `Ok` or a clean `Err`, never a panic.
//! Every file here is hand-built by `common::tiffgen`.

mod common;
use common::Rng;
use common::tiffgen::*;
use photocraft_codecs::*;

const ORDERS: [(bool, bool); 4] = [(true, false), (false, false), (true, true), (false, true)];

fn ne16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn ne32f(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// A test image's samples in each storage: 8-bit, 16-bit and 32-bit float.
struct Truth {
    u8s: Vec<u8>,
    u16s: Vec<u16>,
    f32s: Vec<f32>,
}

fn truth(n: usize, seed: u64) -> Truth {
    let mut r = Rng::new(seed);
    let u16s: Vec<u16> = (0..n).map(|_| r.next_u64() as u16).collect();
    Truth { u8s: u16s.iter().map(|v| (v >> 8) as u8).collect(), f32s: u16s.iter().map(|&v| f32::from(v) / 65535.0).collect(), u16s }
}

/// `(bits, sample format, file bytes in this order, expected native bytes, sample type)`.
type Depth = (u16, u16, Vec<u8>, Vec<u8>, SampleType);

fn depths(t: &Truth, little: bool) -> Vec<Depth> {
    vec![
        (8, 1, t.u8s.clone(), t.u8s.clone(), SampleType::U8),
        (16, 1, u16s(&t.u16s, little), ne16(&t.u16s), SampleType::U16),
        (32, 3, f32s(&t.f32s, little), ne32f(&t.f32s), SampleType::F32),
    ]
}

fn decode_ok(bytes: &[u8]) -> Image {
    decode(bytes).unwrap_or_else(|e| panic!("decode failed: {e}"))
}

#[test]
fn bigtiff_and_classic_in_both_byte_orders_at_every_depth() {
    let (w, h) = (7u32, 5u32);
    for spp in [1u16, 3, 4] {
        let t = truth((w * h) as usize * spp as usize, u64::from(spp));
        for (little, big) in ORDERS {
            for (bits, format, file, want, sample) in depths(&t, little) {
                let photometric = if spp == 1 { 1 } else { 2 };
                let mut d = strips(w, h, bits, spp, photometric, 2, &file).tag(339, SHORT, &vec![u64::from(format); spp as usize]);
                if spp == 4 {
                    d = d.tag(338, SHORT, &[2]);
                }
                let b = build(little, big, &[d], &[0]);
                let img = decode_ok(&b.bytes);
                assert_eq!((img.dimensions(), img.sample_type()), ((w, h), sample), "little {little} big {big} bits {bits} spp {spp}");
                assert_eq!(img.layout().channels(), spp as usize);
                assert_eq!(img.data(), &want[..], "little {little} big {big} bits {bits} spp {spp}");
                assert_eq!(img.warnings, []);
                let info = tiff_info(&b.bytes).unwrap();
                assert_eq!((info.big_tiff, info.little_endian, info.pages.len(), info.complete), (big, little, 1, true));
            }
        }
    }
}

#[test]
fn tiles_with_padding_in_both_planar_configurations() {
    let (w, h) = (37usize, 29usize);
    for spp in [1usize, 3, 4] {
        let t = truth(w * h * spp, 9 + spp as u64);
        for (little, big) in ORDERS {
            for (bits, format, file, want, _) in depths(&t, little) {
                let bps = bits as usize / 8;
                for planar in [false, true] {
                    let mut d = Dir {
                        entries: base(w as u32, h as u32, bits, spp as u16, if spp == 1 { 1 } else { 2 }),
                        chunks: tile_split(&file, w, h, spp, bps, 16, 16, planar),
                        tiled: true,
                    };
                    d = d.tag(322, SHORT, &[16]).tag(323, SHORT, &[16]).tag(339, SHORT, &[u64::from(format)]);
                    if planar {
                        d = d.tag(284, SHORT, &[2]);
                    }
                    if spp == 4 {
                        d = d.tag(338, SHORT, &[2]);
                    }
                    let b = build(little, big, &[d], &[0]);
                    let img = decode_ok(&b.bytes);
                    assert_eq!(img.data(), &want[..], "tiles: little {little} big {big} bits {bits} spp {spp} planar {planar}");
                    let info = tiff_info(&b.bytes).unwrap();
                    assert!(info.pages[0].tiled && info.pages[0].planar == planar);
                }
            }
        }
    }
}

#[test]
fn planar_strips_at_every_depth() {
    let (w, h) = (11usize, 9usize);
    for spp in [3usize, 4] {
        let t = truth(w * h * spp, 3);
        for (little, big) in ORDERS {
            for (bits, format, file, want, _) in depths(&t, little) {
                for rows in [1usize, 4, 9] {
                    let mut d = Dir {
                        entries: base(w as u32, h as u32, bits, spp as u16, 2),
                        chunks: planar_strips(&file, w, h, spp, bits as usize / 8, rows),
                        tiled: false,
                    };
                    d = d.tag(278, LONG, &[rows as u64]).tag(284, SHORT, &[2]).tag(339, SHORT, &[u64::from(format)]);
                    if spp == 4 {
                        d = d.tag(338, SHORT, &[2]);
                    }
                    let img = decode_ok(&build(little, big, &[d], &[0]).bytes);
                    assert_eq!(img.data(), &want[..], "planar strips: little {little} big {big} bits {bits} spp {spp} rows {rows}");
                }
            }
        }
    }
}

#[test]
fn extra_samples_that_are_not_alpha_are_dropped() {
    let (w, h) = (5usize, 3usize);
    let t = truth(w * h * 5, 4);
    let rgb_of = |spp: usize, keep: usize| -> Vec<u8> { t.u8s[..w * h * spp].chunks(spp).flat_map(|p| p[..keep].to_vec()).collect() };
    for planar in [false, true] {
        // RGB + one unspecified extra sample: RGB.
        let px = &t.u8s[..w * h * 4];
        let chunks = if planar { planar_strips(px, w, h, 4, 1, 2) } else { px.chunks(w * 4 * 2).map(<[u8]>::to_vec).collect() };
        let mut d = Dir { entries: base(w as u32, h as u32, 8, 4, 2), chunks, tiled: false }.tag(278, LONG, &[2]).tag(338, SHORT, &[0]);
        if planar {
            d = d.tag(284, SHORT, &[2]);
        }
        let img = decode_ok(&build(true, false, &[d], &[0]).bytes);
        assert_eq!(img.layout(), ChannelLayout::Rgb);
        assert_eq!(img.data(), &rgb_of(4, 3)[..], "planar {planar}");
        // RGB + alpha + an unspecified extra sample: RGBA.
        let px = &t.u8s[..w * h * 5];
        let chunks = if planar { planar_strips(px, w, h, 5, 1, 2) } else { px.chunks(w * 5 * 2).map(<[u8]>::to_vec).collect() };
        let mut d = Dir { entries: base(w as u32, h as u32, 8, 5, 2), chunks, tiled: false }.tag(278, LONG, &[2]).tag(338, SHORT, &[2, 0]);
        if planar {
            d = d.tag(284, SHORT, &[2]);
        }
        let img = decode_ok(&build(false, true, &[d], &[0]).bytes);
        assert_eq!(img.layout(), ChannelLayout::Rgba);
        assert_eq!(img.data(), &rgb_of(5, 4)[..], "planar {planar}");
    }
    // Gray + alpha + extra: gray and alpha; gray + an unspecified extra: gray.
    let px = &t.u8s[..w * h * 3];
    let d = strips(w as u32, h as u32, 8, 3, 1, 3, px).tag(338, SHORT, &[2, 0]);
    let img = decode_ok(&build(true, false, &[d], &[0]).bytes);
    assert_eq!(img.layout(), ChannelLayout::GrayA);
    assert_eq!(img.data(), &rgb_of(3, 2)[..]);
    let px = &t.u8s[..w * h * 2];
    let d = strips(w as u32, h as u32, 8, 2, 1, 3, px).tag(338, SHORT, &[0]);
    let img = decode_ok(&build(true, false, &[d], &[0]).bytes);
    assert_eq!(img.layout(), ChannelLayout::Gray);
    assert_eq!(img.data(), &rgb_of(2, 1)[..]);
}

#[test]
fn alpha_after_an_unspecified_extra_sample_is_kept() {
    // #1349: TIFF 6.0 lets alpha be any extra sample; ExtraSamples [0, 2] puts it second.
    let (w, h) = (5usize, 3usize);
    for (photometric, color, layout) in [(2u16, 3usize, ChannelLayout::Rgba), (5, 4, ChannelLayout::CmykA), (1, 1, ChannelLayout::GrayA)] {
        let spp = color + 2;
        let t = truth(w * h * spp, 13 + spp as u64);
        for (little, big) in ORDERS {
            for (bits, format, file, want_all, sample) in depths(&t, little) {
                let bps = bits as usize / 8;
                // The colour samples, then the second extra sample (the alpha).
                let want: Vec<u8> = want_all.chunks(spp * bps).flat_map(|p| [&p[..color * bps], &p[(color + 1) * bps..]].concat()).collect();
                for planar in [false, true] {
                    let chunks = if planar { planar_strips(&file, w, h, spp, bps, 2) } else { file.chunks(w * spp * bps * 2).map(<[u8]>::to_vec).collect() };
                    let mut d = Dir { entries: base(w as u32, h as u32, bits, spp as u16, photometric), chunks, tiled: false }
                        .tag(278, LONG, &[2])
                        .tag(339, SHORT, &vec![u64::from(format); spp])
                        .tag(338, SHORT, &[0, 2]);
                    if planar {
                        d = d.tag(284, SHORT, &[2]);
                    }
                    let img = decode_ok(&build(little, big, &[d], &[0]).bytes);
                    let what = format!("photometric {photometric} little {little} big {big} bits {bits} planar {planar}");
                    assert_eq!((img.layout(), img.sample_type()), (layout, sample), "{what}");
                    assert_eq!(img.data(), &want[..], "{what}");
                    assert_eq!(img.warnings, [], "{what}");
                }
            }
        }
    }
    // The issue's file: 2x2 RGB, every pixel 200 100 50, then 17 (unspecified), then alpha 128.
    let px: Vec<u8> = [200u8, 100, 50, 17, 128].repeat(4);
    let d = strips(2, 2, 8, 5, 2, 2, &px).tag(338, SHORT, &[0, 2]);
    let img = decode_ok(&build(true, false, &[d], &[0]).bytes);
    assert_eq!(img.data(), &[200u8, 100, 50, 128].repeat(4)[..]);
    // Associated alpha in the same position is made straight.
    let px: Vec<u8> = [100u8, 50, 25, 17, 128].repeat(4);
    let d = strips(2, 2, 8, 5, 2, 2, &px).tag(338, SHORT, &[0, 1]);
    let img = decode_ok(&build(true, false, &[d], &[0]).bytes);
    assert_eq!(img.data(), &[199u8, 100, 50, 128].repeat(4)[..]);
}

#[test]
fn associated_alpha_is_made_straight() {
    // Premultiplied: half-transparent mid gray is stored as 64 with alpha 128.
    let px = [64u8, 128, 200, 255, 0, 0];
    let d = strips(3, 1, 8, 2, 1, 1, &px).tag(338, SHORT, &[1]);
    let img = decode_ok(&build(true, false, &[d], &[0]).bytes);
    assert_eq!(img.layout(), ChannelLayout::GrayA);
    assert_eq!(img.data(), &[128, 128, 200, 255, 0, 0]);
    let px16 = u16s(&[16384, 32768], false);
    let d = strips(1, 1, 16, 2, 1, 1, &px16).tag(338, SHORT, &[1]);
    let img = decode_ok(&build(false, false, &[d], &[0]).bytes);
    assert_eq!(img.to_u16_samples().unwrap(), [32768, 32768]);
}

#[test]
fn white_is_zero_opens_with_the_right_tones() {
    // #691: 00 FF 00 in a WhiteIsZero file is white, black, white.
    let b = build(true, false, &[strips(3, 1, 8, 1, 0, 1, &[0x00, 0xff, 0x00])], &[0]);
    let ours = decode_ok(&b.bytes);
    let oracle = image::load_from_memory_with_format(&b.bytes, image::ImageFormat::Tiff).unwrap().into_luma8();
    assert_eq!(oracle.as_raw(), &[255u8, 0, 255]);
    assert_eq!(ours.data(), &oracle.as_raw()[..]);
    // The BlackIsZero control keeps the stored bytes.
    let b = build(true, false, &[strips(3, 1, 8, 1, 1, 1, &[0x00, 0xff, 0x00])], &[0]);
    assert_eq!(decode_ok(&b.bytes).data(), &[0, 255, 0]);
    // 16-bit, in both byte orders and BigTIFF.
    for (little, big) in ORDERS {
        let b = build(little, big, &[strips(2, 1, 16, 1, 0, 1, &u16s(&[0, 65535], little))], &[0]);
        assert_eq!(decode_ok(&b.bytes).to_u16_samples().unwrap(), [65535, 0]);
    }
    // 1-bit (a fax-style scan): bit set = black.
    let b = build(true, false, &[strips(8, 1, 1, 1, 0, 1, &[0b1010_0000])], &[0]);
    assert_eq!(decode_ok(&b.bytes).data(), &[0, 255, 0, 255, 255, 255, 255, 255]);
    // With alpha: the gray inverts, the alpha does not.
    let b = build(true, false, &[strips(2, 1, 8, 2, 0, 1, &[0, 255, 255, 10]).tag(338, SHORT, &[2])], &[0]);
    let img = decode_ok(&b.bytes);
    assert_eq!(img.layout(), ChannelLayout::GrayA);
    assert_eq!(img.data(), &[255, 255, 0, 10]);
    // 32-bit float.
    let b = build(true, false, &[strips(2, 1, 32, 1, 0, 1, &f32s(&[0.0, 0.25], true)).tag(339, SHORT, &[3])], &[0]);
    assert_eq!(decode_ok(&b.bytes).to_f32_samples().unwrap(), [1.0, 0.75]);
}

#[test]
fn bigtiff_orientation_is_applied_like_classic_tiff() {
    // #693: the same 2×1 gray image with Orientation 6, classic and BigTIFF, both byte orders.
    for (little, big) in ORDERS {
        let d = strips(2, 1, 8, 1, 1, 1, &[10, 200]).tag(274, SHORT, &[6]);
        let b = build(little, big, &[d], &[0]);
        assert_eq!(exif_orientation(&b.bytes), 6, "little {little} big {big}");
        assert_eq!(tiff_orientation(&b.bytes), 6);
        let img = decode_ok(&b.bytes);
        assert_eq!(img.dimensions(), (1, 2), "little {little} big {big}");
        assert_eq!(img.data(), &[10, 200]);
        let kept = decode_with(&b.bytes, &DecodeOptions { keep_orientation: true, ..Default::default() }).unwrap();
        assert_eq!(kept.dimensions(), (2, 1));
    }
}

#[test]
fn palette_images_open_as_rgb() {
    // Two 8-bit entries used: 0 = red, 1 = (16, 32, 48); the map has 256 entries per channel.
    let mut map = vec![0u64; 768];
    map[0] = 0xffff;
    map[1] = 16 * 257;
    map[256 + 1] = 32 * 257;
    map[512 + 1] = 48 * 257;
    for (little, big) in ORDERS {
        let d = strips(3, 1, 8, 1, 3, 1, &[0, 1, 0]).tag(320, SHORT, &map);
        let img = decode_ok(&build(little, big, &[d], &[0]).bytes);
        assert_eq!((img.layout(), img.sample_type()), (ChannelLayout::Rgb, SampleType::U8));
        assert_eq!(img.data(), &[255, 0, 0, 16, 32, 48, 255, 0, 0]);
    }
    // 4-bit indices.
    let mut map = vec![0u64; 48];
    map[15] = 0xffff;
    map[16 + 15] = 0x8000;
    let d = strips(2, 1, 4, 1, 3, 1, &[0xf0]).tag(320, SHORT, &map);
    assert_eq!(decode_ok(&build(true, false, &[d], &[0]).bytes).data(), &[255, 128, 0, 0, 0, 0]);
    // A missing or short colour map is an error.
    let d = strips(2, 1, 8, 1, 3, 1, &[0, 1]);
    assert!(decode(&build(true, false, &[d], &[0]).bytes).is_err());
    let d = strips(2, 1, 8, 1, 3, 1, &[0, 1]).tag(320, SHORT, &[1, 2, 3]);
    assert!(decode(&build(true, false, &[d], &[0]).bytes).is_err());
}

#[test]
fn wide_and_odd_sample_formats() {
    for (little, big) in ORDERS {
        let e = |v: u32| if little { v.to_le_bytes() } else { v.to_be_bytes() };
        // 32-bit unsigned keeps its top 16 bits.
        let px: Vec<u8> = [0x1234_5678u32, 0xffff_0000].iter().flat_map(|&v| e(v)).collect();
        let img = decode_ok(&build(little, big, &[strips(2, 1, 32, 1, 1, 1, &px)], &[0]).bytes);
        assert_eq!(img.to_u16_samples().unwrap(), [0x1234, 0xffff]);
        // 64-bit float becomes 32-bit float.
        let px: Vec<u8> = [0.5f64, 2.0].iter().flat_map(|v| if little { v.to_le_bytes() } else { v.to_be_bytes() }).collect();
        let img = decode_ok(&build(little, big, &[strips(2, 1, 64, 1, 1, 1, &px).tag(339, SHORT, &[3])], &[0]).bytes);
        assert_eq!(img.to_f32_samples().unwrap(), [0.5, 2.0]);
        // 16-bit float stays half.
        let px: Vec<u8> = [f16::from_f32(0.25), f16::ONE].iter().flat_map(|v| if little { v.to_le_bytes() } else { v.to_be_bytes() }).collect();
        let img = decode_ok(&build(little, big, &[strips(2, 1, 16, 1, 1, 1, &px).tag(339, SHORT, &[3])], &[0]).bytes);
        assert_eq!(img.sample_type(), SampleType::F16);
        assert_eq!(img.to_normalized(), [0.25, 1.0]);
    }
    // 2- and 4-bit gray, rows padded to a byte, in several strips.
    let img = decode_ok(&build(true, false, &[strips(3, 2, 2, 1, 1, 1, &[0b0001_1011, 0b1110_0100])], &[0]).bytes);
    assert_eq!(img.data(), &[0, 85, 170, 255, 170, 85]);
    let img = decode_ok(&build(false, true, &[strips(3, 2, 4, 1, 1, 1, &[0x0f, 0x80, 0x12, 0x30])], &[0]).bytes);
    assert_eq!(img.data(), &[0, 255, 136, 17, 34, 51]);
    // Signed integers are unsupported, not a crash.
    let d = strips(2, 1, 16, 1, 1, 1, &[0, 0, 0, 0]).tag(339, SHORT, &[2]);
    assert!(matches!(decode(&build(true, false, &[d], &[0]).bytes), Err(CodecError::Unsupported { .. })));
}

/// A flat gray page of `w`×`h` at `level`, with an optional NewSubfileType.
fn gray_page(w: u32, h: u32, level: u8, kind: Option<u64>) -> Dir {
    let d = strips(w, h, 8, 1, 1, h, &vec![level; (w * h) as usize]);
    match kind {
        Some(k) => d.tag(254, LONG, &[k]),
        None => d,
    }
}

#[test]
fn every_ifd_and_subifd_is_listed_and_decodable() {
    for (little, big) in ORDERS {
        let sub_ty = if big { IFD8 } else { IFD };
        let dirs = vec![
            gray_page(4, 3, 1, Some(1)),                        // 0: thumbnail first
            gray_page(9, 7, 2, None).sub_ifds(sub_ty, &[4, 5]), // 1: page A, two SubIFDs
            gray_page(9, 7, 3, Some(4)),                        // 2: its transparency mask
            gray_page(5, 5, 4, Some(2)),                        // 3: page B (multi-page bit)
            gray_page(3, 2, 5, Some(1)),                        // 4: SubIFD: reduced copy of A
            gray_page(2, 1, 6, Some(1)),                        // 5: SubIFD: smaller copy
        ];
        let b = build(little, big, &dirs, &[0, 1, 2, 3]);
        let info = tiff_info(&b.bytes).unwrap();
        assert!(info.complete);
        let summary: Vec<(u32, Option<usize>, TiffPageKind)> = info.pages.iter().map(|p| (p.width, p.parent, p.kind)).collect();
        use TiffPageKind::*;
        assert_eq!(
            summary,
            [(4, None, ReducedResolution), (9, None, Page), (3, Some(1), ReducedResolution), (2, Some(1), ReducedResolution), (9, None, Mask), (5, None, Page)]
        );
        assert_eq!(info.pages[1].ifd_offset, b.ifd_at[1]);
        assert_eq!((info.default_page(), info.page_count()), (Some(1), 2));
        // Like Photoshop: the first full-resolution page opens; the thumbnail is skipped.
        let img = decode_ok(&b.bytes);
        assert_eq!((img.dimensions(), img.data()[0]), ((9, 7), 2));
        assert_eq!(img.warnings, [DecodeWarning::MorePages { total: Some(2) }]);
        // Every directory decodes on request.
        for (i, (w, level)) in [(4, 1u8), (9, 2), (3, 5), (2, 6), (9, 3), (5, 4)].into_iter().enumerate() {
            let img = decode_tiff_page(&b.bytes, i, &DecodeOptions::default()).unwrap();
            assert_eq!((img.width(), img.data()[0]), (w, level), "page {i} little {little} big {big}");
        }
        assert!(matches!(decode_tiff_page(&b.bytes, 6, &DecodeOptions::default()), Err(CodecError::InvalidImage(_))));
    }
}

#[test]
fn a_file_of_only_thumbnails_still_opens() {
    let b = build(true, false, &[gray_page(4, 3, 7, Some(1))], &[0]);
    assert_eq!(tiff_info(&b.bytes).unwrap().default_page(), Some(0));
    assert_eq!(decode_ok(&b.bytes).data()[0], 7);
}

#[test]
fn cyclic_chains_and_subifds_terminate() {
    for (little, big) in ORDERS {
        // IFD1 points back to IFD0.
        let mut b = build(little, big, &[gray_page(2, 2, 1, None), gray_page(2, 2, 2, None)], &[0, 1]);
        let first = b.ifd_at[0];
        set_next(&mut b, little, big, 1, first);
        let info = tiff_info(&b.bytes).unwrap();
        assert_eq!((info.pages.len(), info.complete), (2, false));
        let img = decode_ok(&b.bytes);
        assert_eq!(img.warnings, [DecodeWarning::MorePages { total: None }]);
        // A directory pointing at itself.
        let mut b = build(little, big, &[gray_page(2, 2, 1, None)], &[0]);
        let first = b.ifd_at[0];
        set_next(&mut b, little, big, 0, first);
        assert_eq!(tiff_info(&b.bytes).unwrap().pages.len(), 1);
        assert_eq!(decode_ok(&b.bytes).warnings, []);
        // A SubIFD listing its own parent, and one listing itself.
        let sub_ty = if big { IFD8 } else { IFD };
        let b = build(little, big, &[gray_page(2, 2, 1, None).sub_ifds(sub_ty, &[1]), gray_page(1, 1, 2, Some(1)).sub_ifds(sub_ty, &[0, 1])], &[0]);
        let info = tiff_info(&b.bytes).unwrap();
        assert_eq!((info.pages.len(), info.complete), (2, false));
        decode_ok(&b.bytes);
        // A next pointer past the end, or into the header.
        for to in [b.bytes.len() as u64 + 100, 3, u64::MAX] {
            let mut b = build(little, big, &[gray_page(2, 2, 1, None), gray_page(2, 2, 2, None)], &[0, 1]);
            set_next(&mut b, little, big, 1, to);
            let info = tiff_info(&b.bytes).unwrap();
            assert_eq!((info.pages.len(), info.complete), (2, false));
            decode_ok(&b.bytes);
        }
    }
}

#[test]
fn the_chain_walk_is_capped() {
    // 10 001 tiny directories: the walk stops at 10 000 and says the list is incomplete.
    let dirs: Vec<Dir> = (0..10_001).map(|_| gray_page(1, 1, 9, None)).collect();
    let chain: Vec<usize> = (0..dirs.len()).collect();
    let b = build(true, false, &dirs, &chain);
    let info = tiff_info(&b.bytes).unwrap();
    assert_eq!((info.pages.len(), info.complete), (10_000, false));
    assert_eq!(decode_ok(&b.bytes).warnings, [DecodeWarning::MorePages { total: None }]);
}

#[test]
fn image_data_cut_off_decodes_with_a_warning() {
    let (w, h) = (4u32, 8u32);
    let px: Vec<u8> = (0..32).collect();
    for (little, big) in ORDERS {
        // The last strip starts past the end of the file: its rows stay empty.
        let mut b = build(little, big, &[strips(w, h, 8, 1, 1, 2, &px)], &[0]);
        let len = b.bytes.len() as u64;
        patch_value(&mut b, little, big, 0, 273, 3, len + 1000);
        let img = decode_ok(&b.bytes);
        assert_eq!(img.warnings, [DecodeWarning::Truncated { format: Format::Tiff }]);
        assert_eq!(&img.data()[..24], &px[..24]);
        assert_eq!(&img.data()[24..], &[0; 8]);
        // A strip running into the end of the file.
        let mut b = build(little, big, &[strips(w, h, 8, 1, 1, 2, &px)], &[0]);
        let len = b.bytes.len() as u64;
        patch_value(&mut b, little, big, 0, 273, 1, len - 3);
        assert_eq!(decode_ok(&b.bytes).warnings, [DecodeWarning::Truncated { format: Format::Tiff }]);
        // Every strip missing: nothing to show, an error.
        let mut b = build(little, big, &[strips(w, h, 8, 1, 1, 2, &px)], &[0]);
        let len = b.bytes.len() as u64;
        for i in 0..4 {
            patch_value(&mut b, little, big, 0, 273, i, len + 7);
        }
        assert!(decode(&b.bytes).is_err());
    }
}

#[test]
fn hostile_sizes_and_counts_are_refused_without_allocating() {
    let tight = DecodeOptions { limits: Limits { max_alloc: 64 << 20, ..Limits::default() }, ..Default::default() };
    for (little, big) in ORDERS {
        let px = vec![7u8; 16];
        let fresh = || build(little, big, &[strips(4, 4, 8, 1, 1, 1, &px)], &[0]);
        // Huge declared dimensions: over the limits, or far more than the file can hold.
        for (wv, hv) in [(1u64 << 31, 4u64), (4, 1 << 31), (u32::MAX as u64, u32::MAX as u64), (200_000, 200_000), (30_000, 30_000)] {
            let mut b = fresh();
            patch_value(&mut b, little, big, 0, 256, 0, wv);
            patch_value(&mut b, little, big, 0, 257, 0, hv);
            assert!(decode(&b.bytes).is_err(), "{wv}x{hv}");
            assert!(decode_with(&b.bytes, &DecodeOptions { limits: Limits::none(), ..Default::default() }).is_err(), "{wv}x{hv} without limits");
        }
        // A strip-offset count far past the end of the file.
        let mut b = fresh();
        patch_count(&mut b, little, big, 0, 273, 1 << 30);
        assert!(matches!(decode(&b.bytes), Err(CodecError::Malformed { .. })));
        let mut b = fresh();
        patch_count(&mut b, little, big, 0, 258, u32::MAX as u64);
        assert!(decode(&b.bytes).is_err());
        // Zero rows per strip, zero samples, absurd samples per pixel.
        for (tag, v) in [(278u16, 0u64), (277, 0), (277, 60_000)] {
            let mut b = fresh();
            patch_value(&mut b, little, big, 0, tag, 0, v);
            assert!(decode(&b.bytes).is_err(), "tag {tag} = {v}");
        }
        // Absurd tiles over a tiny image, chunky and planar.
        for planar in [false, true] {
            for (tw, th) in [(1u64 << 30, 16u64), (16, 1 << 30), (u32::MAX as u64, u32::MAX as u64)] {
                let spp = if planar { 3 } else { 1 };
                let mut d = Dir { entries: base(4, 4, 8, spp, if planar { 2 } else { 1 }), chunks: vec![vec![1u8; 64]; spp as usize], tiled: true };
                d = d.tag(322, LONG, &[tw]).tag(323, LONG, &[th]);
                if planar {
                    d = d.tag(284, SHORT, &[2]);
                }
                let b = build(little, big, &[d], &[0]);
                let _ = decode_with(&b.bytes, &tight);
                let _ = decode(&b.bytes);
            }
        }
    }
    // A BigTIFF directory claiming 2^60 entries.
    let mut b = build(true, true, &[gray_page(2, 2, 1, None)], &[0]);
    let at = b.ifd_at[0] as usize;
    b.bytes[at..at + 8].copy_from_slice(&(1u64 << 60).to_le_bytes());
    assert!(tiff_info(&b.bytes).is_err());
    assert!(decode(&b.bytes).is_err());
}

#[test]
fn banded_decode_matches_the_encoder_on_larger_images() {
    // Many strips decoded in parallel bands, at every depth and compression.
    for (layout, sample) in [
        (ChannelLayout::Rgb, SampleType::U8),
        (ChannelLayout::Rgba, SampleType::U16),
        (ChannelLayout::Gray, SampleType::F32),
        (ChannelLayout::Cmyk, SampleType::U8),
    ] {
        let img = common::synth(613, 411, layout, sample, 3, 0.2);
        for c in [TiffCompression::None, TiffCompression::Lzw, TiffCompression::Deflate, TiffCompression::PackBits] {
            let bytes = encode(&img, Format::Tiff, &EncodeOptions { tiff_compression: c, ..Default::default() }).unwrap();
            let back = decode_ok(&bytes);
            assert_eq!(back.data(), img.data(), "{layout:?} {sample:?} {c:?}");
        }
    }
    // Tiles across many bands.
    let t = truth(300 * 260 * 3, 77);
    let d = Dir { entries: base(300, 260, 16, 3, 2), chunks: tile_split(&u16s(&t.u16s, false), 300, 260, 3, 2, 64, 32, false), tiled: true }
        .tag(322, SHORT, &[64])
        .tag(323, SHORT, &[32]);
    assert_eq!(decode_ok(&build(false, true, &[d], &[0]).bytes).data(), &ne16(&t.u16s)[..]);
}

#[test]
fn bigtiff_is_written_on_request_and_reads_back() {
    let mut img = common::synth(23, 17, ChannelLayout::Rgba, SampleType::U16, 2, 0.3);
    img.meta.photoshop_layers = Some(b"Adobe Photoshop Document Data Block\0MIB8ryaL".to_vec());
    img.meta.photoshop_resources = Some(b"8BIM\x04\x04\0\0\0\0\0\0".to_vec());
    for c in [TiffCompression::None, TiffCompression::Deflate, TiffCompression::Lzw] {
        let classic = encode(&img, Format::Tiff, &EncodeOptions { tiff_compression: c, ..Default::default() }).unwrap();
        let big = encode(&img, Format::Tiff, &EncodeOptions { tiff_compression: c, tiff_bigtiff: true, ..Default::default() }).unwrap();
        let version = |b: &[u8]| if b.starts_with(b"II") { u16::from_le_bytes([b[2], b[3]]) } else { u16::from_be_bytes([b[2], b[3]]) };
        assert_eq!((version(&classic), version(&big)), (42, 43));
        assert!(tiff_info(&big).unwrap().big_tiff);
        let back = decode_ok(&big);
        assert_eq!(back.data(), img.data(), "{c:?}");
        assert_eq!(back.meta.photoshop_layers, img.meta.photoshop_layers);
        assert_eq!(back.meta.photoshop_resources, img.meta.photoshop_resources);
        assert_eq!(tiff_photoshop_tags(&big), (img.meta.photoshop_resources.as_deref(), img.meta.photoshop_layers.as_deref()));
    }
}

/// Small files covering every structure: multi-IFD, SubIFDs, BigTIFF both orders, tiles, planar.
fn zoo() -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    let t = truth(6 * 5 * 4, 5);
    for (little, big) in ORDERS {
        let sub_ty = if big { IFD8 } else { IFD };
        let tiles = Dir { entries: base(6, 5, 8, 3, 2), chunks: tile_split(&t.u8s, 6, 5, 3, 1, 16, 16, true), tiled: true }
            .tag(322, SHORT, &[16])
            .tag(323, SHORT, &[16])
            .tag(284, SHORT, &[2]);
        let planar = Dir { entries: base(6, 5, 16, 4, 2), chunks: planar_strips(&u16s(&t.u16s, little), 6, 5, 4, 2, 2), tiled: false }
            .tag(278, SHORT, &[2])
            .tag(284, SHORT, &[2])
            .tag(338, SHORT, &[1]);
        let dirs = vec![gray_page(3, 2, 1, Some(1)), tiles.sub_ifds(sub_ty, &[3]), planar, gray_page(2, 2, 4, Some(1)).tag(274, SHORT, &[6])];
        v.push(build(little, big, &dirs, &[0, 1, 2]).bytes);
        v.push(build(little, big, &[strips(5, 3, 1, 1, 0, 2, &[0xa5; 3])], &[0]).bytes);
    }
    v
}

fn poke(bytes: &[u8]) {
    let _ = decode(bytes);
    let _ = decode_with(bytes, &DecodeOptions { limits: Limits { max_alloc: 16 << 20, ..Limits::default() }, ..Default::default() });
    let _ = exif_orientation(bytes);
    let _ = tiff_orientation(bytes);
    if let Ok(info) = tiff_info(bytes) {
        for i in 0..info.pages.len().min(8) {
            let _ = decode_tiff_page(bytes, i, &DecodeOptions::default());
        }
    }
}

#[test]
fn zoo_decodes_and_never_panics_when_cut_or_corrupted() {
    for f in zoo() {
        let info = tiff_info(&f).unwrap();
        for i in 0..info.pages.len() {
            decode_tiff_page(&f, i, &DecodeOptions::default()).unwrap();
        }
        for cut in 0..f.len() {
            poke(&f[..cut]);
        }
        for i in 0..f.len() {
            for flip in [0xffu8, 0x80, 0x01] {
                let mut g = f.clone();
                g[i] ^= flip;
                poke(&g);
            }
        }
        let mut r = Rng::new(f.len() as u64);
        for _ in 0..300 {
            let mut g = f.clone();
            for _ in 0..4 {
                let i = (r.next_u64() as usize) % g.len();
                g[i] = r.next_u64() as u8;
            }
            poke(&g);
        }
    }
}

/// A longer random-mutation run over the zoo (ignored by default; `-- --ignored`, with
/// `TIFF_FUZZ_ITERS` to change the count).
#[test]
#[ignore = "long fuzz run"]
fn zoo_long_random_mutations() {
    let iters: usize = std::env::var("TIFF_FUZZ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let mut r = Rng::new(0x5eed);
    let zoo = zoo();
    for k in 0..iters {
        let mut g = zoo[k % zoo.len()].clone();
        for _ in 0..1 + r.next_u64() % 8 {
            let i = (r.next_u64() as usize) % g.len();
            g[i] = match r.next_u64() % 4 {
                0 => 0,
                1 => 0xff,
                _ => r.next_u64() as u8,
            };
        }
        poke(&g);
    }
}
