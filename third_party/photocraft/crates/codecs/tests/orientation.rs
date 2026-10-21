//! EXIF / TIFF orientation (#285): files are opened upright, and every export
//! writes Orientation = 1 so the upright pixels are never rotated twice.

mod common;
use common::*;
use photocraft_codecs::*;
use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

/// A minimal EXIF (TIFF-structured) block: a filler tag, then Orientation.
fn exif(o: u32, big: bool, long: bool) -> Vec<u8> {
    let u16b = |v: u16| if big { v.to_be_bytes() } else { v.to_le_bytes() };
    let u32b = |v: u32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
    let mut v = if big { b"MM\0*".to_vec() } else { b"II*\0".to_vec() };
    v.extend_from_slice(&u32b(8));
    v.extend_from_slice(&u16b(2));
    // ImageDescription-ish filler (ASCII, 4 bytes inline).
    v.extend_from_slice(&u16b(0x010E));
    v.extend_from_slice(&u16b(2));
    v.extend_from_slice(&u32b(4));
    v.extend_from_slice(b"abc\0");
    v.extend_from_slice(&u16b(0x0112));
    if long {
        v.extend_from_slice(&u16b(4));
        v.extend_from_slice(&u32b(1));
        v.extend_from_slice(&u32b(o));
    } else {
        v.extend_from_slice(&u16b(3));
        v.extend_from_slice(&u32b(1));
        v.extend_from_slice(&u16b(o as u16));
        v.extend_from_slice(&[0, 0]);
    }
    v.extend_from_slice(&u32b(0));
    v
}

/// The "upright" test picture: `w`×`h` gray values, distinct per pixel.
fn upright(w: usize, h: usize, value: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| value(x, y)).collect()
}

/// What a camera stores for orientation `o` when the scene is `up` (`w`×`h`): the inverse
/// of the turn a viewer applies. Returns the stored pixels and their size.
fn stored(up: &[u8], w: usize, h: usize, o: u16) -> (Vec<u8>, usize, usize) {
    let (sw, sh) = if o >= 5 { (h, w) } else { (w, h) };
    let mut s = vec![0u8; up.len()];
    for y in 0..h {
        for x in 0..w {
            // Where upright pixel (x, y) sits in the stored image.
            let (sx, sy) = match o {
                1 => (x, y),
                2 => (w - 1 - x, y),         // mirrored horizontally
                3 => (w - 1 - x, h - 1 - y), // rotated 180°
                4 => (x, h - 1 - y),         // mirrored vertically
                5 => (y, x),                 // transposed
                6 => (y, w - 1 - x),         // viewer turns it 90° clockwise
                7 => (h - 1 - y, w - 1 - x), // transverse
                8 => (h - 1 - y, x),         // viewer turns it 90° counter-clockwise
                _ => unreachable!(),
            };
            s[sy * sw + sx] = up[y * w + x];
        }
    }
    (s, sw, sh)
}

/// Inserts a JPEG APP1 EXIF segment right after SOI.
fn jpeg_with_exif(jpeg: &[u8], exif: &[u8]) -> Vec<u8> {
    let mut seg = b"Exif\0\0".to_vec();
    seg.extend_from_slice(exif);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&seg);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// Inserts a PNG eXIf chunk before the first IDAT.
fn png_with_exif(png: &[u8], exif: &[u8]) -> Vec<u8> {
    let at = png.windows(4).position(|w| w == b"IDAT").unwrap() - 4;
    let mut chunk = (exif.len() as u32).to_be_bytes().to_vec();
    let mut body = b"eXIf".to_vec();
    body.extend_from_slice(exif);
    let mut crc = flate2::Crc::new();
    crc.update(&body);
    chunk.extend_from_slice(&body);
    chunk.extend_from_slice(&crc.sum().to_be_bytes());
    let mut out = png[..at].to_vec();
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[at..]);
    out
}

fn tiff_with_orientation(data: &[u8], w: u32, h: u32, o: u16) -> Vec<u8> {
    use tiff::encoder::{TiffEncoder, colortype::Gray8};
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut cursor).unwrap();
    let mut im = enc.new_image::<Gray8>(w, h).unwrap();
    im.encoder().write_tag(tiff::tags::Tag::Orientation, o).unwrap();
    im.write_data(data).unwrap();
    cursor.into_inner()
}

fn gray(w: usize, h: usize, px: Vec<u8>) -> Image {
    Image::from_u8(w as u32, h as u32, ChannelLayout::Gray, px).unwrap()
}

/// Exact pixels: distinct values per pixel.
fn exact_scene(w: usize, h: usize) -> Vec<u8> {
    upright(w, h, |x, y| (y * w + x) as u8 * 3 + 1)
}

#[test]
fn reads_orientation_in_every_encoding() {
    for o in 1..=8 {
        for big in [false, true] {
            for long in [false, true] {
                let e = exif(o, big, long);
                assert_eq!(exif_orientation(&e), o as u16, "o={o} big={big} long={long}");
                let mut prefixed = b"Exif\0\0".to_vec();
                prefixed.extend_from_slice(&e);
                assert_eq!(exif_orientation(&prefixed), o as u16);
                let fixed = upright_exif(&prefixed);
                assert_eq!(exif_orientation(&fixed), 1);
                assert_eq!(fixed.len(), prefixed.len(), "rewritten in place");
                assert_eq!(&fixed[..6], b"Exif\0\0");
                // Rewritten in the entry's own type and width: exactly `exif(1, ..)`.
                assert_eq!(&fixed[6..], &exif(1, big, long)[..], "o={o} big={big} long={long}");
            }
        }
    }
    // Already upright, no tag, or garbage: borrowed unchanged.
    assert!(matches!(upright_exif(&exif(1, false, false)), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(upright_exif(&sample_exif()), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(upright_exif(b"nonsense"), std::borrow::Cow::Borrowed(_)));
}

#[test]
fn malformed_or_out_of_range_orientation_reads_as_1() {
    for bad in [0, 9, 255, 0xFFFF, 0x1_0006] {
        assert_eq!(exif_orientation(&exif(bad, false, true)), 1, "{bad}");
    }
    let good = exif(6, false, false);
    assert_eq!(exif_orientation(&good), 6);
    for n in 0..good.len() {
        // Any cut, even one that keeps the Orientation entry but loses the next-IFD pointer.
        assert_eq!(exif_orientation(&good[..n]), 1, "truncated at {n}");
        assert!(matches!(upright_exif(&good[..n]), std::borrow::Cow::Borrowed(_)), "truncated at {n}");
    }
    // IFD offset past the end, a huge entry count, a wrong field type.
    let mut far = good.clone();
    far[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(exif_orientation(&far), 1);
    let mut many = good.clone();
    many[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
    assert_eq!(exif_orientation(&many), 1, "an entry count past the end is a truncated table");
    let mut typed = good.clone();
    typed[24..26].copy_from_slice(&2u16.to_le_bytes()); // ASCII
    assert_eq!(exif_orientation(&typed), 1);
    assert_eq!(exif_orientation(&[]), 1);
    assert_eq!(exif_orientation(b"II*\0"), 1);
}

proptest! {
    #[test]
    fn random_exif_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..256), le in any::<bool>()) {
        let mut b = if le { b"II*\0\x08\0\0\0".to_vec() } else { b"MM\0*\0\0\0\x08".to_vec() };
        b.extend_from_slice(&bytes);
        let o = exif_orientation(&b);
        prop_assert!((1..=8).contains(&o));
        prop_assert_eq!(exif_orientation(&upright_exif(&b)), 1);
        let _ = exif_orientation(&bytes);
        let _ = upright_exif(&bytes);
    }

    #[test]
    fn random_xmp_never_panics(s in ".{0,64}", tail in ".{0,16}") {
        let x = format!("{s}tiff:Orientation{tail}");
        let _ = upright_xmp(&x);
    }
}

#[test]
fn xmp_orientation_is_rewritten() {
    let attr = r#"<rdf:Description tiff:Orientation="6" tiff:Make="X"/>"#;
    assert_eq!(upright_xmp(attr), r#"<rdf:Description tiff:Orientation="1" tiff:Make="X"/>"#);
    let elem = "<tiff:Orientation>8</tiff:Orientation><tiff:Orientation>3</tiff:Orientation>";
    assert_eq!(upright_xmp(elem), "<tiff:Orientation>1</tiff:Orientation><tiff:Orientation>1</tiff:Orientation>");
    assert_eq!(upright_xmp("tiff:Orientation='12'"), "tiff:Orientation='1'");
    for same in [r#"tiff:Orientation="1""#, "no orientation", "tiff:Orientation", "tiff:Orientation=\"\"", "tiff:Orientation: 6"] {
        assert!(matches!(upright_xmp(same), std::borrow::Cow::Borrowed(_)), "{same}");
    }
}

#[test]
fn png_opens_upright_for_every_orientation() {
    let (w, h) = (5, 3);
    let up = exact_scene(w, h);
    for o in 1..=8u16 {
        let (s, sw, sh) = stored(&up, w, h, o);
        let png = encode(&gray(sw, sh, s.clone()), Format::Png, &EncodeOptions::default()).unwrap();
        let file = png_with_exif(&png, &exif(u32::from(o), o % 2 == 0, false));
        let img = decode(&file).unwrap();
        assert_eq!(img.dimensions(), (w as u32, h as u32), "o={o}");
        assert_eq!(img.data(), &up[..], "o={o}");
        assert_eq!(exif_orientation(img.meta.exif.as_deref().unwrap()), 1, "o={o}: metadata says upright");
        // Opting out keeps the stored pixels and the tag.
        let raw = decode_with(&file, &DecodeOptions { keep_orientation: true, ..Default::default() }).unwrap();
        assert_eq!((raw.dimensions(), raw.data()), ((sw as u32, sh as u32), &s[..]), "o={o}");
        assert_eq!(exif_orientation(raw.meta.exif.as_deref().unwrap()), o);
    }
}

#[test]
fn tiff_opens_upright_for_every_orientation() {
    let (w, h) = (4, 7);
    let up = exact_scene(w, h);
    for o in 1..=8u16 {
        let (s, sw, sh) = stored(&up, w, h, o);
        let img = decode(&tiff_with_orientation(&s, sw as u32, sh as u32, o)).unwrap();
        assert_eq!(img.dimensions(), (w as u32, h as u32), "o={o}");
        assert_eq!(img.data(), &up[..], "o={o}");
    }
}

#[test]
fn webp_opens_upright() {
    let (w, h) = (6, 2);
    let up = exact_scene(w, h);
    for o in [3, 6, 8] {
        let (s, sw, sh) = stored(&up, w, h, o);
        let mut out = Vec::new();
        let mut enc = image_webp::WebPEncoder::new(&mut out);
        enc.set_exif_metadata(exif(u32::from(o), false, false));
        enc.encode(&s, sw as u32, sh as u32, image_webp::ColorType::L8).unwrap();
        // WebP has no gray mode: it comes back as RGB.
        let img = decode(&out).unwrap().convert(ChannelLayout::Gray, SampleType::U8);
        assert_eq!((img.dimensions(), img.data()), ((w as u32, h as u32), &up[..]), "o={o}");
    }
}

/// 8×8-block scene so JPEG keeps each block's value: block (bx, by) is distinct.
fn block_scene(w: usize, h: usize) -> Vec<u8> {
    upright(w, h, |x, y| (20 + 25 * (x / 8 + (w / 8) * (y / 8))) as u8)
}

fn assert_blocks(img: &Image, up: &[u8], w: usize, h: usize, what: &str) {
    assert_eq!(img.dimensions(), (w as u32, h as u32), "{what}");
    let d = img.data();
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let i = (by * 8 + 4) * w + bx * 8 + 4;
            assert!((i32::from(d[i]) - i32::from(up[i])).abs() <= 4, "{what}: block ({bx},{by}) is {} not {}", d[i], up[i]);
        }
    }
}

#[test]
fn jpeg_opens_upright_for_every_orientation() {
    let (w, h) = (32, 16);
    let up = block_scene(w, h);
    for o in 1..=8u16 {
        let (s, sw, sh) = stored(&up, w, h, o);
        let jpeg = encode(&gray(sw, sh, s), Format::Jpeg, &EncodeOptions { jpeg_quality: 100, ..Default::default() }).unwrap();
        let file = jpeg_with_exif(&jpeg, &exif(u32::from(o), o > 4, o == 7));
        let img = decode(&file).unwrap();
        assert_blocks(&img, &up, w, h, &format!("o={o}"));
        assert_eq!(exif_orientation(img.meta.exif.as_deref().unwrap()), 1);
    }
}

#[test]
fn rgb_jpeg_with_dpi_swaps_resolution_on_a_quarter_turn() {
    let img = Image::from_u8(16, 8, ChannelLayout::Rgb, vec![128; 16 * 8 * 3]).unwrap().with_meta(Metadata { dpi: Some((300.0, 150.0)), ..Default::default() });
    let jpeg = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
    let back = decode(&jpeg_with_exif(&jpeg, &exif(6, false, false))).unwrap();
    assert_eq!(back.dimensions(), (8, 16));
    assert_eq!(back.meta.dpi, Some((150.0, 300.0)));
}

#[test]
fn export_writes_orientation_1_and_keeps_pixels() {
    let (w, h) = (32, 16);
    let up = block_scene(w, h);
    let (s, sw, sh) = stored(&up, w, h, 6);
    let jpeg = encode(&gray(sw, sh, s), Format::Jpeg, &EncodeOptions { jpeg_quality: 100, ..Default::default() }).unwrap();
    let opened = decode(&jpeg_with_exif(&jpeg, &exif(6, false, false))).unwrap();
    assert_blocks(&opened, &up, w, h, "opened");

    // Even an image whose metadata still claims 6 (e.g. from a PSD) is written as 1.
    let mut stale = opened.clone();
    stale.meta.exif = Some(exif(6, true, false));
    stale.meta.xmp = Some(r#"<x tiff:Orientation="6"/>"#.into());
    for format in [Format::Png, Format::Jpeg, Format::WebP, Format::Tiff] {
        let bytes = encode(&stale, format, &EncodeOptions { jpeg_quality: 100, ..Default::default() }).unwrap();
        let raw = decode_with(&bytes, &DecodeOptions { keep_orientation: true, ..Default::default() }).unwrap();
        if let Some(e) = &raw.meta.exif {
            assert_eq!(exif_orientation(e), 1, "{format:?}");
        }
        if let Some(x) = &raw.meta.xmp {
            assert!(x.contains(r#"tiff:Orientation="1""#), "{format:?}: {x}");
        }
        assert_eq!(exif_orientation(&bytes), 1, "{format:?} file");
        // Pixels as shown, both with and without applying orientation on reopen.
        let back = decode(&bytes).unwrap();
        assert_eq!(back.dimensions(), raw.dimensions(), "{format:?}");
        if format == Format::Jpeg {
            assert_blocks(&back, &up, w, h, "jpeg reopen");
        } else {
            assert_eq!(back.convert(ChannelLayout::Gray, SampleType::U8).data(), opened.data(), "{format:?} lossless");
        }
    }
}

#[test]
fn round_trip_open_save_reopen_is_stable() {
    let (w, h) = (24, 16);
    let up = block_scene(w, h);
    for o in 1..=8u16 {
        let (s, sw, sh) = stored(&up, w, h, o);
        let jpeg = encode(&gray(sw, sh, s), Format::Jpeg, &EncodeOptions { jpeg_quality: 100, ..Default::default() }).unwrap();
        let first = decode(&jpeg_with_exif(&jpeg, &exif(u32::from(o), false, false))).unwrap();
        let saved = encode(&first, Format::Jpeg, &EncodeOptions { jpeg_quality: 100, ..Default::default() }).unwrap();
        let second = decode(&saved).unwrap();
        assert_blocks(&second, &up, w, h, &format!("reopened o={o}"));
        assert_eq!(exif_orientation(second.meta.exif.as_deref().unwrap()), 1);
    }
}

#[test]
fn malformed_exif_in_a_jpeg_opens_as_stored() {
    let img = gray(16, 8, vec![90; 128]);
    let jpeg = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
    let mut rng = Rng::new(285);
    let mut cases = vec![Vec::new(), b"II*\0".to_vec(), b"MM\0*\xff\xff\xff\xff".to_vec(), exif(6, false, false)[..20].to_vec()];
    cases.extend((0..64).map(|i| {
        let mut b = if i % 2 == 0 { b"II*\0\x08\0\0\0".to_vec() } else { b"MM\0*\0\0\0\x08".to_vec() };
        b.extend(rng.bytes(i * 7));
        b
    }));
    for e in cases {
        let back = decode(&jpeg_with_exif(&jpeg, &e)).unwrap();
        let o = exif_orientation(&e);
        let want = if o >= 5 { (8, 16) } else { (16, 8) };
        assert_eq!(back.dimensions(), want, "{e:?}");
    }
}

#[test]
fn every_layout_and_depth_orients_and_inverts() {
    let inverse = |o: u16| match o {
        6 => 8,
        8 => 6,
        o => o,
    };
    for layout in ChannelLayout::ALL {
        for sample in SampleType::ALL {
            let img = test_image(layout, sample);
            for o in 0..=9u16 {
                let turned = img.clone().oriented(o).unwrap();
                if (2..=8).contains(&o) && o >= 5 {
                    assert_eq!(turned.dimensions(), (img.height(), img.width()));
                } else {
                    assert_eq!(turned.dimensions(), img.dimensions());
                }
                let back = turned.oriented(inverse(o)).unwrap();
                assert_eq!(back.data(), img.data(), "{layout:?} {sample:?} o={o}");
            }
        }
    }
}

/// #285 performance budget: turning a 24 MP photo upright on open adds ≤ 50 ms
/// (release build: `cargo test --release -p photocraft-codecs --test orientation -- --ignored`).
#[test]
#[ignore = "timing; run in release"]
fn rotating_24_mp_is_fast() {
    let (w, h) = (6000u32, 4000u32);
    let img = Image::from_u8(w, h, ChannelLayout::Rgb, (0..w as usize * h as usize * 3).map(|i| i as u8).collect()).unwrap();
    for o in [6u16, 3, 8] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let c = img.clone();
            let t = std::time::Instant::now();
            let r = c.oriented(o).unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(r);
        }
        eprintln!("orientation {o}: 24 MP RGB8 in {best:.1} ms");
        assert!(best <= 50.0, "orientation {o} took {best:.1} ms");
    }
}

/// A TIFF-structured EXIF block: IFD0 at offset 8 holding `entries`
/// (`tag, type, count, 4-byte value field`), a zero next-IFD pointer, then `tail`.
fn ifd(big: bool, entries: &[(u16, u16, u32, [u8; 4])], tail: &[u8]) -> Vec<u8> {
    let u16b = |v: u16| if big { v.to_be_bytes() } else { v.to_le_bytes() };
    let u32b = |v: u32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
    let mut v = if big { b"MM\0*".to_vec() } else { b"II*\0".to_vec() };
    v.extend_from_slice(&u32b(8));
    v.extend_from_slice(&u16b(entries.len() as u16));
    for &(tag, ty, count, value) in entries {
        v.extend_from_slice(&u16b(tag));
        v.extend_from_slice(&u16b(ty));
        v.extend_from_slice(&u32b(count));
        v.extend_from_slice(&value);
    }
    v.extend_from_slice(&u32b(0));
    v.extend_from_slice(tail);
    v
}

/// The 4-byte value field of a SHORT.
fn short(v: u16, big: bool) -> [u8; 4] {
    let b = if big { v.to_be_bytes() } else { v.to_le_bytes() };
    [b[0], b[1], 0, 0]
}

#[test]
fn duplicate_orientation_entries_first_wins() {
    for big in [false, true] {
        let o = |v| (0x0112, 3, 1, short(v, big));
        assert_eq!(exif_orientation(&ifd(big, &[o(6), o(3)], &[])), 6);
        assert_eq!(exif_orientation(&ifd(big, &[o(3), o(6)], &[])), 3);
        assert_eq!(exif_orientation(&ifd(big, &[o(1), o(8)], &[])), 1);
        // A malformed first entry is ignored as a whole: a later good one does not count.
        let bad_first = ifd(big, &[(0x0112, 3, 2, short(6, big)), o(8)], &[]);
        assert_eq!(exif_orientation(&bad_first), 1);
        let long = |v: u32| (0x0112, 4, 1, if big { v.to_be_bytes() } else { v.to_le_bytes() });
        let long_first = ifd(big, &[long(6), o(8)], &[]);
        assert_eq!(exif_orientation(&long_first), 6, "a LONG first entry counts");
        assert_eq!(&upright_exif(&long_first)[..], &ifd(big, &[long(1), o(1)], &[])[..], "each in its own width");
        let rational_first = ifd(big, &[(0x0112, 5, 1, short(6, big)), o(8)], &[]);
        assert_eq!(exif_orientation(&rational_first), 1);
        // Upright: every well-formed duplicate becomes 1, so no reader rotates twice.
        let dup = ifd(big, &[o(6), o(3)], &[]);
        let fixed = upright_exif(&dup);
        assert_eq!(fixed.len(), dup.len());
        assert_eq!(&fixed[..], &ifd(big, &[o(1), o(1)], &[])[..]);
        // Only a later duplicate disagrees: it is rewritten too.
        assert_eq!(&upright_exif(&ifd(big, &[o(1), o(8)], &[]))[..], &ifd(big, &[o(1), o(1)], &[])[..]);
        assert!(matches!(upright_exif(&ifd(big, &[o(1), o(1)], &[])), std::borrow::Cow::Borrowed(_)));
    }
}

#[test]
fn strict_ifd_parsing_ignores_the_tag() {
    for big in [false, true] {
        let o6 = (0x0112, 3, 1, short(6, big));
        let good = ifd(big, &[o6], &[]);
        assert_eq!(exif_orientation(&good), 6);
        // Truncated before (or inside) the next-IFD pointer.
        for cut in 1..=4 {
            assert_eq!(exif_orientation(&good[..good.len() - cut]), 1, "cut {cut}");
        }
        // IFD offsets inside the 8-byte header.
        for off in 0..8u32 {
            let mut b = good.clone();
            b[4..8].copy_from_slice(&if big { off.to_be_bytes() } else { off.to_le_bytes() });
            assert_eq!(exif_orientation(&b), 1, "offset {off}");
            assert!(matches!(upright_exif(&b), std::borrow::Cow::Borrowed(_)));
        }
        // Count must be exactly 1, type must be SHORT or LONG.
        let long6 = if big { 6u32.to_be_bytes() } else { 6u32.to_le_bytes() };
        for count in [0, 2, 3, u32::MAX] {
            assert_eq!(exif_orientation(&ifd(big, &[(0x0112, 3, count, short(6, big))], &[])), 1, "count {count}");
            assert_eq!(exif_orientation(&ifd(big, &[(0x0112, 4, count, long6)], &[])), 1, "LONG count {count}");
        }
        for ty in [1, 2, 5, 6, 7, 8, 9, 0, 0xFFFF] {
            let b = ifd(big, &[(0x0112, ty, 1, short(6, big))], &[]);
            assert_eq!(exif_orientation(&b), 1, "type {ty}");
            assert!(matches!(upright_exif(&b), std::borrow::Cow::Borrowed(_)), "type {ty}");
        }
    }
}

#[test]
fn upright_exif_keeps_every_other_byte() {
    for big in [false, true] {
        let u32b = |v: u32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        // IFD0: Make (12-byte ASCII stored after the IFD), Orientation 6, Software (inline ASCII),
        // ExifIFD pointer to a sub-IFD with its own entries.
        let ifd_end = 8 + 2 + 4 * 12 + 4;
        let make_at = ifd_end as u32;
        let sub_at = make_at + 12;
        let mut tail = b"PhotoCraftCo".to_vec();
        // Sub-IFD: one DateTimeOriginal-ish SHORT entry, then next-IFD 0.
        tail.extend_from_slice(&if big { 1u16.to_be_bytes() } else { 1u16.to_le_bytes() });
        tail.extend_from_slice(&if big { 0x9000u16.to_be_bytes() } else { 0x9000u16.to_le_bytes() });
        tail.extend_from_slice(&if big { 7u16.to_be_bytes() } else { 7u16.to_le_bytes() });
        tail.extend_from_slice(&u32b(4));
        tail.extend_from_slice(b"0232");
        tail.extend_from_slice(&u32b(0));
        let e = ifd(big, &[(0x010F, 2, 12, u32b(make_at)), (0x0112, 3, 1, short(6, big)), (0x0131, 2, 4, *b"PC1\0"), (0x8769, 4, 1, u32b(sub_at))], &tail);
        let mut prefixed = b"Exif\0\0".to_vec();
        prefixed.extend_from_slice(&e);
        let fixed = upright_exif(&prefixed).into_owned();
        assert_eq!(exif_orientation(&fixed), 1);
        assert_eq!(fixed.len(), prefixed.len());
        // Only the Orientation value (2 bytes in the second entry) differs.
        let value = 6 + 8 + 2 + 12 + 8;
        let changed: Vec<usize> = (0..fixed.len()).filter(|&i| fixed[i] != prefixed[i]).collect();
        assert_eq!(changed, vec![if big { value + 1 } else { value }], "big={big}");
        assert_eq!(&fixed[..value], &prefixed[..value]);
        assert_eq!(&fixed[value + 2..], &prefixed[value + 2..]);
        // The offset-based Make string and the inline Software tag are intact.
        assert_eq!(&fixed[6 + make_at as usize..][..12], b"PhotoCraftCo");
        assert!(fixed.windows(4).any(|w| w == b"PC1\0"));
    }
}

#[test]
fn limits_apply_to_the_upright_size() {
    // Stored 32×16 landscape; Orientation 6 makes it a 16×32 portrait.
    let jpeg = encode(&gray(32, 16, vec![90; 32 * 16]), Format::Jpeg, &EncodeOptions::default()).unwrap();
    let tall = jpeg_with_exif(&jpeg, &exif(6, false, false));
    let limits = Limits { max_width: 32, max_height: 16, ..Limits::default() };
    let opts = DecodeOptions { limits, ..Default::default() };
    assert!(matches!(decode_with(&tall, &opts), Err(CodecError::LimitExceeded(_))), "the upright portrait is too tall");
    // As stored it fits, and a 180° turn keeps the size.
    let kept = decode_with(&tall, &DecodeOptions { keep_orientation: true, ..opts.clone() }).unwrap();
    assert_eq!(kept.dimensions(), (32, 16));
    let flipped = decode_with(&jpeg_with_exif(&jpeg, &exif(3, false, false)), &opts).unwrap();
    assert_eq!(flipped.dimensions(), (32, 16));
    // The same holds for a TIFF, whose orientation lives in its own IFD0.
    let tiff = tiff_with_orientation(&[7; 32 * 16], 32, 16, 8);
    assert!(matches!(decode_with(&tiff, &opts), Err(CodecError::LimitExceeded(_))));
    assert_eq!(decode_with(&tiff, &DecodeOptions::default()).unwrap().dimensions(), (16, 32));
}

#[test]
fn long_orientation_reads_and_rewrites_in_both_byte_orders() {
    for big in [false, true] {
        let u32b = |v: u32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        let long = |v: u32| (0x0112u16, 4u16, 1u32, u32b(v));
        for o in 1..=8u32 {
            let e = ifd(big, &[(0x0131, 2, 4, *b"PC1\0"), long(o)], b"tail");
            assert_eq!(exif_orientation(&e), o as u16, "o={o} big={big}");
            let fixed = upright_exif(&e);
            assert_eq!(exif_orientation(&fixed), 1);
            // Still a LONG, all 4 value bytes set to 1 in the file's byte order; nothing else moved.
            assert_eq!(&fixed[..], &ifd(big, &[(0x0131, 2, 4, *b"PC1\0"), long(1)], b"tail")[..], "o={o} big={big}");
        }
        // Out of range, including values that only fit in a LONG, reads as 1 and is left alone.
        for bad in [0, 9, 0x1_0006, u32::MAX] {
            let e = ifd(big, &[long(bad)], &[]);
            assert_eq!(exif_orientation(&e), 1, "{bad}");
        }
        // A LONG 6 in a JPEG opens upright.
        let jpeg = encode(&gray(32, 16, vec![90; 32 * 16]), Format::Jpeg, &EncodeOptions::default()).unwrap();
        let img = decode(&jpeg_with_exif(&jpeg, &ifd(big, &[long(6)], &[]))).unwrap();
        assert_eq!(img.dimensions(), (16, 32));
        let kept = img.meta.exif.as_deref().unwrap();
        assert_eq!(kept.strip_prefix(b"Exif\0\0").unwrap_or(kept), &ifd(big, &[long(1)], &[])[..]);
    }
}
