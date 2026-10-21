//! Resolution recorded in metadata (#1691): a camera JPEG keeps its resolution in EXIF only, and
//! Photoshop opens it at that resolution; Photoshop's own APP13 ResolutionInfo and the XMP
//! `tiff:` properties outrank EXIF, and the JFIF density comes last. Exports write the image's
//! resolution into JFIF, EXIF and XMP alike. Every file here is built by hand.

use std::borrow::Cow;

use photocraft_codecs::resolution::{exif_with_resolution, xmp_with_resolution};
use photocraft_codecs::*;
use proptest::prelude::{any, proptest};

const X: u16 = 282;
const Y: u16 = 283;
const UNIT: u16 = 296;
const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;
const ASCII: u16 = 2;

/// One IFD0 entry: tag, type, count and the value bytes (in the block's byte order).
struct E(u16, u16, u32, Vec<u8>);

struct Exif {
    big: bool,
}

impl Exif {
    fn u16(&self, v: u16) -> [u8; 2] {
        if self.big { v.to_be_bytes() } else { v.to_le_bytes() }
    }
    fn u32(&self, v: u32) -> [u8; 4] {
        if self.big { v.to_be_bytes() } else { v.to_le_bytes() }
    }
    fn rational(&self, n: u32, d: u32) -> Vec<u8> {
        [self.u32(n), self.u32(d)].concat()
    }
    fn short(&self, v: u16) -> Vec<u8> {
        self.u16(v).to_vec()
    }
    fn x(&self, n: u32, d: u32) -> E {
        E(X, RATIONAL, 1, self.rational(n, d))
    }
    fn y(&self, n: u32, d: u32) -> E {
        E(Y, RATIONAL, 1, self.rational(n, d))
    }
    fn unit(&self, u: u16) -> E {
        E(UNIT, SHORT, 1, self.short(u))
    }
    fn make(&self) -> E {
        E(0x010F, ASCII, 6, b"NIKON\0".to_vec())
    }
    fn orientation(&self, o: u16) -> E {
        E(0x0112, SHORT, 1, self.short(o))
    }

    /// A TIFF-structured EXIF block: header, IFD0 at 8, then the out-of-line values.
    fn build(&self, entries: &[E]) -> Vec<u8> {
        let mut v = if self.big { b"MM\0*".to_vec() } else { b"II*\0".to_vec() };
        v.extend_from_slice(&self.u32(8));
        v.extend_from_slice(&self.u16(entries.len() as u16));
        let mut data_at = 8 + 2 + entries.len() * 12 + 4;
        let mut data = Vec::new();
        for E(tag, ty, count, value) in entries {
            v.extend_from_slice(&self.u16(*tag));
            v.extend_from_slice(&self.u16(*ty));
            v.extend_from_slice(&self.u32(*count));
            if value.len() <= 4 {
                let mut inline = value.clone();
                inline.resize(4, 0);
                v.extend_from_slice(&inline);
            } else {
                v.extend_from_slice(&self.u32(data_at as u32));
                data.extend_from_slice(value);
                data_at += value.len();
            }
        }
        v.extend_from_slice(&self.u32(0));
        v.extend_from_slice(&data);
        v
    }
}

const LE: Exif = Exif { big: false };
const BE: Exif = Exif { big: true };

/// A D3200-style block: Make, XResolution 300/1, YResolution 300/1, ResolutionUnit 2.
fn camera_exif(e: &Exif) -> Vec<u8> {
    e.build(&[e.make(), e.x(300, 1), e.y(300, 1), e.unit(2)])
}

fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut s = vec![0xFF, marker];
    s.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    s.extend_from_slice(payload);
    s
}

fn jfif(unit: u8, x: u16, y: u16) -> Vec<u8> {
    let mut p = b"JFIF\0\x01\x02".to_vec();
    p.push(unit);
    p.extend_from_slice(&x.to_be_bytes());
    p.extend_from_slice(&y.to_be_bytes());
    p.extend_from_slice(&[0, 0]);
    segment(0xE0, &p)
}

fn exif_app1(exif: &[u8]) -> Vec<u8> {
    segment(0xE1, &[b"Exif\0\0".as_slice(), exif].concat())
}

fn xmp_app1(xmp: &str) -> Vec<u8> {
    segment(0xE1, &[b"http://ns.adobe.com/xap/1.0/\0".as_slice(), xmp.as_bytes()].concat())
}

/// An 8BIM resource block holding ResolutionInfo at `ppi` (display unit inch), after an
/// unrelated resource with an odd-length name and an odd-sized body.
fn irb(ppi: f64) -> Vec<u8> {
    let mut b = b"8BIM".to_vec();
    b.extend_from_slice(&0x0404u16.to_be_bytes());
    b.extend_from_slice(&[3, b'a', b'b', b'c']);
    b.extend_from_slice(&3u32.to_be_bytes());
    b.extend_from_slice(&[1, 2, 3, 0]);
    b.extend_from_slice(b"8BIM");
    b.extend_from_slice(&0x03EDu16.to_be_bytes());
    b.extend_from_slice(&[0, 0]);
    b.extend_from_slice(&16u32.to_be_bytes());
    let fixed = ((ppi * 65536.0).round() as u32).to_be_bytes();
    for _ in 0..2 {
        b.extend_from_slice(&fixed);
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&2u16.to_be_bytes());
    }
    b
}

fn app13(resources: &[u8]) -> Vec<u8> {
    segment(0xED, &[b"Photoshop 3.0\0".as_slice(), resources].concat())
}

/// A 16×8 baseline JPEG whose APPn segments are exactly `segments` (the encoder's own JFIF
/// header is taken out).
fn jpeg_with(segments: &[Vec<u8>]) -> Vec<u8> {
    let img = Image::from_raw(16, 8, ChannelLayout::Gray, SampleType::U8, vec![128; 128]).unwrap();
    let base = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
    let mut i = 2;
    while (0xE0..=0xEF).contains(&base[i + 1]) {
        i += 2 + u16::from_be_bytes([base[i + 2], base[i + 3]]) as usize;
    }
    let mut out = base[..2].to_vec();
    for s in segments {
        out.extend_from_slice(s);
    }
    out.extend_from_slice(&base[i..]);
    out
}

fn dpi_of(bytes: &[u8]) -> Option<(f32, f32)> {
    decode(bytes).unwrap().meta.dpi
}

fn near(got: Option<(f32, f32)>, want: f32) {
    let (x, y) = got.unwrap_or_else(|| panic!("no resolution, wanted {want}"));
    assert!((x - want).abs() < 0.01 && (y - want).abs() < 0.01, "got {x}×{y}, wanted {want}");
}

/// The APPn payloads of a JPEG: (marker, payload).
fn app_segments(b: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= b.len() && b[i] == 0xFF && b[i + 1] != 0xDA {
        let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        if (0xE0..=0xEF).contains(&b[i + 1]) {
            out.push((b[i + 1], b[i + 4..i + 2 + len].to_vec()));
        }
        i += 2 + len;
    }
    out
}

/// (unit, x, y) of a JPEG's JFIF header.
fn jfif_density(b: &[u8]) -> (u8, u16, u16) {
    let (_, p) = app_segments(b).into_iter().find(|(m, p)| *m == 0xE0 && p.starts_with(b"JFIF\0")).unwrap();
    (p[7], u16::from_be_bytes([p[8], p[9]]), u16::from_be_bytes([p[10], p[11]]))
}

fn exif_of(b: &[u8]) -> Vec<u8> {
    let (_, p) = app_segments(b).into_iter().find(|(m, p)| *m == 0xE1 && p.starts_with(b"Exif\0\0")).unwrap();
    p[6..].to_vec()
}

#[test]
fn exif_only_camera_jpeg_opens_at_its_exif_resolution() {
    for e in [&LE, &BE] {
        near(dpi_of(&jpeg_with(&[exif_app1(&camera_exif(e))])), 300.0);
        // ResolutionUnit absent: inch, the EXIF default.
        near(dpi_of(&jpeg_with(&[exif_app1(&e.build(&[e.x(300, 1), e.y(300, 1)]))])), 300.0);
        // Non-unit denominators.
        near(dpi_of(&jpeg_with(&[exif_app1(&e.build(&[e.x(3_000_000, 10_000), e.y(1200, 4), e.unit(2)]))])), 300.0);
    }
}

#[test]
fn exif_resolution_in_centimetres_is_converted() {
    for e in [&LE, &BE] {
        let exif = e.build(&[e.x(11811, 100), e.y(11811, 100), e.unit(3)]);
        near(exif_resolution(&exif), 299.9994);
        near(dpi_of(&jpeg_with(&[exif_app1(&exif)])), 299.9994);
    }
}

#[test]
fn jfif_aspect_ratio_only_with_exif_uses_exif() {
    near(dpi_of(&jpeg_with(&[jfif(0, 1, 1), exif_app1(&camera_exif(&LE))])), 300.0);
}

#[test]
fn exif_outranks_a_default_jfif_density() {
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif_app1(&camera_exif(&LE))])), 300.0);
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif_app1(&camera_exif(&BE))])), 300.0);
    // JFIF alone still counts, in either unit.
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72)])), 72.0);
    near(dpi_of(&jpeg_with(&[jfif(2, 100, 100)])), 254.0);
    assert_eq!(dpi_of(&jpeg_with(&[jfif(0, 1, 1)])), None);
}

#[test]
fn photoshop_resolution_info_wins_then_xmp_then_exif() {
    let xmp = r#"<x:xmpmeta><rdf:Description tiff:XResolution="200/1" tiff:YResolution="200/1" tiff:ResolutionUnit="2"/></x:xmpmeta>"#;
    let exif = exif_app1(&camera_exif(&LE));
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif.clone(), xmp_app1(xmp), app13(&irb(240.0))])), 240.0);
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif.clone(), xmp_app1(xmp)])), 200.0);
    // A fractional Photoshop resolution survives the 16.16 fixed point.
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif, app13(&irb(299.5))])), 299.5);
    near(photoshop_resolution(&irb(150.0)), 150.0);
    // Element form, centimetres.
    let elem = "<tiff:XResolution>100/1</tiff:XResolution><tiff:YResolution>100</tiff:YResolution><tiff:ResolutionUnit>3</tiff:ResolutionUnit>";
    near(xmp_resolution(elem), 254.0);
}

#[test]
fn unusable_exif_falls_back_to_jfif() {
    let e = &LE;
    let bad = [
        // No absolute unit.
        e.build(&[e.x(300, 1), e.y(300, 1), e.unit(1)]),
        // Unknown unit, malformed unit entry.
        e.build(&[e.x(300, 1), e.y(300, 1), e.unit(9)]),
        e.build(&[e.x(300, 1), e.y(300, 1), E(UNIT, LONG, 1, e.u32(2).to_vec())]),
        // Zero denominator, zero value, wrong type, wrong count.
        e.build(&[e.x(300, 0), e.y(300, 1)]),
        e.build(&[e.x(0, 1), e.y(300, 1)]),
        e.build(&[E(X, SHORT, 1, e.short(300)), e.y(300, 1)]),
        e.build(&[E(X, RATIONAL, 2, [e.rational(300, 1), e.rational(300, 1)].concat()), e.y(300, 1)]),
        // Only one axis.
        e.build(&[e.x(300, 1)]),
    ];
    for exif in &bad {
        assert_eq!(exif_resolution(exif), None);
        near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif_app1(exif)])), 72.0);
    }
    // A value offset pointing past the end.
    let mut far = camera_exif(e);
    let at = 8 + 2 + 12 + 8; // XResolution entry's value field
    far[at..at + 4].copy_from_slice(&e.u32(60_000));
    assert_eq!(exif_resolution(&far), None);
    near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif_app1(&far)])), 72.0);
}

#[test]
fn truncated_metadata_never_panics_and_falls_back() {
    let exif = camera_exif(&BE);
    for n in 0..exif.len() {
        let _ = exif_resolution(&exif[..n]);
        let _ = exif_with_resolution(&exif[..n], (240.0, 240.0));
        let dpi = dpi_of(&jpeg_with(&[jfif(1, 72, 72), exif_app1(&exif[..n])]));
        assert!(dpi == Some((72.0, 72.0)) || dpi == Some((300.0, 300.0)), "{n}: {dpi:?}");
    }
    let r = irb(300.0);
    for n in 0..r.len() {
        assert_eq!(photoshop_resolution(&r[..n]), None, "{n}");
        near(dpi_of(&jpeg_with(&[jfif(1, 72, 72), app13(&r[..n])])), 72.0);
    }
    for x in ["tiff:XResolution=", "tiff:XResolution=\"", "tiff:XResolution=\"\" tiff:YResolution=\"5/0\"", "<tiff:XResolution>", "</tiff:XResolution>9<"] {
        assert_eq!(xmp_resolution(x), None, "{x}");
        let _ = xmp_with_resolution(x, (1.0, 1.0));
    }
}

proptest! {
    #[test]
    fn hostile_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..256), ppi in 0.0f32..2e6) {
        let _ = exif_resolution(&bytes);
        let _ = exif_with_resolution(&bytes, (ppi, ppi));
        let _ = photoshop_resolution(&bytes);
        let s = String::from_utf8_lossy(&bytes);
        let _ = xmp_resolution(&s);
        let _ = xmp_with_resolution(&s, (ppi, ppi));
    }

    #[test]
    fn mutated_exif_never_panics(i in 0usize..80, v in any::<u8>(), ppi in 1.0f32..10_000.0) {
        let mut exif = camera_exif(&LE);
        if let Some(b) = exif.get_mut(i) { *b = v; }
        let _ = exif_resolution(&exif);
        let fixed = exif_with_resolution(&exif, (ppi, ppi));
        let _ = exif_resolution(&fixed);
        let _ = exif_orientation(&fixed);
    }
}

#[test]
fn rewriting_keeps_every_other_byte() {
    for e in [&LE, &BE] {
        let src = e.build(&[e.make(), e.orientation(6), e.x(300, 1), e.y(300, 1), e.unit(3)]);
        let out = exif_with_resolution(&src, (240.0, 240.0)).into_owned();
        assert_eq!(out.len(), src.len());
        near(exif_resolution(&out), 240.0);
        let want = e.build(&[e.make(), e.orientation(6), e.x(240, 1), e.y(240, 1), e.unit(2)]);
        assert_eq!(out, want);
        // Fractions keep three decimals.
        near(exif_resolution(&exif_with_resolution(&src, (299.5, 72.25))).map(|(x, _)| (x, x)), 299.5);
        // Nothing to change: borrowed.
        assert!(matches!(exif_with_resolution(&camera_exif(e), (300.0, 300.0)), Cow::Borrowed(_)));
        let no_res = e.build(&[e.make()]);
        assert!(matches!(exif_with_resolution(&no_res, (240.0, 240.0)), Cow::Borrowed(_)));
        // With the JPEG prefix.
        let prefixed = [b"Exif\0\0".as_slice(), &src].concat();
        assert_eq!(exif_with_resolution(&prefixed, (240.0, 240.0)).as_ref(), [b"Exif\0\0".as_slice(), &want].concat());
    }
}

#[test]
fn entries_that_cannot_be_rewritten_are_removed() {
    for e in [&LE, &BE] {
        // XResolution as a SHORT and a LONG unit: both go, the rest stays readable.
        let src = e.build(&[e.make(), E(X, SHORT, 1, e.short(300)), e.y(300, 1), E(UNIT, LONG, 1, e.u32(2).to_vec()), e.orientation(6)]);
        let out = exif_with_resolution(&src, (240.0, 240.0)).into_owned();
        assert_eq!(out.len(), src.len());
        let count = if e.big { u16::from_be_bytes([out[8], out[9]]) } else { u16::from_le_bytes([out[8], out[9]]) };
        assert_eq!(count, 3);
        assert_eq!(exif_orientation(&out), 6);
        // Only YResolution is left (240): no X, so no resolution at all, and no stale 300.
        assert_eq!(exif_resolution(&out), None);
        let tags: Vec<u16> = (0..3)
            .map(|k| {
                let at = 10 + k * 12;
                if e.big { u16::from_be_bytes([out[at], out[at + 1]]) } else { u16::from_le_bytes([out[at], out[at + 1]]) }
            })
            .collect();
        assert_eq!(tags, vec![0x010F, Y, 0x0112]);
        // The next-IFD pointer followed the table, the freed entries are zero.
        assert_eq!(&out[10 + 36..10 + 40], &[0, 0, 0, 0]);
        assert!(out[10 + 40..10 + 64].iter().all(|&b| b == 0));
        // The Make string (out of line) did not move.
        assert!(out.windows(6).any(|w| w == b"NIKON\0"));
    }
}

#[test]
fn xmp_resolution_is_rewritten() {
    let attr = r#"<rdf:Description tiff:XResolution="300/1" tiff:YResolution='300/1' tiff:ResolutionUnit="3" tiff:Make="X"/>"#;
    assert_eq!(
        xmp_with_resolution(attr, (240.0, 240.0)),
        r#"<rdf:Description tiff:XResolution="240/1" tiff:YResolution='240/1' tiff:ResolutionUnit="2" tiff:Make="X"/>"#
    );
    let elem = "<tiff:XResolution>300/1</tiff:XResolution><tiff:YResolution>300/1</tiff:YResolution>";
    assert_eq!(xmp_with_resolution(elem, (72.5, 72.5)), "<tiff:XResolution>72500/1000</tiff:XResolution><tiff:YResolution>72500/1000</tiff:YResolution>");
    for same in [r#"tiff:XResolution="240" tiff:YResolution="240/1""#, "<x/>"] {
        assert!(matches!(xmp_with_resolution(same, (240.0, 240.0)), Cow::Borrowed(_)), "{same}");
    }
}

#[test]
fn jpeg_export_writes_one_resolution_everywhere() {
    for e in [&LE, &BE] {
        let xmp = r#"<rdf:Description tiff:XResolution="300/1" tiff:YResolution="300/1"/>"#;
        let src = jpeg_with(&[jfif(1, 72, 72), exif_app1(&camera_exif(e)), xmp_app1(xmp)]);
        let mut img = decode(&src).unwrap();
        near(img.meta.dpi, 300.0);
        img.meta.dpi = Some((240.0, 240.0));
        let out = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
        assert_eq!(jfif_density(&out), (1, 240, 240));
        near(exif_resolution(&exif_of(&out)), 240.0);
        let (_, x) = app_segments(&out).into_iter().find(|(m, p)| *m == 0xE1 && p.starts_with(b"http://ns.adobe.com/xap/1.0/\0")).unwrap();
        near(xmp_resolution(std::str::from_utf8(&x).unwrap()), 240.0);
        near(dpi_of(&out), 240.0);
        // The rest of the EXIF came along.
        assert!(exif_of(&out).windows(6).any(|w| w == b"NIKON\0"));
    }
}

#[test]
fn other_formats_take_the_exif_resolution_when_they_have_none() {
    let mut img = Image::from_raw(4, 4, ChannelLayout::Rgb, SampleType::U8, vec![90; 48]).unwrap();
    img.meta.exif = Some(camera_exif(&LE));
    for fmt in [Format::WebP, Format::Png] {
        // Written at 240 with the 300 ppi EXIF: the EXIF follows the image.
        img.meta.dpi = Some((240.0, 240.0));
        let out = encode(&img, fmt, &EncodeOptions::default()).unwrap();
        let back = decode(&out).unwrap();
        near(back.meta.dpi, 240.0);
        near(back.meta.exif.as_deref().and_then(exif_resolution), 240.0);
        // Written without a resolution: the EXIF one is read back.
        img.meta.dpi = None;
        let out = encode(&img, fmt, &EncodeOptions::default()).unwrap();
        near(decode(&out).unwrap().meta.dpi, 300.0);
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

#[test]
fn png_phys_outranks_its_exif() {
    // eXIf says 300 ppi, then a pHYs chunk of 150 ppi (5906 px/m) goes in after IHDR.
    let mut img = Image::from_raw(4, 4, ChannelLayout::Rgb, SampleType::U8, vec![90; 48]).unwrap();
    img.meta.exif = Some(camera_exif(&LE));
    let png = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    near(decode(&png).unwrap().meta.dpi, 300.0);
    near(decode(&with_phys(&png, 5906)).unwrap().meta.dpi, 150.0);
}

/// `png` with a `pHYs` chunk of `ppm` pixels per metre after IHDR.
fn with_phys(png: &[u8], ppm: u32) -> Vec<u8> {
    let mut body = b"pHYs".to_vec();
    body.extend_from_slice(&ppm.to_be_bytes());
    body.extend_from_slice(&ppm.to_be_bytes());
    body.push(1);
    let mut chunk = 9u32.to_be_bytes().to_vec();
    chunk.extend_from_slice(&body);
    chunk.extend_from_slice(&crc32(&body).to_be_bytes());
    let ihdr_end = 8 + 8 + 13 + 4;
    [&png[..ihdr_end], &chunk, &png[ihdr_end..]].concat()
}

#[test]
fn png_phys_of_a_whole_ppi_reads_back_whole() {
    // pHYs holds whole pixels per metre: 300 ppi is stored as 11811 (299.9994 ppi), 72 as 2835
    // (72.009). The whole number that rounds to the stored value is read, as Photoshop does; a
    // value no whole ppi rounds to stays as it is.
    let img = Image::from_raw(4, 4, ChannelLayout::Rgb, SampleType::U8, vec![90; 48]).unwrap();
    let png = encode(&img, Format::Png, &EncodeOptions { embed_metadata: false, ..Default::default() }).unwrap();
    for (ppm, want) in [(11811, 300.0), (2835, 72.0), (3937, 100.0), (11812, 300.0248)] {
        let (x, y) = decode(&with_phys(&png, ppm)).unwrap().meta.dpi.unwrap();
        assert!((x - want).abs() < 1e-4 && (y - want).abs() < 1e-4, "{ppm} px/m: {x} × {y}, wanted {want}");
    }
    // And a 300 ppi PNG written by the encoder opens at exactly 300.
    let mut img = img;
    img.meta.dpi = Some((300.0, 300.0));
    let png = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    assert_eq!(decode(&png).unwrap().meta.dpi, Some((300.0, 300.0)));
}

/// `camera_exif` plus IFD1, the embedded thumbnail a camera (or the last editor) wrote:
/// JPEGInterchangeFormat / JPEGInterchangeFormatLength naming a fake JPEG after the IFD.
fn exif_with_thumbnail(e: &Exif) -> Vec<u8> {
    let mut v = camera_exif(e);
    let ifd1 = v.len() as u32;
    // IFD0 (4 entries) ends with its next-IFD pointer.
    v[8 + 2 + 4 * 12..8 + 2 + 4 * 12 + 4].copy_from_slice(&e.u32(ifd1));
    let thumb = b"\xFF\xD8stale thumbnail\xFF\xD9";
    let thumb_at = ifd1 + 2 + 2 * 12 + 4;
    v.extend_from_slice(&e.u16(2));
    for (tag, value) in [(0x0201u16, thumb_at), (0x0202, thumb.len() as u32)] {
        v.extend_from_slice(&e.u16(tag));
        v.extend_from_slice(&e.u16(LONG));
        v.extend_from_slice(&e.u32(1));
        v.extend_from_slice(&e.u32(value));
    }
    v.extend_from_slice(&e.u32(0));
    v.extend_from_slice(thumb);
    v
}

fn next_ifd(e: &Exif, exif: &[u8]) -> u32 {
    let at = 8 + 2 + 4 * 12;
    let b: [u8; 4] = exif[at..at + 4].try_into().unwrap();
    if e.big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) }
}

/// #2147 / #2373: an export must not carry the source's EXIF thumbnail (IFD1). Finder, Android
/// and upload pickers show it instead of the image, so an edited photo kept its old thumbnail.
#[test]
fn exports_drop_the_source_exif_thumbnail() {
    for e in [&LE, &BE] {
        let exif = exif_with_thumbnail(e);
        assert_ne!(next_ifd(e, &exif), 0);
        for ppi in [None, Some((300.0, 300.0)), Some((240.0, 240.0))] {
            let out = export_exif(&exif, ppi);
            assert_eq!(next_ifd(e, &out), 0, "IFD1 is unlinked ({ppi:?})");
            assert_eq!(out.len(), exif.len(), "nothing moves");
            assert!(out.windows(6).any(|w| w == b"NIKON\0"), "the rest of the EXIF is kept");
            near(exif_resolution(&out), ppi.map_or(300.0, |p| p.0));
        }
        // Only the pointer changes.
        let out = export_exif(&exif, None);
        let differing: Vec<usize> = (0..exif.len()).filter(|&i| exif[i] != out[i]).collect();
        assert!(differing.iter().all(|&i| (8 + 2 + 4 * 12..8 + 2 + 4 * 12 + 4).contains(&i)), "{differing:?}");
        // With a JPEG-style prefix too.
        let prefixed = [b"Exif\0\0".as_slice(), &exif].concat();
        assert_eq!(next_ifd(e, &export_exif(&prefixed, None)[6..]), 0);
        // An EXIF without a thumbnail comes through untouched.
        assert!(matches!(export_exif(&camera_exif(e), None), Cow::Borrowed(_)));
        // Through every encoder that writes EXIF.
        let mut img = Image::from_raw(4, 4, ChannelLayout::Rgb, SampleType::U8, vec![90; 48]).unwrap();
        img.meta.exif = Some(exif.clone());
        let jpeg = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
        assert_eq!(next_ifd(e, &exif_of(&jpeg)), 0, "JPEG");
        for format in [Format::Png, Format::WebP] {
            let back = decode(&encode(&img, format, &EncodeOptions::default()).unwrap()).unwrap();
            let got = back.meta.exif.unwrap_or_else(|| panic!("{format:?} kept no EXIF"));
            let body = got.strip_prefix(b"Exif\0\0").unwrap_or(&got);
            assert_eq!(next_ifd(e, body), 0, "{format:?}");
        }
    }
}

#[test]
fn thumbnail_removal_never_panics_on_truncated_exif() {
    for e in [&LE, &BE] {
        let exif = exif_with_thumbnail(e);
        for n in 0..exif.len() {
            let _ = export_exif(&exif[..n], None);
            let _ = export_exif(&exif[..n], Some((300.0, 300.0)));
        }
    }
}
