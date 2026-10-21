//! Decode warnings: a file that holds more than was decoded (#523: animations, multi-page
//! TIFF) or less than it should (#518: truncated JPEG) says so; complete single images don't.

mod common;
use common::*;
use photocraft_codecs::*;

const W: u32 = 24;
const H: u32 = 16;

fn solid(rgb: [u8; 3]) -> Image {
    Image::from_u8(W, H, ChannelLayout::Rgb, rgb.repeat((W * H) as usize)).unwrap()
}

const COLOURS: [[u8; 3]; 3] = [[200, 30, 10], [10, 200, 30], [30, 10, 200]];

// ---------------------------------------------------------------------------
// #518: truncated JPEG
// ---------------------------------------------------------------------------

fn jpeg(layout: ChannelLayout, progressive: bool) -> Vec<u8> {
    let img = synth(64, 48, layout, SampleType::U8, 3, 0.3);
    let ct = match layout {
        ChannelLayout::Gray => jpeg_encoder::ColorType::Luma,
        ChannelLayout::Cmyk => jpeg_encoder::ColorType::Cmyk,
        _ => jpeg_encoder::ColorType::Rgb,
    };
    let mut out = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut out, 90);
    enc.set_progressive(progressive);
    enc.encode(img.data(), 64, 48, ct).unwrap();
    out
}

/// Offset of the first byte of entropy-coded data (just past the first SOS header).
fn scan_start(b: &[u8]) -> usize {
    let sos = b.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
    sos + 2 + u16::from_be_bytes([b[sos + 2], b[sos + 3]]) as usize
}

const TRUNCATED: DecodeWarning = DecodeWarning::Truncated { format: Format::Jpeg };

#[test]
fn truncated_jpeg_decodes_with_a_warning() {
    for layout in [ChannelLayout::Gray, ChannelLayout::Rgb, ChannelLayout::Cmyk] {
        for progressive in [false, true] {
            let full = jpeg(layout, progressive);
            assert_eq!(decode(&full).unwrap().warnings, [], "{layout:?} progressive {progressive}");
            let half = decode(&full[..full.len() / 2]).unwrap();
            assert_eq!(half.warnings, [TRUNCATED], "{layout:?} progressive {progressive}");
            assert_eq!(half.dimensions(), (64, 48));
            // One byte short: the end-of-image marker is incomplete.
            assert_eq!(decode(&full[..full.len() - 1]).unwrap().warnings, [TRUNCATED]);
            // A few bytes of scan data still decode, with the warning.
            let start = scan_start(&full);
            assert_eq!(decode(&full[..start + 8]).unwrap().warnings, [TRUNCATED]);
        }
    }
}

#[test]
fn truncated_jpeg_message_names_the_problem() {
    let s = TRUNCATED.to_string();
    assert!(s.starts_with("JPEG data ends early") && s.contains("truncated"), "{s}");
}

#[test]
fn jpeg_without_any_scan_data_is_an_error() {
    let full = jpeg(ChannelLayout::Rgb, false);
    let start = scan_start(&full);
    for cut in [start - 3, start - 1, start] {
        let e = decode(&full[..cut]).unwrap_err().to_string();
        assert!(e.contains("before any image data"), "cut {cut}: {e}");
    }
}

#[test]
fn data_after_the_end_of_image_is_not_truncation() {
    let mut b = jpeg(ChannelLayout::Rgb, false);
    b.extend_from_slice(b"\0\0trailing bytes \xFF\xD8\xFF");
    assert_eq!(decode(&b).unwrap().warnings, []);
}

#[test]
fn an_end_marker_inside_metadata_does_not_hide_truncation() {
    // An EXIF block holding FF D9 (as an embedded thumbnail does) must be skipped as a segment.
    let mut img = synth(64, 48, ChannelLayout::Rgb, SampleType::U8, 3, 0.3);
    let mut exif = sample_exif();
    exif.extend_from_slice(&[0xFF, 0xD8, 0x00, 0xFF, 0xD9]);
    img.meta.exif = Some(exif);
    let full = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
    assert_eq!(decode(&full).unwrap().warnings, []);
    assert_eq!(decode(&full[..full.len() * 2 / 3]).unwrap().warnings, [TRUNCATED]);
}

#[test]
fn restart_markers_and_fill_bytes_are_scan_data() {
    // Restart markers (FF D0–D7) and fill bytes (FF FF) inside a scan don't end it.
    let full = jpeg(ChannelLayout::Rgb, false);
    let start = scan_start(&full);
    let mut b = full[..start].to_vec();
    b.extend_from_slice(&[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD0, 0x56, 0xFF, 0xFF, 0xD7, 0x78]);
    assert_eq!(decode(&b).unwrap().warnings, [TRUNCATED]);
    b.extend_from_slice(&[0xFF, 0xFF, 0xD9]);
    assert_eq!(decode(&b).unwrap().warnings, []);
}

// ---------------------------------------------------------------------------
// #523: only the first frame or page is decoded
// ---------------------------------------------------------------------------

fn more_frames(n: u32) -> Vec<DecodeWarning> {
    vec![DecodeWarning::MoreFrames { total: Some(n) }]
}

fn gif(frames: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = image::codecs::gif::GifEncoder::new(&mut out);
    let frames = COLOURS.iter().take(frames).map(|c| image::Frame::new(image::RgbaImage::from_pixel(W, H, image::Rgba([c[0], c[1], c[2], 255]))));
    enc.encode_frames(frames).unwrap();
    drop(enc);
    out
}

#[test]
fn animated_gif_warns_and_keeps_the_first_frame() {
    let one = decode(&gif(1)).unwrap();
    assert_eq!(one.warnings, []);
    for n in [2, 3] {
        let img = decode(&gif(n)).unwrap();
        assert_eq!(img.warnings, more_frames(n as u32));
        assert_eq!(img.to_rgba8(), one.to_rgba8(), "{n} frames: first frame pixels");
    }
    // A single-frame GIF written by our own encoder.
    let ours = encode(&solid(COLOURS[0]), Format::Gif, &EncodeOptions::default()).unwrap();
    assert_eq!(decode(&ours).unwrap().warnings, []);
}

#[test]
fn gif_without_a_trailer_warns_without_a_total() {
    let b = gif(3);
    assert_eq!(b.last(), Some(&0x3B));
    let img = decode(&b[..b.len() - 1]).unwrap();
    assert_eq!(img.warnings, [DecodeWarning::MoreFrames { total: None }]);
    assert_eq!(img.warnings[0].to_string(), "only the first frame of the animation was imported");
}

/// An APNG of `frames` solid frames at 8 or 16 bits; with `separate_default` the IDAT image is
/// an extra default image outside the animation.
fn apng(frames: u32, sixteen: bool, separate_default: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, W, H);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(if sixteen { png::BitDepth::Sixteen } else { png::BitDepth::Eight });
    let images = frames + u32::from(separate_default);
    if frames > 0 {
        enc.set_animated(frames, 0).unwrap();
        enc.set_sep_def_img(separate_default).unwrap();
    }
    let mut w = enc.write_header().unwrap();
    for c in COLOURS.iter().cycle().take(images.max(1) as usize) {
        let px: Vec<u8> = if sixteen { c.iter().flat_map(|&v| [v, v]).collect::<Vec<u8>>().repeat((W * H) as usize) } else { c.repeat((W * H) as usize) };
        w.write_image_data(&px).unwrap();
    }
    w.finish().unwrap();
    out
}

#[test]
fn apng_warns_and_keeps_the_first_frame() {
    for sixteen in [false, true] {
        let still = decode(&apng(0, sixteen, false)).unwrap();
        assert_eq!(still.warnings, []);
        assert_eq!(decode(&apng(1, sixteen, false)).unwrap().warnings, [], "a one-frame APNG is a still");
        let img = decode(&apng(3, sixteen, false)).unwrap();
        assert_eq!(img.warnings, more_frames(3), "16-bit {sixteen}");
        assert_eq!((img.sample_type(), img.data()), (still.sample_type(), still.data()), "16-bit {sixteen}: first frame pixels");
        // Default image + 2 animation frames: three images, the default one kept.
        let img = decode(&apng(2, sixteen, true)).unwrap();
        assert_eq!(img.warnings, more_frames(3));
        assert_eq!(img.data(), still.data());
    }
}

fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut c = id.to_vec();
    c.extend_from_slice(&(data.len() as u32).to_le_bytes());
    c.extend_from_slice(data);
    if data.len() % 2 == 1 {
        c.push(0);
    }
    c
}

fn u24(v: u32) -> [u8; 3] {
    let b = v.to_le_bytes();
    [b[0], b[1], b[2]]
}

/// An animated WebP: each frame is a full-canvas lossless still wrapped in an ANMF chunk.
fn animated_webp(frames: &[Image]) -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    let mut vp8x = vec![0x02, 0, 0, 0]; // animation flag
    vp8x.extend_from_slice(&u24(W - 1));
    vp8x.extend_from_slice(&u24(H - 1));
    body.extend(chunk(b"VP8X", &vp8x));
    body.extend(chunk(b"ANIM", &[0, 0, 0, 0, 0, 0]));
    for f in frames {
        let still = encode(f, Format::WebP, &EncodeOptions::default()).unwrap();
        let at = still.windows(4).position(|w| w == b"VP8L").unwrap();
        let len = u32::from_le_bytes(still[at + 4..at + 8].try_into().unwrap()) as usize;
        let mut anmf = vec![0; 6]; // frame x, y
        anmf.extend_from_slice(&u24(W - 1));
        anmf.extend_from_slice(&u24(H - 1));
        anmf.extend_from_slice(&u24(100)); // duration
        anmf.push(0);
        anmf.extend(chunk(b"VP8L", &still[at + 8..at + 8 + len]));
        body.extend(chunk(b"ANMF", &anmf));
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend(body);
    out
}

#[test]
fn animated_webp_warns_and_keeps_the_first_frame() {
    let frames: Vec<Image> = COLOURS.iter().map(|&c| solid(c)).collect();
    let still = decode(&encode(&frames[0], Format::WebP, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(still.warnings, []);
    assert_eq!(decode(&animated_webp(&frames[..1])).unwrap().warnings, [], "a one-frame animation");
    for n in [2, 3] {
        let img = decode(&animated_webp(&frames[..n])).unwrap();
        assert_eq!(img.warnings, more_frames(n as u32));
        // The animation decoder composites the frame onto its canvas: within one level.
        let (a, b) = (img.to_rgba8(), still.to_rgba8());
        assert!(a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.abs_diff(*y) <= 1), "{n} frames: first frame pixels");
    }
}

/// A TIFF with one IFD per entry: `(NewSubfileType, sample type)`.
fn tiff(ifds: &[(u32, SampleType)]) -> Vec<u8> {
    use tiff::encoder::{TiffEncoder, colortype};
    use tiff::tags::Tag;
    let mut out = std::io::Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut out).unwrap();
    let n = (W * H) as usize;
    // Each page is a flat grey at its own level (`k`), written at `sample` depth.
    for (k, &(kind, sample)) in ifds.iter().enumerate() {
        macro_rules! page {
            ($ct:ty, $data:expr) => {{
                let mut im = enc.new_image::<$ct>(W, H).unwrap();
                if kind != 0 {
                    im.encoder().write_tag(Tag::NewSubfileType, kind).unwrap();
                }
                im.write_data($data).unwrap();
            }};
        }
        let v = 40 * k as u8 + 10;
        match sample {
            SampleType::U8 => page!(colortype::Gray8, &vec![v; n]),
            SampleType::U16 => page!(colortype::Gray16, &vec![u16::from(v) * 257; n]),
            _ => page!(colortype::Gray32Float, &vec![f32::from(v) / 255.0; n]),
        }
    }
    out.into_inner()
}

#[test]
fn multi_page_tiff_warns_and_keeps_the_first_page() {
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let one = decode(&tiff(&[(0, sample)])).unwrap();
        assert_eq!(one.warnings, [], "{sample:?}");
        let img = decode(&tiff(&[(0, sample), (2, sample)])).unwrap();
        assert_eq!(img.warnings, [DecodeWarning::MorePages { total: Some(2) }], "{sample:?}");
        assert_eq!(img.data(), one.data(), "{sample:?}: first page pixels");
        let img = decode(&tiff(&[(0, sample), (0, sample), (0, sample)])).unwrap();
        assert_eq!(img.warnings, [DecodeWarning::MorePages { total: Some(3) }], "{sample:?}");
        assert_eq!(img.warnings[0].to_string(), "only the first of 3 pages was imported");
    }
}

#[test]
fn tiff_thumbnails_and_masks_are_not_pages() {
    let u8 = SampleType::U8;
    // A reduced-resolution copy (bit 0) or a transparency mask (bit 2) is part of the page.
    assert_eq!(decode(&tiff(&[(0, u8), (1, u8)])).unwrap().warnings, []);
    assert_eq!(decode(&tiff(&[(0, u8), (4, u8)])).unwrap().warnings, []);
    assert_eq!(decode(&tiff(&[(0, u8), (1, u8), (0, u8)])).unwrap().warnings, [DecodeWarning::MorePages { total: Some(2) }]);
}

#[test]
fn single_images_carry_no_warnings() {
    for f in rw_formats() {
        let img = synth(9, 7, ChannelLayout::Rgba, SampleType::U8, 5, 0.2);
        let Ok(bytes) = encode(&img, f, &EncodeOptions::default()) else { continue };
        assert_eq!(decode(&bytes).unwrap().warnings, [], "{f:?}");
    }
}

#[test]
fn warnings_survive_orientation() {
    let mut img = synth(8, 4, ChannelLayout::Rgb, SampleType::U8, 1, 0.0);
    img.warnings.push(DecodeWarning::MorePages { total: None });
    for o in 1..=8 {
        assert_eq!(img.clone().oriented(o).unwrap().warnings, img.warnings, "orientation {o}");
    }
    assert_eq!(img.convert(ChannelLayout::Rgba, SampleType::U16).warnings, img.warnings);
}
