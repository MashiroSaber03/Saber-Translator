//! `decode_gif_frames`: every frame of an animated GIF, composited, with its delay.

use photocraft_codecs::{AnimationFrame, CodecError, DecodeOptions, DecodeWarning, Format, Limits, decode, decode_gif_frames};

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const CLEAR: [u8; 4] = [0, 0, 0, 0];

fn frames(bytes: &[u8]) -> Vec<AnimationFrame> {
    decode_gif_frames(bytes, &DecodeOptions::default()).unwrap()
}

fn pixels(f: &AnimationFrame) -> Vec<u8> {
    f.image.to_rgba8()
}

/// Solid full-screen frames written by the `image` crate's encoder, `delay_ms` apart.
fn solid_gif(colours: &[[u8; 4]], w: u32, h: u32, delay_ms: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = image::codecs::gif::GifEncoder::new(&mut out);
    let list = colours.iter().map(|c| {
        let buf = image::RgbaImage::from_pixel(w, h, image::Rgba(*c));
        image::Frame::from_parts(buf, 0, 0, image::Delay::from_numer_denom_ms(delay_ms, 1))
    });
    enc.encode_frames(list).unwrap();
    drop(enc);
    out
}

/// A palette-index image LZW-packed with a 2-bit minimum code size. Few enough pixels that the
/// code size never grows past 3 bits.
fn lzw2(indices: &[u8]) -> Vec<u8> {
    let mut codes = vec![4u8];
    codes.extend_from_slice(indices);
    codes.push(5);
    let (mut acc, mut nbits, mut out) = (0u32, 0, Vec::new());
    for c in codes {
        acc |= u32::from(c) << nbits;
        nbits += 3;
        while nbits >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push(acc as u8);
    }
    out
}

/// One image of [`raw_gif`]: `(x, y, width, height, disposal method, palette indices)`.
type Part<'a> = (u16, u16, u16, u16, u8, &'a [u8]);

/// A hand-built GIF89a: a `w`x`h` screen with the 4-colour global palette red, green, blue,
/// black, and one image per [`Part`]. Frames are 10 cs apart.
fn raw_gif(w: u16, h: u16, parts: &[Part]) -> Vec<u8> {
    let mut b = b"GIF89a".to_vec();
    b.extend_from_slice(&w.to_le_bytes());
    b.extend_from_slice(&h.to_le_bytes());
    b.extend_from_slice(&[0b1000_0001, 0, 0]);
    b.extend_from_slice(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0]);
    for &(x, y, fw, fh, disposal, indices) in parts {
        // Graphic control extension: disposal method, 10 cs delay, no transparency.
        b.extend_from_slice(&[0x21, 0xF9, 4, disposal << 2, 10, 0, 0, 0]);
        b.push(0x2C);
        for v in [x, y, fw, fh] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&[0, 2]);
        let data = lzw2(indices);
        b.push(data.len() as u8);
        b.extend_from_slice(&data);
        b.push(0);
    }
    b.push(0x3B);
    b
}

#[test]
fn a_still_gif_gives_one_frame_equal_to_decode() {
    let still = solid_gif(&[RED], 3, 2, 100);
    let all = frames(&still);
    assert_eq!(all.len(), 1);
    assert_eq!((all[0].image.width(), all[0].image.height()), (3, 2));
    assert_eq!(pixels(&all[0]), decode(&still).unwrap().to_rgba8());
    assert!(all[0].image.warnings.is_empty());
}

#[test]
fn every_frame_comes_back_in_order_with_its_delay() {
    let all = frames(&solid_gif(&[RED, GREEN, BLUE], 2, 2, 120));
    assert_eq!(all.len(), 3);
    for (f, want) in all.iter().zip([RED, GREEN, BLUE]) {
        assert_eq!(pixels(f), want.repeat(4));
        assert_eq!(f.delay_ms, 120);
    }
}

#[test]
fn a_smaller_frame_is_drawn_over_the_previous_one() {
    // Frame 2 is one pixel at x = 1, kept ("do not dispose", 1): the screen is red, blue.
    let all = frames(&raw_gif(2, 1, &[(0, 0, 2, 1, 1, &[0, 0]), (1, 0, 1, 1, 1, &[2])]));
    assert_eq!(all.len(), 2);
    assert_eq!(pixels(&all[0]), [RED, RED].concat());
    assert_eq!(pixels(&all[1]), [RED, BLUE].concat());
    assert_eq!(all[1].delay_ms, 100);
}

#[test]
fn disposal_to_background_clears_the_previous_frame() {
    // Frame 1 (disposal 2, restore to background) covers the screen; frame 2 draws one pixel.
    let all = frames(&raw_gif(2, 1, &[(0, 0, 2, 1, 2, &[1, 1]), (1, 0, 1, 1, 1, &[2])]));
    assert_eq!(pixels(&all[0]), [GREEN, GREEN].concat());
    assert_eq!(pixels(&all[1]), [CLEAR, BLUE].concat());
}

#[test]
fn disposal_to_previous_restores_what_was_there() {
    // Frame 1 is kept; frame 2 (disposal 3, restore previous) paints one pixel over it, so frame
    // 3, which draws elsewhere, sees frame 1 again.
    let all = frames(&raw_gif(2, 1, &[(0, 0, 2, 1, 1, &[0, 0]), (0, 0, 1, 1, 3, &[1]), (1, 0, 1, 1, 1, &[2])]));
    assert_eq!(pixels(&all[1]), [GREEN, RED].concat());
    assert_eq!(pixels(&all[2]), [RED, BLUE].concat());
}

#[test]
fn a_damaged_later_frame_keeps_the_frames_before_it() {
    let good = solid_gif(&[RED, GREEN, BLUE], 8, 8, 100);
    // Cut inside the last frame's data, so frames 1 and 2 are intact.
    let cut = &good[..good.len() - 12];
    let all = frames(cut);
    assert!((1..3).contains(&all.len()), "{} frames", all.len());
    assert_eq!(all[0].image.warnings, [DecodeWarning::Truncated { format: Format::Gif }]);
    assert!(all[1..].iter().all(|f| f.image.warnings.is_empty()));
}

#[test]
fn the_first_frame_must_decode() {
    assert!(decode_gif_frames(b"", &DecodeOptions::default()).is_err());
    assert!(decode_gif_frames(b"GIF89a", &DecodeOptions::default()).is_err());
    assert!(decode_gif_frames(&[0xFF; 64], &DecodeOptions::default()).is_err());
}

#[test]
fn every_prefix_of_a_gif_returns() {
    let g = solid_gif(&[RED, GREEN, BLUE], 5, 4, 100);
    for n in 0..g.len() {
        let _ = decode_gif_frames(&g[..n], &DecodeOptions::default());
    }
}

#[test]
fn dimension_limits_apply() {
    let g = solid_gif(&[RED], 4, 4, 100);
    let opts = DecodeOptions { limits: Limits { max_width: 3, ..Limits::default() }, ..Default::default() };
    assert!(matches!(decode_gif_frames(&g, &opts), Err(CodecError::LimitExceeded(_))));
}

#[test]
fn the_byte_budget_covers_all_frames_together() {
    // One 4x4 RGBA frame is 64 bytes: two frames fit in 128, three do not.
    let limits = Limits { max_alloc: 128, ..Limits::default() };
    let opts = DecodeOptions { limits, ..Default::default() };
    assert_eq!(decode_gif_frames(&solid_gif(&[RED, GREEN], 4, 4, 100), &opts).unwrap().len(), 2);
    assert!(matches!(decode_gif_frames(&solid_gif(&[RED, GREEN, BLUE], 4, 4, 100), &opts), Err(CodecError::LimitExceeded(_))));
}
