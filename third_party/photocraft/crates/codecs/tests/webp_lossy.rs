//! Lossy WebP (our VP8 encoder) checked against an independent decoder (`image-webp`): the
//! files decode, look like the input (PSNR by quality), get smaller as quality drops and are far
//! smaller than lossless on photographic content; alpha and metadata travel in the container;
//! odd sizes and the limits behave.

mod common;
use common::*;
use photocraft_codecs::*;

fn lossy(q: u8) -> EncodeOptions {
    EncodeOptions { webp_lossless: false, webp_quality: q, ..Default::default() }
}

/// A smooth photograph-like image: gradients with a little grain.
fn photo(w: u32, h: u32, seed: u64) -> Image {
    synth(w, h, ChannelLayout::Rgb, SampleType::U8, seed, 0.03)
}

#[test]
fn decodes_with_the_right_size_and_looks_like_the_input() {
    let img = photo(97, 61, 3);
    for (q, min_psnr) in [(100u8, 40.0), (90, 36.0), (75, 33.0), (50, 30.0), (20, 26.0)] {
        let bytes = encode(&img, Format::WebP, &lossy(q)).unwrap();
        let back = decode(&bytes).unwrap_or_else(|e| panic!("q{q}: {e}"));
        assert_eq!(back.dimensions(), (97, 61));
        assert_eq!(back.layout(), ChannelLayout::Rgb);
        let p = psnr(&img, &back);
        assert!(p >= min_psnr, "quality {q}: PSNR {p:.1} dB below {min_psnr}");
    }
}

#[test]
fn smaller_at_lower_quality_and_much_smaller_than_lossless_on_photos() {
    let img = synth(256, 192, ChannelLayout::Rgb, SampleType::U8, 9, 0.08);
    let lossless = encode(&img, Format::WebP, &EncodeOptions::default()).unwrap().len();
    let sizes: Vec<usize> = [95u8, 80, 60, 40, 20].iter().map(|&q| encode(&img, Format::WebP, &lossy(q)).unwrap().len()).collect();
    for w in sizes.windows(2) {
        assert!(w[1] <= w[0], "sizes must not grow as quality drops: {sizes:?}");
    }
    assert!(sizes[1] * 4 < lossless, "q80 {} vs lossless {lossless}", sizes[1]);
    assert!(sizes[4] * 10 < lossless, "q20 {} vs lossless {lossless}", sizes[4]);
}

#[test]
fn flat_areas_cost_almost_nothing() {
    // A single colour: every macroblock is skipped; the file is dominated by the header.
    let img = Image::from_u8(640, 480, ChannelLayout::Rgb, vec![200u8; 640 * 480 * 3]).unwrap();
    let bytes = encode(&img, Format::WebP, &lossy(80)).unwrap();
    assert!(bytes.len() < 1200, "{} bytes", bytes.len());
    let back = decode(&bytes).unwrap();
    assert!(max_abs_diff(&img, &back) <= 3.0 / 255.0, "flat colour reproduced within rounding");
}

#[test]
fn alpha_is_kept_exactly_and_metadata_travels() {
    let mut img = synth(40, 30, ChannelLayout::Rgba, SampleType::U8, 5, 0.05);
    img.icc = Some(sample_icc(400));
    img.meta.exif = Some(sample_exif());
    img.meta.xmp = Some(SAMPLE_XMP.to_owned());
    let bytes = encode(&img, Format::WebP, &lossy(85)).unwrap();
    assert_eq!(&bytes[12..16], b"VP8X", "extended container");
    let back = decode(&bytes).unwrap();
    assert_eq!(back.layout(), ChannelLayout::Rgba);
    for (a, b) in img.data().as_chunks::<4>().0.iter().zip(back.data().as_chunks::<4>().0) {
        assert_eq!(a[3], b[3], "alpha is stored losslessly");
    }
    assert!(psnr(&img, &back) > 30.0);
    assert_eq!(back.icc, img.icc);
    assert!(back.meta.xmp.is_some());
    assert!(back.meta.exif.is_some());
    // Fully opaque alpha is dropped rather than stored.
    let mut data = synth(40, 30, ChannelLayout::Rgba, SampleType::U8, 5, 0.0).data().to_vec();
    for px in data.as_chunks_mut::<4>().0 {
        px[3] = 255;
    }
    let opaque = Image::from_u8(40, 30, ChannelLayout::Rgba, data).unwrap();
    let bytes = encode(&opaque, Format::WebP, &EncodeOptions { embed_icc: false, embed_metadata: false, ..lossy(85) }).unwrap();
    assert_eq!(&bytes[12..16], b"VP8 ");
    assert_eq!(decode(&bytes).unwrap().layout(), ChannelLayout::Rgb);
}

#[test]
fn gray_and_odd_sizes_and_tiny_images() {
    for (w, h) in [(1, 1), (2, 3), (17, 9), (33, 1), (1, 47), (16, 16), (31, 15)] {
        for layout in [ChannelLayout::Gray, ChannelLayout::GrayA, ChannelLayout::Rgb, ChannelLayout::Rgba] {
            // Little noise: at these sizes the 2 × 2 chroma average alone would otherwise dominate.
            let img = synth(w, h, layout, SampleType::U8, u64::from(w * 100 + h), 0.02);
            let bytes = encode(&img, Format::WebP, &lossy(80)).unwrap_or_else(|e| panic!("{w}x{h} {layout:?}: {e}"));
            let back = decode(&bytes).unwrap_or_else(|e| panic!("{w}x{h} {layout:?}: {e}"));
            assert_eq!(back.dimensions(), (w, h));
            // Below 8 px the 2 × 2 chroma average of random saturated pixels is the whole error.
            if w >= 8 && h >= 8 {
                let p = psnr(&img.convert(back.layout(), SampleType::U8), &back);
                assert!(p > 24.0, "{w}x{h} {layout:?}: {p:.1} dB");
            }
        }
    }
}

#[test]
fn sixteen_bit_input_is_quantised_to_eight() {
    let img = synth(24, 24, ChannelLayout::Rgb, SampleType::U16, 2, 0.02);
    let bytes = encode(&img, Format::WebP, &lossy(90)).unwrap();
    let back = decode(&bytes).unwrap();
    assert!(psnr(&img.convert(ChannelLayout::Rgb, SampleType::U8), &back) > 34.0);
}

#[test]
fn dimension_limit_is_an_error_not_a_panic() {
    let img = Image::from_u8(16384, 1, ChannelLayout::Rgb, vec![0u8; 16384 * 3]).unwrap();
    let e = encode(&img, Format::WebP, &lossy(50)).unwrap_err();
    assert!(matches!(e, CodecError::Encode { .. }), "{e}");
    let img = Image::from_u8(16383, 1, ChannelLayout::Rgb, vec![0u8; 16383 * 3]).unwrap();
    let bytes = encode(&img, Format::WebP, &lossy(50)).unwrap();
    assert_eq!(decode(&bytes).unwrap().dimensions(), (16383, 1));
}

#[test]
fn hard_edges_and_text_like_content_keep_their_structure() {
    // Black text-like strokes on white: B_PRED sub-block modes keep edges sharp enough that the
    // thresholded result matches the source almost everywhere.
    let (w, h) = (128u32, 64u32);
    let mut data = vec![255u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let stroke = (x % 11 < 2 && y > 8 && y < 56) || (y % 13 < 2 && x > 4 && x < 124);
            if stroke {
                let i = ((y * w + x) * 3) as usize;
                data[i..i + 3].fill(0);
            }
        }
    }
    let img = Image::from_u8(w, h, ChannelLayout::Rgb, data).unwrap();
    let bytes = encode(&img, Format::WebP, &lossy(90)).unwrap();
    let back = decode(&bytes).unwrap();
    let wrong = img.data().as_chunks::<3>().0.iter().zip(back.data().as_chunks::<3>().0).filter(|(a, b)| (a[0] < 128) != (b[0] < 128)).count();
    let total = (w * h) as usize;
    assert!(wrong * 100 < total * 2, "{wrong} of {total} pixels flipped side of mid-gray");
    assert!(psnr(&img, &back) > 30.0, "{:.1}", psnr(&img, &back));
}
