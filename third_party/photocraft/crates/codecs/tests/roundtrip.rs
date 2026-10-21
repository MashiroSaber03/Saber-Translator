//! Round-trip every writable format × every sample type × every layout.
//! Lossless formats must reproduce the (planned) conversion of the input
//! exactly; lossy formats must stay above a PSNR threshold.

mod common;
use common::*;
use photocraft_codecs::*;

/// Minimum PSNR (dB) for lossy formats at default options.
fn psnr_threshold(format: Format) -> f64 {
    match format {
        Format::Jpeg => 30.0,
        Format::Gif => 20.0,
        Format::Hdr => 40.0,
        Format::Avif => 28.0,
        _ => f64::INFINITY,
    }
}

fn check_roundtrip(format: Format, layout: ChannelLayout, sample: SampleType) {
    let c = caps(format);
    if !c.write {
        return;
    }
    // GIF quantizes; keep its input smooth so the threshold is meaningful.
    let img = if format == Format::Gif {
        synth(37, 23, layout, sample, 7, 0.0)
    } else if c.lossy {
        synth(37, 23, layout, sample, 7, 0.02)
    } else {
        test_image(layout, sample)
    };
    let opts = EncodeOptions::default();
    let bytes = encode(&img, format, &opts).unwrap_or_else(|e| panic!("{format:?} {layout:?} {sample:?}: {e}"));
    assert_eq!(detect(&bytes), Some(format), "detect after encode");
    if !c.read {
        return;
    }
    let back = decode(&bytes).unwrap_or_else(|e| panic!("{format:?} {layout:?} {sample:?} decode: {e}"));
    assert_eq!(back.dimensions(), img.dimensions());

    // Native storage must be preserved when the format declares support.
    let float_pnm_special = format == Format::Pnm && sample.is_float();
    if c.layouts.contains(&layout) && c.depths.contains(&sample) && !float_pnm_special {
        assert_eq!(back.layout(), layout, "{format:?} must keep layout");
        assert_eq!(back.sample_type(), sample, "{format:?} must keep sample type");
    }
    if c.depths.contains(&sample) && !float_pnm_special {
        assert_eq!(back.sample_type(), sample, "{format:?} must keep supported depth");
    }
    // Integer sources never lose precision in formats offering that depth.
    if sample == SampleType::U16 && c.depths.contains(&SampleType::U16) {
        assert!(matches!(back.sample_type(), SampleType::U16 | SampleType::F32));
    }

    let expected = img.convert(back.layout(), back.sample_type());
    let mut expected_cmp = expected.clone();
    if format == Format::Gif && layout.has_alpha() {
        // GIF alpha is binary: compare colour only on opaque pixels.
        let ch = back.layout().channels();
        let mut e = expected.to_normalized();
        let b = back.to_normalized();
        for (pe, pb) in e.chunks_mut(ch).zip(b.chunks(ch)) {
            if pb[ch - 1] == 0.0 {
                pe.copy_from_slice(pb);
            }
            pe[ch - 1] = pb[ch - 1];
        }
        expected_cmp = Image::from_normalized(img.width(), img.height(), back.layout(), back.sample_type(), &e).unwrap();
    }

    if c.lossy {
        let p = psnr(&expected_cmp, &back);
        assert!(p >= psnr_threshold(format), "{format:?} {layout:?} {sample:?}: PSNR {p:.2} dB");
    } else {
        assert_eq!(
            expected_cmp.data(),
            back.data(),
            "{format:?} {layout:?} {sample:?}: lossless round-trip mismatch (max diff {})",
            max_abs_diff(&expected_cmp, &back)
        );
    }
}

macro_rules! cases {
    ($fmt:expr) => {
        use super::*;
        cases!(@ $fmt;
            gray_u8: Gray, U8; gray_u16: Gray, U16; gray_f16: Gray, F16; gray_f32: Gray, F32;
            graya_u8: GrayA, U8; graya_u16: GrayA, U16; graya_f16: GrayA, F16; graya_f32: GrayA, F32;
            rgb_u8: Rgb, U8; rgb_u16: Rgb, U16; rgb_f16: Rgb, F16; rgb_f32: Rgb, F32;
            rgba_u8: Rgba, U8; rgba_u16: Rgba, U16; rgba_f16: Rgba, F16; rgba_f32: Rgba, F32;
            cmyk_u8: Cmyk, U8; cmyk_u16: Cmyk, U16; cmyk_f16: Cmyk, F16; cmyk_f32: Cmyk, F32;
            cmyka_u8: CmykA, U8; cmyka_u16: CmykA, U16; cmyka_f16: CmykA, F16; cmyka_f32: CmykA, F32
        );
    };
    (@ $fmt:expr; $($name:ident : $l:ident, $s:ident);*) => {
        $(
            #[test]
            fn $name() {
                check_roundtrip($fmt, ChannelLayout::$l, SampleType::$s);
            }
        )*
    };
}

mod png {
    cases!(Format::Png);
}
mod jpeg {
    cases!(Format::Jpeg);
}
mod tiff {
    cases!(Format::Tiff);
}
mod webp {
    cases!(Format::WebP);
}
mod gif {
    cases!(Format::Gif);
}
mod bmp {
    cases!(Format::Bmp);
}
mod tga {
    cases!(Format::Tga);
}
mod ico {
    cases!(Format::Ico);
}
mod pnm {
    cases!(Format::Pnm);
}
mod fits {
    cases!(Format::Fits);
}
mod dpx {
    cases!(Format::Dpx);
}
mod qoi {
    cases!(Format::Qoi);
}
mod exr {
    cases!(Format::OpenExr);
}
mod hdr {
    cases!(Format::Hdr);
}

// ---------------------------------------------------------------------------
// Targeted round-trips beyond the matrix
// ---------------------------------------------------------------------------

/// Float formats must keep HDR values (> 1, and negative where allowed).
#[test]
fn hdr_values_survive_float_formats() {
    let vals: Vec<f32> = (0..16 * 8 * 3).map(|i| (i as f32 - 40.0) * 0.37).collect();
    let img = Image::from_f32(16, 8, ChannelLayout::Rgb, &vals).unwrap();
    for f in [Format::Tiff, Format::OpenExr, Format::Pnm] {
        let back = decode(&encode(&img, f, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(back.to_f32_samples().unwrap(), vals, "{f:?}");
    }
}

#[test]
fn hdr_format_keeps_bright_values_approximately() {
    let vals: Vec<f32> = (0..16 * 8 * 3).map(|i| 0.01 + i as f32 * 0.25).collect();
    let img = Image::from_f32(16, 8, ChannelLayout::Rgb, &vals).unwrap();
    let back = decode(&encode(&img, Format::Hdr, &EncodeOptions::default()).unwrap()).unwrap();
    // RGBE shares one exponent per pixel: error is relative to the brightest channel.
    let got = back.to_f32_samples().unwrap();
    for (pa, pb) in vals.chunks(3).zip(got.chunks(3)) {
        let m = pa.iter().copied().fold(0.0, f32::max);
        for (a, b) in pa.iter().zip(pb) {
            assert!((a - b).abs() <= m * 0.01, "{a} vs {b}");
        }
    }
}

#[test]
fn f16_exr_is_bit_exact() {
    let vals: Vec<f16> = (0..20 * 10 * 4).map(|i| f16::from_f32(i as f32 * 0.013 - 1.0)).collect();
    let img = Image::from_f16(20, 10, ChannelLayout::Rgba, &vals).unwrap();
    for comp in [ExrCompression::None, ExrCompression::Rle, ExrCompression::Zip1, ExrCompression::Zip16, ExrCompression::Piz] {
        let opts = EncodeOptions { exr_compression: comp, ..Default::default() };
        let back = decode(&encode(&img, Format::OpenExr, &opts).unwrap()).unwrap();
        assert_eq!(back.to_f16_samples().unwrap(), vals, "{comp:?}");
    }
}

#[test]
fn tiff_all_compressions_lossless() {
    for layout in ChannelLayout::ALL {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let img = test_image(layout, sample);
            for comp in [TiffCompression::None, TiffCompression::Lzw, TiffCompression::Deflate, TiffCompression::PackBits] {
                let opts = EncodeOptions { tiff_compression: comp, ..Default::default() };
                let back = decode(&encode(&img, Format::Tiff, &opts).unwrap()).unwrap();
                assert_eq!(back, img.clone().with_meta(back.meta.clone()), "{layout:?} {sample:?} {comp:?}");
            }
        }
    }
}

#[test]
fn png_all_compressions_lossless() {
    let img = test_image(ChannelLayout::Rgba, SampleType::U16);
    for comp in [PngCompression::None, PngCompression::Fast, PngCompression::Default, PngCompression::Best] {
        let opts = EncodeOptions { png_compression: comp, ..Default::default() };
        let back = decode(&encode(&img, Format::Png, &opts).unwrap()).unwrap();
        assert_eq!(back.data(), img.data(), "{comp:?}");
    }
}

#[test]
fn jpeg_quality_affects_size_and_error() {
    let img = test_image(ChannelLayout::Rgb, SampleType::U8);
    let lo = encode(&img, Format::Jpeg, &EncodeOptions { jpeg_quality: 20, ..Default::default() }).unwrap();
    let hi = encode(&img, Format::Jpeg, &EncodeOptions { jpeg_quality: 98, ..Default::default() }).unwrap();
    assert!(lo.len() < hi.len());
    let plo = psnr(&img, &decode(&lo).unwrap());
    let phi = psnr(&img, &decode(&hi).unwrap());
    assert!(phi > plo, "{phi} <= {plo}");
}

#[test]
fn jpeg_444_beats_420_on_chroma_detail() {
    let img = synth(64, 64, ChannelLayout::Rgb, SampleType::U8, 3, 0.5);
    let a = encode(&img, Format::Jpeg, &EncodeOptions { jpeg_chroma_subsampling: false, jpeg_quality: 95, ..Default::default() }).unwrap();
    let b = encode(&img, Format::Jpeg, &EncodeOptions { jpeg_chroma_subsampling: true, jpeg_quality: 95, ..Default::default() }).unwrap();
    assert!(psnr(&img, &decode(&a).unwrap()) > psnr(&img, &decode(&b).unwrap()));
}

#[test]
fn webp_lossy_request_is_a_lossy_file() {
    let img = test_image(ChannelLayout::Rgb, SampleType::U8);
    let bytes = encode(&img, Format::WebP, &EncodeOptions { webp_lossless: false, ..Default::default() }).unwrap();
    assert_eq!(&bytes[12..16], b"VP8 ", "a bare lossy frame");
    let back = decode(&bytes).unwrap();
    assert_eq!(back.dimensions(), img.dimensions());
    assert!(
        fidelity_warnings_with(&img, Format::WebP, &EncodeOptions { webp_lossless: false, ..Default::default() }).contains(&FidelityWarning::LossyCompression)
    );
}

#[test]
fn one_pixel_images_roundtrip() {
    for f in rw_formats() {
        let img = synth(1, 1, ChannelLayout::Rgb, SampleType::U8, 1, 0.0);
        let back = decode(&encode(&img, f, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(back.dimensions(), (1, 1), "{f:?}");
    }
}

#[test]
fn ico_rejects_oversize() {
    let img = Image::new(300, 10, ChannelLayout::Rgba, SampleType::U8).unwrap();
    assert!(matches!(encode(&img, Format::Ico, &EncodeOptions::default()), Err(CodecError::Encode { .. })));
    let img = Image::new(256, 256, ChannelLayout::Rgba, SampleType::U8).unwrap();
    assert!(encode(&img, Format::Ico, &EncodeOptions::default()).is_ok());
}

#[test]
fn empty_image_rejected() {
    let img = Image::new(0, 5, ChannelLayout::Rgb, SampleType::U8).unwrap();
    for f in writable_formats() {
        assert!(encode(&img, f, &EncodeOptions::default()).is_err(), "{f:?}");
    }
}

#[test]
fn larger_image_png_tiff_exact() {
    let img = synth(513, 257, ChannelLayout::Rgba, SampleType::U16, 99, 1.0);
    for f in [Format::Png, Format::Tiff, Format::Pnm] {
        let back = decode(&encode(&img, f, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(back.data(), img.data(), "{f:?}");
    }
}
