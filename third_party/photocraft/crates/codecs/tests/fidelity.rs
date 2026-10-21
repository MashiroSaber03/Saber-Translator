//! `fidelity_warnings` cases, and consistency with what `encode` does.

mod common;
use FidelityWarning as W;
use common::*;
use photocraft_codecs::*;

fn warns(img: &Image, f: Format) -> Vec<FidelityWarning> {
    fidelity_warnings(img, f)
}

#[test]
fn png_u8_rgb_is_clean() {
    assert!(warns(&test_image(ChannelLayout::Rgb, SampleType::U8), Format::Png).is_empty());
}
#[test]
fn png_u16_rgba_is_clean() {
    assert!(warns(&test_image(ChannelLayout::Rgba, SampleType::U16), Format::Png).is_empty());
}
#[test]
fn tiff_is_clean_for_all_integer_and_f32_layouts() {
    for l in ChannelLayout::ALL {
        for s in [SampleType::U8, SampleType::U16, SampleType::F32, SampleType::F16] {
            assert!(warns(&test_image(l, s), Format::Tiff).is_empty(), "{l:?} {s:?}");
        }
    }
}
#[test]
fn sixteen_bit_to_jpeg_reduces() {
    let w = warns(&test_image(ChannelLayout::Rgb, SampleType::U16), Format::Jpeg);
    assert!(w.contains(&W::DepthReduced { from: SampleType::U16, to: SampleType::U8 }));
    assert!(w.contains(&W::LossyCompression));
}
#[test]
fn sixteen_bit_message_text() {
    let w = W::DepthReduced { from: SampleType::U16, to: SampleType::U8 };
    assert_eq!(w.to_string(), "16-bit will be reduced to 8-bit");
}
#[test]
fn alpha_to_jpeg_discarded() {
    let w = warns(&test_image(ChannelLayout::Rgba, SampleType::U8), Format::Jpeg);
    assert!(w.contains(&W::AlphaDiscarded));
}
#[test]
fn opaque_alpha_not_reported() {
    let mut img = test_image(ChannelLayout::Rgba, SampleType::U8);
    for px in img.data_mut().chunks_mut(4) {
        px[3] = 255;
    }
    assert!(!warns(&img, Format::Jpeg).contains(&W::AlphaDiscarded));
}
#[test]
fn alpha_to_hdr_discarded() {
    assert!(warns(&test_image(ChannelLayout::Rgba, SampleType::F32), Format::Hdr).contains(&W::AlphaDiscarded));
}
#[test]
fn cmyk_to_png_converted() {
    let w = warns(&test_image(ChannelLayout::Cmyk, SampleType::U8), Format::Png);
    assert!(w.contains(&W::CmykConverted { to: ChannelLayout::Rgb }));
}
#[test]
fn cmyka_to_webp_converted_keeps_alpha() {
    let w = warns(&test_image(ChannelLayout::CmykA, SampleType::U8), Format::WebP);
    assert!(w.contains(&W::CmykConverted { to: ChannelLayout::Rgba }));
    assert!(!w.contains(&W::AlphaDiscarded));
}
#[test]
fn cmyk_to_jpeg_and_tiff_not_converted() {
    for f in [Format::Jpeg, Format::Tiff, Format::Pnm] {
        let w = warns(&test_image(ChannelLayout::Cmyk, SampleType::U8), f);
        assert!(!w.iter().any(|w| matches!(w, W::CmykConverted { .. })), "{f:?}");
    }
}
#[test]
fn icc_dropped_for_bmp() {
    let img = test_image(ChannelLayout::Rgb, SampleType::U8).with_icc(Some(sample_icc(100)));
    assert!(warns(&img, Format::Bmp).contains(&W::IccDropped));
    assert!(!warns(&img, Format::Png).contains(&W::IccDropped));
}
#[test]
fn icc_warning_suppressed_when_not_embedding() {
    let img = test_image(ChannelLayout::Rgb, SampleType::U8).with_icc(Some(sample_icc(100)));
    let opts = EncodeOptions { embed_icc: false, ..Default::default() };
    assert!(!fidelity_warnings_with(&img, Format::Bmp, &opts).contains(&W::IccDropped));
}
#[test]
fn exif_xmp_dpi_text_dropped_for_qoi() {
    let mut img = test_image(ChannelLayout::Rgb, SampleType::U8);
    img.meta =
        Metadata { exif: Some(sample_exif()), xmp: Some("x".into()), dpi: Some((72.0, 72.0)), text: vec![("a".into(), "b".into())], ..Default::default() };
    let w = warns(&img, Format::Qoi);
    for x in [W::ExifDropped, W::XmpDropped, W::DpiDropped, W::TextDropped] {
        assert!(w.contains(&x), "{x:?}");
    }
}
#[test]
fn webp_drops_dpi_and_text_only() {
    let mut img = test_image(ChannelLayout::Rgb, SampleType::U8);
    img.meta =
        Metadata { exif: Some(sample_exif()), xmp: Some("x".into()), dpi: Some((72.0, 72.0)), text: vec![("a".into(), "b".into())], ..Default::default() };
    let w = warns(&img, Format::WebP);
    assert_eq!(w, vec![W::DpiDropped, W::TextDropped]);
}
#[test]
fn float_to_png_reduces_and_clips() {
    let vals: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 0.5).collect();
    let img = Image::from_f32(2, 2, ChannelLayout::Rgb, &vals).unwrap();
    let w = warns(&img, Format::Png);
    assert!(w.contains(&W::DepthReduced { from: SampleType::F32, to: SampleType::U16 }));
    assert!(w.contains(&W::RangeClipped));
}
#[test]
fn in_range_float_not_clipped() {
    let w = warns(&test_image(ChannelLayout::Rgb, SampleType::F32), Format::Png);
    assert!(!w.contains(&W::RangeClipped));
}
#[test]
fn float_to_exr_and_tiff_is_clean() {
    for f in [Format::OpenExr, Format::Tiff, Format::Pnm] {
        assert!(warns(&test_image(ChannelLayout::Rgb, SampleType::F32), f).is_empty(), "{f:?}");
    }
}
#[test]
fn f16_to_exr_is_clean() {
    assert!(warns(&test_image(ChannelLayout::Rgba, SampleType::F16), Format::OpenExr).is_empty());
}
#[test]
fn integer_to_exr_is_lossless() {
    for s in [SampleType::U8, SampleType::U16] {
        assert!(warns(&test_image(ChannelLayout::Rgb, s), Format::OpenExr).is_empty(), "{s:?}");
    }
}
#[test]
fn float_pnm_drops_alpha() {
    let w = warns(&test_image(ChannelLayout::Rgba, SampleType::F32), Format::Pnm);
    assert!(w.contains(&W::AlphaDiscarded));
}
#[test]
fn gif_quantizes_and_binarizes() {
    let w = warns(&test_image(ChannelLayout::Rgba, SampleType::U8), Format::Gif);
    assert!(w.contains(&W::PaletteQuantized));
    assert!(w.contains(&W::AlphaBinarized));
    assert!(!w.contains(&W::LossyCompression));
}
#[test]
fn hdr_is_lossy() {
    assert!(warns(&test_image(ChannelLayout::Rgb, SampleType::F32), Format::Hdr).contains(&W::LossyCompression));
}
#[test]
fn ico_oversize_is_fatal() {
    let img = Image::new(512, 16, ChannelLayout::Rgba, SampleType::U8).unwrap();
    let w = warns(&img, Format::Ico);
    assert!(w.iter().any(|w| w.is_fatal()));
    assert!(w.contains(&W::DimensionsExceeded { max_width: 256, max_height: 256 }));
}
#[test]
fn avif_default_build_write_unsupported() {
    let w = warns(&test_image(ChannelLayout::Rgb, SampleType::U8), Format::Avif);
    if cfg!(feature = "avif") {
        assert!(w.contains(&W::LossyCompression));
    } else {
        assert_eq!(w, vec![W::WriteUnsupported { format: Format::Avif }]);
        assert!(w[0].is_fatal());
    }
}
#[test]
fn every_warning_has_a_message() {
    let all = [
        W::WriteUnsupported { format: Format::Avif },
        W::DepthReduced { from: SampleType::F32, to: SampleType::F16 },
        W::RangeClipped,
        W::AlphaDiscarded,
        W::AlphaBinarized,
        W::CmykConverted { to: ChannelLayout::Rgb },
        W::ColorToGray,
        W::PaletteQuantized,
        W::LossyCompression,
        W::IccDropped,
        W::ExifDropped,
        W::XmpDropped,
        W::DpiDropped,
        W::TextDropped,
        W::DimensionsExceeded { max_width: 1, max_height: 1 },
    ];
    for w in all {
        assert!(w.to_string().len() > 5);
    }
}

/// The warnings must agree with what actually comes back from the file.
#[test]
fn warnings_match_actual_roundtrip_behaviour() {
    for f in rw_formats() {
        for l in ChannelLayout::ALL {
            for s in SampleType::ALL {
                let img = test_image(l, s);
                let w = warns(&img, f);
                let back = decode(&encode(&img, f, &EncodeOptions::default()).unwrap()).unwrap();
                let depth_reduced = w.iter().any(|w| matches!(w, W::DepthReduced { .. }));
                let lost_depth = back.sample_type() != s && back.sample_type() != SampleType::F32 && s != SampleType::U8;
                assert_eq!(depth_reduced, lost_depth, "{f:?} {l:?} {s:?} {w:?} -> {:?}", back.sample_type());
                let alpha_lost = l.has_alpha() && !back.layout().has_alpha();
                assert_eq!(w.contains(&W::AlphaDiscarded), alpha_lost, "{f:?} {l:?} {s:?}");
                let cmyk_lost = l.is_cmyk() && !back.layout().is_cmyk();
                assert_eq!(w.iter().any(|w| matches!(w, W::CmykConverted { .. })), cmyk_lost, "{f:?} {l:?} {s:?}");
            }
        }
    }
}
