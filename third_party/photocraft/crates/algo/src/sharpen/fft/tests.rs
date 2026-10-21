use super::*;
use crate::sharpen::tests::{reference, source};
use crate::{Ctx, image::Image};
use photocraft_color::{ColorMode, PixelFormat, SampleType};

fn compare(old: &Surface, new: &Surface, area: Rect) {
    let tolerance = match old.format().sample {
        SampleType::U8 => 1.0 / 255.0 + 1e-7,
        SampleType::U16 => 2.0 / 65535.0 + 1e-7,
        SampleType::F32 => 3e-5,
    };
    for (a, b) in old.read_region(area).iter().zip(new.read_region(area)) {
        let limit = if old.format().sample == SampleType::F32 { tolerance * (1.0 + a.abs()) } else { tolerance };
        assert!((a - b).abs() <= limit, "{a} != {b}, tolerance {limit}");
    }
}

#[test]
fn fft_preserves_gaussian_quality_at_every_depth_and_model() {
    let area = Rect::new(-3, 2, 10, 13);
    for mode in [
        ColorMode::Bitmap,
        ColorMode::Grayscale,
        ColorMode::Duotone,
        ColorMode::Indexed,
        ColorMode::Rgb,
        ColorMode::Cmyk,
        ColorMode::Lab,
        ColorMode::Multichannel,
    ] {
        for depth in SampleType::ALL {
            for alpha in [false, true] {
                let s = source(area, PixelFormat::new(mode, depth, alpha));
                let ctx = Ctx { bounds: area, mode, alpha };
                for radius in [3.25, 64.0] {
                    let src = Image::read_clamped(&s, area.inflate((radius * 3.0_f32).ceil() as i32 + 1), area);
                    let pixels = reference::unsharp(&src, area, &ctx, 500.0, radius, 0.0);
                    let mut expected = s.clone();
                    expected.write_region(area, &pixels);
                    let actual = run(&s, area, 500.0, radius, None, Some(area), &Interrupt::NONE).unwrap();
                    compare(&expected, &actual, area);
                    if alpha {
                        for (a, b) in s.read_region(area).chunks_exact(s.channels()).zip(actual.read_region(area).chunks_exact(s.channels())) {
                            assert_eq!(a.last(), b.last());
                            if a.last() == Some(&0.0) {
                                assert_eq!(a, b);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fft_keeps_partial_regions_selections_and_edge_policies() {
    let extent = Rect::new(-7, -9, 35, 39);
    let area = Rect::new(-3, 2, 19, 23);
    let mut sel = Surface::new(PixelFormat::GRAY8);
    sel.fill_rect(area, &[0.4]);
    for depth in SampleType::ALL {
        let s = source(extent, PixelFormat::new(ColorMode::Rgb, depth, true));
        let p = FilterParams::UnsharpMask { amount: 150.0, radius: 64.0, threshold: 0.0 };
        for edge in [None, Some(extent)] {
            let expected = crate::sharpen::tests::apply_reference(&s, &p, area, extent, Some(&sel), edge, 256);
            let actual = run(&s, area, 150.0, 64.0, Some(&sel), edge, &Interrupt::NONE).unwrap();
            compare(&expected, &actual, extent);
        }
    }
}

#[test]
fn fft_falls_back_for_thresholds_and_unstable_samples() {
    let area = Rect::new(0, 0, 256, 256);
    let mut s = source(area, PixelFormat::RGBA32F);
    let p = FilterParams::UnsharpMask { amount: 150.0, radius: 64.0, threshold: 8.0 };
    assert!(apply(&s, &p, area, None, Some(area), &Interrupt::NONE).is_none());
    let p = FilterParams::UnsharpMask { amount: 150.0, radius: 64.0, threshold: 0.0 };
    s.fill_rect(Rect::new(0, 0, 1, 1), &[f32::NAN, 0.5, 0.5, 1.0]);
    assert!(apply(&s, &p, area, None, Some(area), &Interrupt::NONE).is_none());
    s.fill_rect(area, &[0.5, 0.5, 0.5, 1e-8]);
    assert!(apply(&s, &p, area, None, Some(area), &Interrupt::NONE).is_none());
    s.fill_rect(area, &[0.5, 0.5, 0.5, -0.1]);
    assert!(apply(&s, &p, area, None, Some(area), &Interrupt::NONE).is_none());
}

#[test]
fn fft_cancellation_discards_partial_work_and_bounds_scratch() {
    let area = Rect::new(0, 0, 256, 256);
    let s = source(area, PixelFormat::RGBA8);
    let original = s.to_interleaved(area);
    let p = FilterParams::UnsharpMask { amount: 150.0, radius: 300.0, threshold: 0.0 };
    let flag = std::sync::atomic::AtomicBool::new(false);
    let cancel = || flag.load(Ordering::Relaxed);
    let progress = |_| flag.store(true, Ordering::Relaxed);
    assert!(apply(&s, &p, area, None, Some(area), &Interrupt::new(&cancel, &progress)).unwrap().is_none());
    assert_eq!(s.to_interleaved(area), original);
    flag.store(false, Ordering::Relaxed);
    let after_writes = |value| {
        if value > 0.55 {
            flag.store(true, Ordering::Relaxed);
        }
    };
    assert!(apply(&s, &p, area, None, Some(area), &Interrupt::new(&cancel, &after_writes)).unwrap().is_none());
    assert_eq!(s.to_interleaved(area), original);
    assert_eq!(group_size(MAX_BYTES, 1, 1, 1), None);
    assert_eq!(group_size(0, usize::MAX, 1, 1), None);
    assert!(Plan::new(usize::MAX, &[1.0], &mut FftPlanner::new()).is_none());
    assert!(buffer::<f32>(usize::MAX, 0.0).is_none());
}

#[test]
fn positive_threshold_dispatch_retains_exact_pixel_decisions() {
    let area = Rect::new(-3, 2, 253, 258);
    for depth in SampleType::ALL {
        let s = source(area, PixelFormat::new(ColorMode::Rgb, depth, true));
        let p = FilterParams::UnsharpMask { amount: 500.0, radius: 64.0, threshold: 8.0 };
        let expected = crate::sharpen::tests::apply_reference(&s, &p, area, area, None, Some(area), 256);
        let actual = crate::apply_in(&s, &p, area, area, None, area);
        assert_eq!(expected.to_interleaved(area), actual.to_interleaved(area));
    }
}

#[test]
fn fft_handles_maximum_radius_and_hdr_alpha_channel_pairs() {
    let area = Rect::new(0, 0, 1, 1);
    let mut s = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
    s.fill_rect(area, &[0.5]);
    let ctx = Ctx { bounds: area, mode: ColorMode::Grayscale, alpha: false };
    let src = Image::read_clamped(&s, area.inflate(3001), area);
    let pixels = reference::unsharp(&src, area, &ctx, 500.0, 1000.0, 0.0);
    let mut expected = s.clone();
    expected.write_region(area, &pixels);
    compare(&expected, &run(&s, area, 500.0, 1000.0, None, Some(area), &Interrupt::NONE).unwrap(), area);

    let area = Rect::new(-2, 3, 11, 14);
    let mut s = Surface::new(PixelFormat::RGBA32F);
    s.fill_rect(area, &[1e6, -1e5, 1e-4, 0.02]);
    let ctx = Ctx { bounds: area, mode: ColorMode::Rgb, alpha: true };
    let src = Image::read_clamped(&s, area.inflate(193), area);
    let pixels = reference::unsharp(&src, area, &ctx, 500.0, 64.0, 0.0);
    let mut expected = s.clone();
    expected.write_region(area, &pixels);
    compare(&expected, &run(&s, area, 500.0, 64.0, None, Some(area), &Interrupt::NONE).unwrap(), area);
}
