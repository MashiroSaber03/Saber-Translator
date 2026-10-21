use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;

fn fixture(format: PixelFormat, rect: Rect, hdr: bool) -> Surface {
    let mut surface = Surface::new(format);
    for tile in rect.tiles() {
        let area = tile.rect().intersect(&rect);
        let mut values = Vec::with_capacity(area.width() as usize * area.height() as usize * format.channels());
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                let seed = (x as u32).wrapping_mul(0x9e37_79b9) ^ (y as u32).wrapping_mul(0x85eb_ca6b);
                for channel in 0..format.channels() {
                    let level = seed.rotate_left(channel as u32 * 7).wrapping_add(channel as u32 * 71) % 256;
                    let alpha = format.alpha && channel + 1 == format.channels();
                    let value = if alpha && seed.is_multiple_of(5) { 0.0 } else { level as f32 / 255.0 };
                    values.push(if hdr && !alpha { value * 4.0 - 0.75 } else { value });
                }
            }
        }
        surface.write_region(area, &values);
    }
    surface
}

fn premultiplied_max_error(mut a: Vec<f32>, mut b: Vec<f32>, channels: usize, alpha: bool) -> f32 {
    crate::image::premultiply(&mut a, channels, alpha);
    crate::image::premultiply(&mut b, channels, alpha);
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

#[test]
fn direct_matches_legacy_depth_models_edges_centers_and_hdr() {
    let source_rect = Rect::new(-17, -13, 45, 39);
    let bounds = Rect::new(-41, -32, 51, 48);
    let out = Rect::new(-22, -19, 47, 43);
    let mut worst = 0.0f32;
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for (mode, alpha) in [
            (ColorMode::Grayscale, false),
            (ColorMode::Grayscale, true),
            (ColorMode::Rgb, false),
            (ColorMode::Rgb, true),
            (ColorMode::Cmyk, false),
            (ColorMode::Cmyk, true),
            (ColorMode::Lab, true),
        ] {
            let format = PixelFormat::new(mode, sample, alpha);
            let src = Image::read(&fixture(format, source_rect, sample == SampleType::F32), source_rect);
            let ctx = Ctx { bounds, mode, alpha };
            for method in [RadialMethod::Spin, RadialMethod::Zoom] {
                for (amount, center) in [(0.0, (0.5, 0.5)), (1.0, (0.5, 0.5)), (20.0, (0.53, -0.17)), (100.0, (0.0, 1.0))] {
                    let direct = Plan::new(bounds, amount, method, center).filter(&src, out, &ctx, &Interrupt::NONE).expect("valid input");
                    let legacy = reference(&src, out, &ctx, amount, method, center);
                    assert_eq!(direct, legacy, "preserve finite upstream samples exactly");
                    let error = premultiplied_max_error(direct, legacy, src.ch, alpha);
                    worst = worst.max(error);
                    assert!(error <= 2e-5, "{format:?}, {method:?}, amount {amount}, center {center:?}: {error}");
                }
            }
        }
    }
    eprintln!("direct vs legacy: maximum premultiplied float error {worst}");
}

#[test]
fn direct_generic_channels_and_unusual_float_alpha_match_legacy() {
    let rect = Rect::new(-4, -3, 21, 16);
    for channels in [1, 2, 4, 5, 7, 12] {
        let mut src = Image::new(rect, channels);
        for (index, pixel) in src.data.chunks_exact_mut(channels).enumerate() {
            for (channel, value) in pixel.iter_mut().enumerate() {
                *value = ((index * 13 + channel * 3) % 37) as f32 / 9.0 - 0.75;
            }
            *pixel.last_mut().expect("nonempty pixel") = [-0.7, 0.0, 1e-9, 0.4, 1.7, f32::NAN, f32::INFINITY, f32::NEG_INFINITY][index % 8];
        }
        for alpha in [false, true] {
            let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha };
            for method in [RadialMethod::Spin, RadialMethod::Zoom] {
                let fast = Plan::new(rect, 53.0, method, (0.51, 0.47)).filter(&src, rect, &ctx, &Interrupt::NONE).expect("valid input");
                let slow = reference(&src, rect, &ctx, 53.0, method, (0.51, 0.47));
                for (a, b) in fast.iter().zip(&slow) {
                    if b.is_nan() {
                        assert!(a.is_nan(), "match legacy NaN classification");
                    } else if b.is_infinite() {
                        assert_eq!(a, b, "match legacy infinity sign");
                    } else {
                        assert_eq!(a, b, "finite upstream output is preserved exactly");
                    }
                }
                let error = fast.iter().zip(&slow).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max);
                assert!(error < 2e-5, "channels {channels}, alpha {alpha}, {method:?}: relative error {error}");
            }
        }
    }
}

#[test]
fn direct_invalid_storage_extreme_coordinates_and_cancellation_are_bounded() {
    let rect = Rect::new(0, 0, 23, 19);
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    let plan = Plan::new(rect, 20.0, RadialMethod::Spin, (0.5, 0.5));
    let mut src = Image::new(rect, 4);
    let cancel = || true;
    assert!(plan.filter(&src, rect, &ctx, &Interrupt::cancel_only(&cancel)).is_none());
    src.data.pop();
    assert!(plan.filter(&src, rect, &ctx, &Interrupt::NONE).is_none());
    src.ch = 0;
    assert!(plan.filter(&src, rect, &ctx, &Interrupt::NONE).is_none());
    let src = Image::new(Rect::EMPTY, 4);
    assert!(plan.filter(&src, Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX), &ctx, &Interrupt::NONE).is_none());
    assert!(Plan::new(rect, f32::NAN, RadialMethod::Spin, (0.5, 0.5)).filter(&src, rect, &ctx, &Interrupt::NONE).is_none());
    assert!(Plan::new(rect, 20.0, RadialMethod::Spin, (f32::INFINITY, 0.5)).filter(&src, rect, &ctx, &Interrupt::NONE).is_none());
    assert_eq!(plan.filter(&src, Rect::EMPTY, &ctx, &Interrupt::NONE), Some(Vec::new()));
    let edge = Rect::new(i32::MAX - 2, i32::MAX - 2, i32::MAX, i32::MAX);
    let src = Image::new(edge, 4);
    assert!(Plan::new(edge, 100.0, RadialMethod::Spin, (-100.0, 100.0)).filter(&src, edge, &ctx, &Interrupt::NONE).is_some());
}

#[test]
fn direct_observes_cancellation_inside_an_output_tile() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let rect = Rect::new(0, 0, 16, 24);
    let src = Image::new(rect, 4);
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    let calls = AtomicUsize::new(0);
    let cancel = || calls.fetch_add(1, Ordering::Relaxed) >= 2;
    assert!(Plan::new(rect, 100.0, RadialMethod::Spin, (0.5, 0.5)).filter(&src, rect, &ctx, &Interrupt::cancel_only(&cancel)).is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 3, "entry, first row, eighth row");
}

#[test]
fn direct_apply_preserves_selection_and_canvas_edges_at_all_depths() {
    let rect = Rect::new(0, 0, 37, 29);
    let bounds = Rect::new(3, 5, 31, 23);
    let selection = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false), &[0.37]);
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let format = PixelFormat::new(ColorMode::Rgb, sample, true);
        let src = fixture(format, rect, sample == SampleType::F32);
        for method in [RadialMethod::Spin, RadialMethod::Zoom] {
            let params = crate::FilterParams::RadialBlur { quality: crate::RadialQuality::Draft, amount: 100.0, method, center_x: 0.2, center_y: 0.8 };
            let fast = crate::apply_in_with(&src, &params, rect, bounds, Some(&selection), rect, &Interrupt::NONE).expect("direct input");
            let legacy = crate::apply_radial_reference(&src, &params, rect, bounds, Some(&selection), rect, &Interrupt::NONE).expect("valid input");
            assert_eq!(fast.read_region(rect), legacy.read_region(rect));
            let error = premultiplied_max_error(fast.read_region(rect), legacy.read_region(rect), 4, true);
            let tolerance = match sample {
                SampleType::U8 => 1.0 / 255.0,
                SampleType::U16 => 1.0 / 65535.0,
                _ => 0.0,
            } + 2e-5;
            assert!(error <= tolerance, "{sample:?}, {method:?}: {error}");
        }
    }
}

#[test]
fn direct_preserves_upstream_quality_ceilings_and_discrete_samples() {
    let bounds = Rect::new(0, 0, 1000, 1000);
    let out = Rect::new(999, 500, 1000, 501);
    let mut source = Image::new(bounds, 1);
    for (i, v) in source.data.iter_mut().enumerate() {
        *v = ((i * 37 + i / 1000 * 13) % 101) as f32 / 100.0;
    }
    let ctx = Ctx { bounds, mode: ColorMode::Grayscale, alpha: false };
    for (quality, limit) in [(crate::RadialQuality::Draft, 64), (crate::RadialQuality::Good, 256), (crate::RadialQuality::Best, 4096)] {
        for method in [RadialMethod::Spin, RadialMethod::Zoom] {
            let plan = Plan::with_quality(bounds, 100.0, method, quality, (0.5, 0.5), &Interrupt::NONE).expect("quality plan");
            assert_eq!(plan.max_intervals, limit);
            assert_eq!(plan.table(10000.0, 0.0).expect("capped table").len(), limit + 1);
            let actual = plan.filter(&source, out, &ctx, &Interrupt::NONE).expect("direct output");
            let expected = super::super::average_samples(&source, out, &ctx, |x, y, samples| {
                let (dx, dy) = (x - 500.0, y - 500.0);
                let extent = match method {
                    RadialMethod::Spin => 100.0f32.to_radians(),
                    RadialMethod::Zoom => 0.5,
                };
                let intervals = (extent * (dx * dx + dy * dy).sqrt()).ceil().clamp(1.0, limit as f32) as usize;
                for i in 0..=intervals {
                    samples.push(match method {
                        RadialMethod::Spin => {
                            let (sin, cos) = ((i as f32 / intervals as f32 - 0.5) * extent).sin_cos();
                            (500.0 + dx * cos - dy * sin, 500.0 + dx * sin + dy * cos)
                        }
                        RadialMethod::Zoom => {
                            let k = 1.0 - extent * i as f32 / intervals as f32;
                            (500.0 + dx * k, 500.0 + dy * k)
                        }
                    });
                }
            });
            assert_eq!(actual, expected, "{quality:?}, {method:?}: preserve discrete positions");
        }
    }
}

#[test]
fn quality_plan_cancellation_stops_table_preparation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let cancel = || calls.fetch_add(1, Ordering::Relaxed) >= 2;
    assert!(
        Plan::with_quality(Rect::new(0, 0, 100, 100), 100.0, RadialMethod::Spin, crate::RadialQuality::Best, (0.5, 0.5), &Interrupt::cancel_only(&cancel))
            .is_none()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn all_qualities_preserve_samples_at_large_radii_across_models_and_depths() {
    let bounds = Rect::new(0, 0, 6000, 4000);
    let source_rect = Rect::new(5970, 1980, 6000, 2020);
    let out = Rect::new(5990, 2000, 5992, 2002);
    for quality in [crate::RadialQuality::Draft, crate::RadialQuality::Good, crate::RadialQuality::Best] {
        for method in [RadialMethod::Spin, RadialMethod::Zoom] {
            let plan = Plan::with_quality(bounds, 100.0, method, quality, (0.5, 0.5), &Interrupt::NONE).unwrap();
            for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
                for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
                    for alpha in [false, true] {
                        let format = PixelFormat::new(mode, sample, alpha);
                        let src = Image::read(&fixture(format, source_rect, sample == SampleType::F32), source_rect);
                        let ctx = Ctx { bounds, mode, alpha };
                        let actual = plan.filter(&src, out, &ctx, &Interrupt::NONE).unwrap();
                        let expected = super::super::radial(&src, out, &ctx, 100.0, method, quality, (0.5, 0.5));
                        assert_eq!(actual, expected, "{quality:?} {method:?} {format:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn unavailable_optional_plan_uses_upstream_instead_of_cancellation() {
    let rect = Rect::new(0, 0, 7, 5);
    let source = fixture(PixelFormat::RGBA8, rect, false);
    // Nonfinite centers decline preparation. The unchanged upstream sampler is
    // still used; a declined optimization must not look like engine cancellation.
    let params =
        crate::FilterParams::RadialBlur { amount: 20.0, method: RadialMethod::Spin, quality: crate::RadialQuality::Good, center_x: f32::NAN, center_y: 0.5 };
    let actual = crate::apply_in_with(&source, &params, rect, rect, None, rect, &Interrupt::NONE).unwrap();
    let expected = crate::apply_radial_reference(&source, &params, rect, rect, None, rect, &Interrupt::NONE).unwrap();
    assert_eq!(actual.read_region(rect), expected.read_region(rect));
}

#[test]
fn small_best_jobs_prepare_only_reachable_tables() {
    let rect = Rect::new(-30, -20, 30, 20);
    let format = PixelFormat::RGBA32F;
    let source = fixture(format, rect, true);
    let ctx = Ctx { bounds: rect, mode: format.mode, alpha: true };
    let image = Image::read(&source, rect);
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let plan = Plan::for_area(rect, 100.0, method, crate::RadialQuality::Best, (0.5, 0.5), rect, &Interrupt::NONE).unwrap();
        assert!(plan.max_intervals < 64, "small Best image does not need 4096 intervals");
        assert!(plan.tables.len() < 2200);
        let actual = plan.filter(&image, rect, &ctx, &Interrupt::NONE).unwrap();
        let expected = super::super::radial(&image, rect, &ctx, 100.0, method, crate::RadialQuality::Best, (0.5, 0.5));
        assert_eq!(actual, expected);
    }
}
