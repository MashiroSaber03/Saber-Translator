use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::{Interrupt, Surface};

#[path = "tests_tilt_shift_reference.rs"]
mod reference;

fn image(rect: Rect, ch: usize) -> Image {
    let data = (rect.y0..rect.y1)
        .flat_map(|y| {
            (rect.x0..rect.x1).flat_map(move |x| {
                (0..ch).map(move |c| match c {
                    0 => (x * 17 + y * 31).rem_euclid(113) as f32 / 37.0 - 0.5,
                    1 => {
                        if (x + y).rem_euclid(3) == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => ((x ^ y).unsigned_abs() as usize + c) as f32 % 23.0 / 22.0,
                })
            })
        })
        .collect();
    Image { rect, ch, data }
}

fn params(blur: f32, centre: (f32, f32), angle: f32, focus: f32, transition: f32) -> crate::FilterParams {
    crate::FilterParams::TiltShift { blur, center_x: centre.0, center_y: centre.1, angle, focus, transition }
}

fn check(src: &Image, out: Rect, ctx: &Ctx, p: &crate::FilterParams) {
    let crate::FilterParams::TiltShift { blur, center_x, center_y, angle, focus, transition } = *p else { panic!("Tilt-Shift fixture") };
    let want = reference::tilt_shift(src, out, ctx, blur, (center_x, center_y), angle, focus, transition);
    let got = tilt_shift(src, out, ctx, blur, (center_x, center_y), angle, focus, transition);
    assert_eq!(got.len(), want.len());
    for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
        assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()), "sample {i}: {got} != {want}, delta {}", (got - want).abs());
    }
}

#[test]
fn tilt_shift_matches_frozen_kernel_at_band_boundaries_and_edges() {
    let bounds = Rect::new(-19, 7, 54, 66);
    let mut cases = Vec::new();
    for blur in [0.0, 0.49, 0.5, 1.0, 15.0, 80.0, 500.0] {
        for angle in [-90.0, -71.0, 0.0, 27.0, 90.0, 180.0] {
            cases.push(params(blur, (0.18, 0.91), angle, 0.03, 0.01));
            cases.push(params(blur, (0.5, 0.5), angle, 0.1, 0.6));
        }
    }
    cases.push(params(80.0, (0.5, 0.5), 0.0, 5.0, 0.1));
    cases.push(params(80.0, (-5.0, -5.0), 27.0, 0.1, 0.15));
    for ch in 1..=6 {
        let src = image(bounds.inflate(12), ch);
        for alpha in [false, true] {
            let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha };
            for p in &cases {
                check(&src, bounds, &ctx, p);
                check(&src, Rect::new(-7, 17, 15, 39), &ctx, p);
            }
        }
    }
}

#[test]
fn tilt_shift_preserves_degenerate_and_nonfinite_parameter_behavior() {
    for bounds in [Rect::EMPTY, Rect::new(1, -1, 2, 0), Rect::new(-5, 3, 14, 4)] {
        let src = image(bounds, 4);
        let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
        for value in [0.0, -1.0, f32::MIN, f32::MAX, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for field in 0..6 {
                // The old NaN blur amount panicked in the sigma clamp. Its
                // corrected identity behavior has a separate regression below.
                if field == 0 && value.is_nan() {
                    continue;
                }
                let mut p = params(15.0, (0.5, 0.5), 0.0, 0.1, 0.15);
                if let crate::FilterParams::TiltShift { blur, center_x, center_y, angle, focus, transition } = &mut p {
                    match field {
                        0 => *blur = value,
                        1 => *center_x = value,
                        2 => *center_y = value,
                        3 => *angle = value,
                        4 => *focus = value,
                        _ => *transition = value,
                    }
                }
                check(&src, bounds, &ctx, &p);
            }
        }
    }
    let bounds = Rect::new(0, 0, 3, 5);
    let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
    assert!(crate::kernel(&params(15.0, (0.5, 0.5), 0.0, 0.1, 0.15), &Image { rect: bounds, ch: 4, data: vec![] }, bounds, &ctx).is_empty());
    assert!(crate::kernel(&params(15.0, (0.5, 0.5), 0.0, 0.1, 0.15), &Image { rect: bounds, ch: 0, data: vec![] }, bounds, &ctx).is_empty());
}

#[test]
fn tilt_shift_nan_blur_is_an_identity_instead_of_panicking() {
    let bounds = Rect::new(-2, 3, 9, 14);
    let src = image(bounds, 4);
    let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
    let p = params(f32::NAN, (0.5, 0.5), 0.0, 0.1, 0.15);
    assert_eq!(crate::kernel(&p, &src, bounds, &ctx), src.data);
    let mut surface = Surface::new(PixelFormat::RGBA32F);
    surface.write_region(bounds, &src.data);
    assert_eq!(crate::apply_in(&surface, &p, bounds, bounds, None, bounds), surface);
    // Prove this synthetic input panicked in the unchanged starting kernel.
    assert!(std::panic::catch_unwind(|| reference::tilt_shift(&src, bounds, &ctx, f32::NAN, (0.5, 0.5), 0.0, 0.1, 0.15)).is_err());
}

#[test]
fn tilt_shift_keeps_running_sum_order_for_extreme_hdr_and_signed_zero() {
    let bounds = Rect::new(-13, -5, 24, 28);
    let mut src = image(bounds.inflate(15), 4);
    let values = [0.0, -0.0, 1e30, -1e30, 1e-30, -1e-30, 0.8];
    for (i, px) in src.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        px[0] = values[i % values.len()];
        px[1] = values[(i + 3) % values.len()];
        px[3] = [0.0, 1e-8, 0.5, 1.0][i % 4];
    }
    let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
    for blur in [0.5, 15.0, 80.0, 300.0] {
        for angle in [0.0, 37.0, -90.0] {
            let p = params(blur, (0.5, 0.5), angle, 0.1, 0.15);
            check(&src, bounds, &ctx, &p);
            check(&src, Rect::new(0, 0, 10, 17), &ctx, &p);
        }
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = src.clone();
        invalid.data[19] = value;
        invalid.data[47] = value;
        check(&invalid, bounds, &ctx, &params(15.0, (0.5, 0.5), 0.0, 0.1, 0.15));
    }
}

#[test]
fn tilt_shift_depths_models_selection_and_tile_edges_match_reference() {
    let bounds = Rect::new(-17, 11, 96, 84);
    let area = Rect::new(-9, 17, 79, 77);
    let p = params(17.0, (0.3, 0.7), 29.0, 0.07, 0.3);
    let mut selection = Surface::new(PixelFormat::GRAY8);
    selection.fill_rect(bounds, &[0.4]);
    selection.fill_rect(Rect::new(3, 19, 35, 44), &[1.0]);
    selection.fill_rect(Rect::new(10, 20, 20, 30), &[0.0]);
    for mode in [
        ColorMode::Rgb,
        ColorMode::Grayscale,
        ColorMode::Cmyk,
        ColorMode::Lab,
        ColorMode::Bitmap,
        ColorMode::Indexed,
        ColorMode::Duotone,
        ColorMode::Multichannel,
    ] {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let fmt = PixelFormat::new(mode, sample, alpha);
                let mut surface = Surface::new(fmt);
                surface.write_region(bounds, &image(bounds, fmt.channels()).data);
                let original = surface.clone();
                let ctx = Ctx { bounds, mode, alpha };
                let radius = reach(17.0).ceil() as i32 + 1;
                assert_eq!(p.halo_for(bounds), crate::Halo::Radius(radius));
                for extent in [None, Some(bounds)] {
                    for tile in [16, 47, 128] {
                        let got = crate::apply_tiled(&surface, &p, area, bounds, Some(&selection), tile, extent);
                        let mut want = surface.clone();
                        for y in (area.y0..area.y1).step_by(tile as usize) {
                            for x in (area.x0..area.x1).step_by(tile as usize) {
                                let t = Rect::new(x, y, (x + tile).min(area.x1), (y + tile).min(area.y1));
                                let src = match extent {
                                    Some(e) => Image::read_clamped(&surface, t.inflate(radius), e),
                                    None => Image::read(&surface, t.inflate(radius)),
                                };
                                let mut data = reference::tilt_shift(&src, t, &ctx, 17.0, (0.3, 0.7), 29.0, 0.07, 0.3);
                                crate::mix_selection(&mut data, t, &selection, &src, alpha);
                                want.write_region(t, &data);
                            }
                        }
                        assert_eq!(got.read_region(bounds), want.read_region(bounds), "{fmt:?}, tile {tile}, extent {extent:?}");
                    }
                }
                assert_eq!(surface, original);
            }
        }
    }
}

#[test]
fn tilt_shift_cancellation_preserves_source() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let bounds = Rect::new(0, 0, 512, 256);
    let mut source = Surface::new(PixelFormat::RGBA8);
    source.write_region(bounds, &image(bounds, 4).data);
    let original = source.clone();
    let p = params(80.0, (0.5, 0.5), 27.0, 0.1, 0.15);
    assert!(crate::apply_in_with(&source, &p, bounds, bounds, None, bounds, &Interrupt::cancel_only(&|| true)).is_none());
    let cancelled = AtomicBool::new(false);
    let cancel = || cancelled.load(Ordering::Relaxed);
    let progress = |p| {
        if p > 0.0 {
            cancelled.store(true, Ordering::Relaxed);
        }
    };
    let ctl = Interrupt::new(&cancel, &progress);
    assert!(crate::apply_in_with(&source, &p, bounds, bounds, None, bounds, &ctl).is_none());
    assert!(cancelled.load(Ordering::Relaxed));
    assert_eq!(source, original);
    let fresh = crate::apply_in_with(&source, &p, bounds, bounds, None, bounds, &Interrupt::NONE);
    assert_eq!(fresh, Some(crate::apply_in(&source, &p, bounds, bounds, None, bounds)));
}
