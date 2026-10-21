use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::{Interrupt, Surface};

#[path = "tests_iris_reference.rs"]
mod reference;

fn image(rect: Rect, ch: usize) -> Image {
    let data = (rect.y0..rect.y1)
        .flat_map(|y| {
            (rect.x0..rect.x1).flat_map(move |x| {
                (0..ch).map(move |c| match c {
                    0 => ((x * 17 + y * 31).rem_euclid(113)) as f32 / 37.0 - 0.5,
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

fn check(src: &Image, out: Rect, ctx: &Ctx, pins: &[IrisPin]) {
    let want = reference::iris(src, out, ctx, pins);
    let got = iris(src, out, ctx, pins);
    assert_eq!(got.len(), want.len());
    for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
        assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()), "sample {i}: {got} != {want}, delta {}", (got - want).abs());
    }
}

#[test]
fn iris_matches_frozen_kernel_at_transitions_and_edges() {
    let bounds = Rect::new(-19, 7, 54, 66);
    let mut cases = vec![vec![], vec![IrisPin::default()]];
    for blur in [0.0, 0.49, 0.5, 1.0, 15.0, 80.0, 500.0] {
        for roundness in [0.0, 0.01, 37.0, 100.0] {
            cases.push(vec![IrisPin { x: 0.18, y: 0.91, radius_x: 0.45, radius_y: 0.13, angle: 71.0, feather: 0.998, roundness, blur }]);
        }
    }
    cases.push(vec![
        IrisPin { x: 0.2, blur: 30.0, angle: -43.0, ..IrisPin::default() },
        IrisPin { x: 0.7, y: 0.1, blur: 11.0, roundness: 80.0, ..IrisPin::default() },
    ]);
    cases.push(vec![IrisPin { radius_x: 8.0, radius_y: 9.0, ..IrisPin::default() }]);
    cases.push(vec![IrisPin { x: -5.0, ..IrisPin::default() }]);
    for ch in 1..=6 {
        let src = image(bounds.inflate(12), ch);
        for alpha in [false, true] {
            let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha };
            for pins in &cases {
                check(&src, bounds, &ctx, pins);
                check(&src, Rect::new(-7, 17, 15, 39), &ctx, pins);
            }
        }
    }
}

#[test]
fn iris_preserves_degenerate_and_nonfinite_pin_behavior() {
    for bounds in [Rect::EMPTY, Rect::new(1, -1, 2, 0), Rect::new(-5, 3, 14, 4)] {
        let src = image(bounds, 4);
        let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
        for value in [0.0, -1.0, f32::MIN, f32::MAX, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for field in 0..9 {
                let mut p = IrisPin::default();
                match field {
                    0 => p.x = value,
                    1 => p.y = value,
                    2 => p.radius_x = value,
                    3 => p.radius_y = value,
                    4 => p.angle = value,
                    5 => p.feather = value,
                    6 => p.blur = value,
                    7 => p.roundness = value,
                    _ => {
                        p.radius_x = value;
                        p.radius_y = value;
                    }
                }
                check(&src, bounds, &ctx, &[p]);
            }
        }
    }
}

#[test]
fn iris_keeps_running_sum_order_for_extreme_hdr_and_signed_zero() {
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
        let pins = [IrisPin { angle: 37.0, roundness: 55.0, blur, ..IrisPin::default() }];
        check(&src, bounds, &ctx, &pins);
        check(&src, Rect::new(0, 0, 10, 17), &ctx, &pins);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid_samples = src.clone();
        invalid_samples.data[19] = value;
        invalid_samples.data[47] = value;
        check(&invalid_samples, bounds, &ctx, &[IrisPin::default()]);
    }
    let pins = vec![IrisPin::default(); 4096];
    check(&src, Rect::new(0, 0, 1, 1), &ctx, &pins);
    let malformed = Image { rect: bounds, ch: 4, data: vec![] };
    assert!(iris(&malformed, bounds, &ctx, &[IrisPin::default()]).is_empty());
    let no_channels = Image { rect: bounds, ch: 0, data: vec![] };
    assert!(iris(&no_channels, bounds, &ctx, &[IrisPin::default()]).is_empty());
}

#[test]
fn iris_cancellation_after_progress_preserves_source() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let bounds = Rect::new(0, 0, 512, 256);
    let mut source = Surface::new(PixelFormat::RGBA8);
    source.write_region(bounds, &image(bounds, 4).data);
    let original = source.clone();
    let cancelled = AtomicBool::new(false);
    let cancel = || cancelled.load(Ordering::Relaxed);
    let progress = |p| {
        if p > 0.0 {
            cancelled.store(true, Ordering::Relaxed);
        }
    };
    let ctl = Interrupt::new(&cancel, &progress);
    let params = crate::FilterParams::IrisBlur { pins: vec![IrisPin::default()] };
    assert!(crate::apply_in_with(&source, &params, bounds, bounds, None, bounds, &ctl).is_none());
    assert!(cancelled.load(Ordering::Relaxed));
    assert_eq!(source, original);
}

#[test]
fn iris_depths_models_selection_tiles_and_cancellation() {
    let bounds = Rect::new(-17, 11, 96, 84);
    let area = Rect::new(-9, 17, 79, 77);
    let pins = vec![IrisPin { angle: 29.0, roundness: 35.0, blur: 17.0, ..IrisPin::default() }];
    let params = crate::FilterParams::IrisBlur { pins: pins.clone() };
    let mut selection = Surface::new(PixelFormat::GRAY8);
    selection.fill_rect(bounds, &[0.4]);
    selection.fill_rect(Rect::new(3, 19, 35, 44), &[1.0]);
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let fmt = PixelFormat::new(mode, sample, alpha);
                let mut surface = Surface::new(fmt);
                surface.write_region(bounds, &image(bounds, fmt.channels()).data);
                let ctl = Interrupt::cancel_only(&|| true);
                assert!(crate::apply_in_with(&surface, &params, area, bounds, Some(&selection), bounds, &ctl).is_none());
                let original = surface.clone();
                let ctx = Ctx { bounds, mode, alpha };
                let radius = reach(17.0).ceil() as i32 + 1;
                assert_eq!(params.halo_for(bounds), crate::Halo::Radius(radius));
                for tile in [16, 47, 128] {
                    let got = crate::apply_tiled(&surface, &params, area, bounds, Some(&selection), tile, Some(bounds));
                    let mut want = surface.clone();
                    for y in (area.y0..area.y1).step_by(tile as usize) {
                        for x in (area.x0..area.x1).step_by(tile as usize) {
                            let t = Rect::new(x, y, (x + tile).min(area.x1), (y + tile).min(area.y1));
                            let src = Image::read_clamped(&surface, t.inflate(radius), bounds);
                            let mut data = reference::iris(&src, t, &ctx, &pins);
                            crate::mix_selection(&mut data, t, &selection, &src, alpha);
                            want.write_region(t, &data);
                        }
                    }
                    assert_eq!(got.read_region(bounds), want.read_region(bounds), "{fmt:?}, tile {tile}");
                }
                assert_eq!(surface, original);
            }
        }
    }
}

#[test]
#[ignore = "release profiling; run alone with --nocapture"]
fn iris_profile_baseline_stages() {
    for blur in [15.0, 80.0, 300.0] {
        let out = Rect::new(1500, 1000, 2012, 1512);
        let bounds = Rect::new(0, 0, 6000, 4000);
        let src = image(out.inflate(reach(blur).ceil() as i32), 4);
        let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
        let pin = IrisPin { blur, angle: 27.0, roundness: 60.0, ..IrisPin::default() };
        let start = std::time::Instant::now();
        let sig: Vec<f32> = (0..out.width() as usize * out.height() as usize)
            .map(|i| {
                let (x, y) = xy(out, i);
                let (e, _, _) = pin_distance(x as f32 + 0.5, y as f32 + 0.5, bounds, pin.x, pin.y, pin.radius_x, pin.radius_y, pin.angle, 5.6);
                sigma_of(blur) * smoothstep(pin.feather, 1.0, e)
            })
            .collect();
        println!("blur {blur}: geometry {:?}", start.elapsed());
        std::hint::black_box(sig);
        let start = std::time::Instant::now();
        let p = premul_window(&src, src.rect, ctx.alpha);
        println!("blur {blur}: premul {:?}", start.elapsed());
        for k in 0..LEVELS {
            let mut buf = p.clone();
            let sigma = sigma_of(blur) / 2f32.powi((LEVELS - 1 - k) as i32);
            let start = std::time::Instant::now();
            gauss_blur_n(&mut buf, src.rect.width() as usize, src.rect.height() as usize, src.ch, sigma);
            println!("blur {blur}: level {sigma} {:?}", start.elapsed());
            std::hint::black_box(buf);
        }
        let start = std::time::Instant::now();
        std::hint::black_box(reference::iris(&src, out, &ctx, std::slice::from_ref(&pin)));
        println!("blur {blur}: reference whole kernel {:?}", start.elapsed());
        let start = std::time::Instant::now();
        std::hint::black_box(iris(&src, out, &ctx, &[pin]));
        println!("blur {blur}: candidate whole kernel {:?}", start.elapsed());
    }
}
