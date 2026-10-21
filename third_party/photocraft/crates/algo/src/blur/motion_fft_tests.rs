//! Numerical and geometric oracles for convolution of the finite motion kernel.
//! The reference independently builds the current upstream merged f32 taps and
//! source premultiplication, then performs convolution and accumulation in f64.

use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;
use std::collections::BTreeMap;

fn offsets(angle: f32, distance: f32) -> Vec<(f64, f64)> {
    let distance = distance.abs();
    let steps = distance.ceil() as i32;
    let (sin, cos) = angle.to_radians().sin_cos();
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32 - 0.5;
            (f64::from(cos * distance * t), -f64::from(sin * distance * t))
        })
        .collect()
}

fn merged_taps(angle: f32, distance: f32) -> BTreeMap<(i32, i32), f32> {
    let offsets = offsets(angle, distance);
    let norm = 1.0 / offsets.len() as f32;
    let mut taps = BTreeMap::new();
    for (x, y) in offsets {
        let (x, y) = (x as f32, y as f32);
        let (x0, y0) = (x.floor(), y.floor());
        let (ax, ay) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        for (dx, dy, weight) in [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)] {
            if weight > 0.0 {
                *taps.entry((x0 + dx, y0 + dy)).or_insert(0.0) += weight * norm;
            }
        }
    }
    taps
}

fn ideal(src: &Image, out: Rect, alpha: bool, angle: f32, distance: f32) -> Vec<f64> {
    let taps = merged_taps(angle, distance);
    let mut result = Vec::new();
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            let mut acc = vec![0.0; src.ch];
            for (&(dx, dy), &weight) in &taps {
                let (sx, sy) = (x + dx, y + dy);
                let a = if alpha { src.get(sx, sy, src.ch - 1) } else { 1.0 };
                for (c, value) in acc.iter_mut().enumerate() {
                    let sample = src.get(sx, sy, c);
                    let premultiplied = if alpha && c + 1 < src.ch { sample * a } else { sample };
                    *value += f64::from(premultiplied) * f64::from(weight);
                }
            }
            if alpha {
                let a = acc[src.ch - 1];
                for value in acc.iter_mut().take(src.ch - 1) {
                    *value = if a as f32 > 1e-7 { *value / a } else { 0.0 };
                }
            }
            result.extend(acc);
        }
    }
    result
}

fn fft(src: &Image, out: Rect, alpha: bool, angle: f32, distance: f32) -> Vec<f32> {
    super::motion_fft::filter(src, out, alpha, angle, distance).expect("supported finite motion convolution")
}

fn assert_ideal(actual: &[f32], expected: &[f64], case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}");
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let rounded = expected as f32;
        // The public result is f32; allow two rounding units and a small absolute
        // FFT residual at exact zero, rather than claiming sub-ULP precision.
        let tolerance = 1e-9 + 2.0 * f64::from(f32::EPSILON) * f64::from(rounded.abs());
        assert!((f64::from(actual) - f64::from(rounded)).abs() <= tolerance, "{case}, sample {i}: actual={actual}, ideal={expected}, tolerance={tolerance}");
    }
}

fn pattern(rect: Rect, ch: usize, alpha: bool) -> Image {
    let mut data = Vec::new();
    for y in 0..rect.height() as usize {
        for x in 0..rect.width() as usize {
            for c in 0..ch {
                let value = if alpha && c + 1 == ch { [0.0, 0.25, 0.5, 1.0][(x + y * 3) % 4] } else { ((x * 31 + y * 17 + c * 43) % 251) as f32 / 250.0 };
                data.push(value);
            }
        }
    }
    Image { rect, ch, data }
}

#[test]
fn motion_fft_matches_independent_ideal_for_channels_angles_and_distances() {
    let rect = Rect::new(-4, -4, 5, 5);
    let out = Rect::new(-1, -1, 2, 2);
    for ch in 1..=5 {
        for alpha in [false, true] {
            let src = pattern(rect, ch, alpha);
            for (angle, distance) in [(0.0, 64.0), (90.0, 65.5), (30.0, 127.0), (-45.0, -256.0), (89.75, 2000.0), (37.0, 2000.0)] {
                let actual = fft(&src, out, alpha, angle, distance);
                assert_ideal(&actual, &ideal(&src, out, alpha, angle, distance), &format!("ch={ch}, alpha={alpha}, angle={angle}, distance={distance}"));
            }
        }
    }
}

fn impulse_kernel(angle: f32, distance: f32) -> BTreeMap<(i32, i32), f64> {
    merged_taps(angle, distance).into_iter().map(|((x, y), weight)| ((-x, -y), f64::from(weight))).collect()
}

fn moments(samples: impl Iterator<Item = ((i32, i32), f64)>, angle: f32) -> [f64; 5] {
    let angle = f64::from(angle).to_radians();
    let (sin, cos) = angle.sin_cos();
    let mut result = [0.0; 5];
    for ((x, y), weight) in samples {
        let (x, y) = (f64::from(x), f64::from(y));
        result[0] += weight;
        result[1] += x * weight;
        result[2] += y * weight;
        result[3] += (x * sin + y * cos).powi(2) * weight;
        result[4] += (x * cos - y * sin).powi(2) * weight;
    }
    result
}

#[test]
fn motion_fft_impulse_preserves_mass_centroid_and_directional_width() {
    let src = Image { rect: Rect::new(0, 0, 1, 1), ch: 1, data: vec![1.0] };
    for distance in [64.0_f32, 127.0] {
        let reach = (distance / 2.0).ceil() as i32 + 2;
        let out = Rect::new(-reach, -reach, reach + 1, reach + 1);
        for angle in [0.0, 30.0, 45.0, 89.75] {
            let expected = impulse_kernel(angle, distance);
            let actual = fft(&src, out, false, angle, distance);
            let width = out.width() as usize;
            let samples = actual.iter().enumerate().map(|(i, &v)| ((out.x0 + (i % width) as i32, out.y0 + (i / width) as i32), f64::from(v)));
            let actual_moments = moments(samples, angle);
            let expected_moments = moments(expected.iter().map(|(&p, &v)| (p, v)), angle);
            assert!((actual_moments[0] - 1.0).abs() <= 3e-6, "angle={angle}, distance={distance}, mass={}", actual_moments[0]);
            for i in 1..=3 {
                assert!(
                    (actual_moments[i] - expected_moments[i]).abs() <= 2e-5,
                    "angle={angle}, distance={distance}, moment={i}, actual={}, expected={}",
                    actual_moments[i],
                    expected_moments[i]
                );
            }
            assert!((actual_moments[4] - expected_moments[4]).abs() <= 5e-4, "along-motion variance changed: angle={angle}, distance={distance}");
            let expected_pixels: Vec<f64> =
                (out.y0..out.y1).flat_map(|y| (out.x0..out.x1).map(move |x| (x, y))).map(|p| expected.get(&p).copied().unwrap_or(0.0)).collect();
            assert_ideal(&actual, &expected_pixels, &format!("impulse angle={angle}, distance={distance}"));
        }
    }
}

#[test]
fn motion_fft_sharp_stripes_match_uniform_streak_without_cross_direction_softening() {
    let rect = Rect::new(-80, -80, 81, 81);
    let out = Rect::new(-2, -2, 3, 3);
    for angle in [0.0_f32, 30.0, 45.0] {
        let (sin, cos) = f64::from(angle).to_radians().sin_cos();
        for perpendicular in [false, true] {
            for frequency in [0.125, 0.375] {
                let mut src = Image::new(rect, 1);
                for (i, value) in src.data.iter_mut().enumerate() {
                    let (x, y) = (rect.x0 + (i % rect.width() as usize) as i32, rect.y0 + (i / rect.width() as usize) as i32);
                    let coordinate = if perpendicular { f64::from(x) * sin + f64::from(y) * cos } else { f64::from(x) * cos - f64::from(y) * sin };
                    *value = (0.5 + 0.5 * (std::f64::consts::TAU * frequency * coordinate).cos()) as f32;
                }
                let distance = if perpendicular { 127.0 } else { 64.0 };
                assert_ideal(
                    &fft(&src, out, false, angle, distance),
                    &ideal(&src, out, false, angle, distance),
                    &format!("stripe angle={angle}, perpendicular={perpendicular}, frequency={frequency}"),
                );
            }
        }
    }
}

#[test]
fn motion_fft_premultiplied_alpha_hidden_colour_hdr_and_cutoff_are_preserved() {
    let rect = Rect::new(-5, -5, 6, 6);
    let out = Rect::new(-1, -1, 2, 2);
    let mut src = pattern(rect, 4, true);
    let mut hidden = src.clone();
    for (pixel, changed) in src.data.as_chunks_mut::<4>().0.iter_mut().zip(hidden.data.as_chunks_mut::<4>().0.iter_mut()) {
        for value in pixel.iter_mut().take(3) {
            *value = *value * 32.0 - 8.0;
        }
        changed.copy_from_slice(pixel);
        if pixel[3] == 0.0 {
            changed[..3].copy_from_slice(&[1e6, 7e5, 9e5]);
        }
    }
    let actual = fft(&src, out, true, 38.0, 67.75);
    assert_ideal(&actual, &ideal(&src, out, true, 38.0, 67.75), "HDR premultiplied image");
    assert_ideal(&fft(&hidden, out, true, 38.0, 67.75), &actual.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), "hidden transparent colours do not bleed");

    let rect = Rect::new(-40, -40, 41, 41);
    for alpha in [0.0_f32, 1e-8, 1e-7, f32::from_bits(1e-7_f32.to_bits() + 1), 0.25, 1.0] {
        let src = Image { rect, ch: 4, data: [0.25, 0.5, 0.75, alpha].repeat((rect.width() * rect.height()) as usize) };
        if (0.99e-7..=1.01e-7).contains(&alpha) {
            // The upstream row f32 sum can move alpha across its discontinuous
            // colour cutoff. Near that boundary, retain the upstream row kernel.
            let actual = fft(&src, out, true, 30.0, 64.0);
            let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
            let expected = motion_rows(&src, out, &ctx, 30.0, 64.0);
            assert_eq!(actual, expected, "near-cutoff alpha={alpha}");
            continue;
        }
        let actual = fft(&src, out, true, 30.0, 64.0);
        assert_ideal(&actual, &ideal(&src, out, true, 30.0, 64.0), &format!("constant alpha={alpha}"));
        if alpha <= 1e-7 {
            assert!(actual.as_chunks::<4>().0.iter().all(|p| p[..3].iter().all(|&v| v == 0.0)), "alpha cutoff={alpha}");
        }
    }
}

#[test]
fn motion_fft_upstream_summation_error_is_bounded_in_supported_coordinates() {
    for origin in [-16_384, 0, 16_375] {
        let rect = Rect::new(origin, origin, origin + 9, origin + 9);
        let out = Rect::new(origin + 2, origin + 2, origin + 5, origin + 5);
        for alpha in [false, true] {
            let src = pattern(rect, 4, alpha);
            let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha };
            for (angle, distance) in [(30.0, 64.0), (89.75, 127.0)] {
                let actual = fft(&src, out, alpha, angle, distance);
                let expected = motion_rows(&src, out, &ctx, angle, distance);
                for (&a, &b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() <= 1e-3, "origin={origin}, alpha={alpha}, angle={angle}, distance={distance}, actual={a}, original={b}");
                }
            }
        }
    }
}

#[test]
fn motion_fft_large_hdr_colours_do_not_change_small_alpha_or_its_cutoff() {
    let rect = Rect::new(-40, -40, 41, 41);
    let out = Rect::new(-1, -1, 2, 2);
    for alpha in [1e-7_f32, f32::from_bits(1e-7_f32.to_bits() + 1), 0.25] {
        let src = Image { rect, ch: 4, data: [1e10, -5e9, 9e9, alpha].repeat((rect.width() * rect.height()) as usize) };
        if (0.99e-7..=1.01e-7).contains(&alpha) {
            let actual = fft(&src, out, true, 33.0, 64.0);
            let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
            assert_eq!(actual, motion_rows(&src, out, &ctx, 33.0, 64.0), "near-cutoff HDR alpha={alpha}");
            continue;
        }
        let actual = fft(&src, out, true, 33.0, 64.0);
        assert_ideal(&actual, &ideal(&src, out, true, 33.0, 64.0), &format!("large HDR, alpha={alpha}"));
        if alpha <= 1e-7 {
            assert!(actual.as_chunks::<4>().0.iter().all(|p| p[..3].iter().all(|&v| v == 0.0)), "HDR changed alpha cutoff");
        }
    }
}

#[test]
fn motion_fft_pipeline_supports_depths_models_edges_selections_and_tile_seams() {
    let rect = Rect::new(-6, 3, 7, 16);
    let mut selection = Surface::new(PixelFormat::GRAY8);
    let coverage: Vec<f32> = (0..rect.width() * rect.height()).map(|i| [0.0, 0.25, 0.5, 1.0][i as usize % 4]).collect();
    selection.write_region(rect, &coverage);
    for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let mut source = Surface::new(PixelFormat::new(mode, depth, alpha));
                let image = pattern(rect, source.channels(), alpha);
                source.write_region(rect, &image.data);
                let params = FilterParams::MotionBlur { angle: 31.0, distance: 73.25 };
                let crate::Halo::Radius(reach) = params.halo() else { panic!("local motion halo") };
                for extent in [None, Some(rect)] {
                    let src =
                        if extent.is_some() { Image::read_clamped(&source, rect.inflate(reach), rect) } else { Image::read(&source, rect.inflate(reach)) };
                    let filtered = fft(&src, rect, alpha, 31.0, 73.25);
                    for mask in [None, Some(&selection)] {
                        let mut expected_data = filtered.clone();
                        if let Some(mask) = mask {
                            crate::mix_selection(&mut expected_data, rect, mask, &src, alpha);
                        }
                        let mut expected = source.clone();
                        expected.write_region(rect, &expected_data);
                        let expected = expected.read_region(rect.inflate(2));
                        let tolerance = match depth {
                            SampleType::U8 => 1.0 / 255.0 + 1e-7,
                            SampleType::U16 => 1.0 / 65_535.0 + 1e-7,
                            SampleType::F32 => 1e-6,
                        };
                        for tile in [5, 17, 256] {
                            let actual = crate::apply_tiled(&source, &params, rect, rect, mask, tile, extent).read_region(rect.inflate(2));
                            assert_eq!(actual.len(), expected.len());
                            for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                                assert!(
                                    (a - b).abs() <= tolerance,
                                    "mode={mode:?}, depth={depth:?}, alpha={alpha}, extent={extent:?}, selection={}, tile={tile}, sample={i}, actual={a}, expected={b}",
                                    mask.is_some()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn motion_fft_rejects_unsupported_values_and_extreme_geometry() {
    let rect = Rect::new(-2, -2, 3, 3);
    let src = pattern(rect, 4, true);
    for (angle, distance) in [(f32::NAN, 64.0), (f32::INFINITY, 64.0), (30.0, f32::NAN), (30.0, f32::INFINITY), (30.0, 2001.0)] {
        assert!(super::motion_fft::filter(&src, rect, true, angle, distance).is_none());
    }
    for alpha in [-0.25, 1.25, f32::NAN, f32::INFINITY] {
        let mut image = src.clone();
        image.data[3] = alpha;
        assert!(super::motion_fft::filter(&image, rect, true, 30.0, 64.0).is_none(), "alpha={alpha}");
    }
    for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2e10] {
        let mut image = src.clone();
        image.data[0] = sample;
        assert!(super::motion_fft::filter(&image, rect, true, 30.0, 64.0).is_none(), "sample={sample}");
    }
    let malformed = Image { rect, ch: 4, data: vec![0.0] };
    assert!(super::motion_fft::filter(&malformed, rect, true, 30.0, 64.0).is_none());
    for origin in [16_385, i32::MIN] {
        let rect = Rect::new(origin, 0, origin + 5, 5);
        let image = Image { rect, ch: 1, data: vec![0.0; 25] };
        assert!(super::motion_fft::filter(&image, rect, false, 30.0, 64.0).is_none());
    }
}

#[test]
fn motion_fft_production_crosses_real_tile_seams_and_leaves_other_pixels_untouched() {
    let extent = Rect::new(-180, -145, 173, 144);
    let area = Rect::new(-170, -135, 163, 134);
    let check = extent.inflate(8);
    let params = FilterParams::MotionBlur { angle: 31.0, distance: 73.25 };
    let crate::Halo::Radius(reach) = params.halo() else { panic!("local motion halo") };
    let mut selection = Surface::new(PixelFormat::GRAY8);
    let mut coverage = Vec::new();
    for y in extent.y0..extent.y1 {
        for x in extent.x0..extent.x1 {
            let inside = Rect::new(-51, -31, 53, 34).contains(x, y);
            coverage.push(if inside { [0.0, 0.25, 0.5, 0.75, 1.0][(x + y).rem_euclid(5) as usize] } else { 0.0 });
        }
    }
    selection.write_region(extent, &coverage);
    for depth in [SampleType::U16, SampleType::F32] {
        let mut source = Surface::new(PixelFormat::new(ColorMode::Rgb, depth, true));
        source.write_region(extent, &pattern(extent, 4, true).data);
        source.fill_rect(Rect::new(-86, -61, -82, -57), &[0.125, 0.25, 0.5, 1.0]);
        let original = Image::read(&source, check);
        for clamp in [false, true] {
            let src = if clamp { Image::read_clamped(&source, area.inflate(reach), extent) } else { Image::read(&source, area.inflate(reach)) };
            let filtered = fft(&src, area, true, 31.0, 73.25);
            for mask in [None, Some(&selection)] {
                let mut expected = source.clone();
                let mut data = filtered.clone();
                if let Some(mask) = mask {
                    crate::mix_selection(&mut data, area, mask, &src, true);
                }
                expected.write_region(area, &data);
                let expected = Image::read(&expected, check);
                let tolerance = if depth == SampleType::U16 { 1.0 / 65_535.0 + 1e-7 } else { 1e-6 };
                for tile in [64, 128, 256] {
                    let result = super::motion_apply::apply_for_bench(&source, &params, area, extent, mask, clamp.then_some(extent), tile, &Interrupt::NONE)
                        .expect("the FFT specialization must actually cross these seams")
                        .expect("not cancelled");
                    let result = Image::read(&result, check);
                    for y in check.y0..check.y1 {
                        for x in check.x0..check.x1 {
                            for c in 0..4 {
                                let actual = result.get(x, y, c);
                                if !area.contains(x, y) {
                                    assert_eq!(
                                        actual.to_bits(),
                                        original.get(x, y, c).to_bits(),
                                        "write outside area: depth={depth:?}, tile={tile}, pixel=({x},{y}), channel={c}"
                                    );
                                } else {
                                    let target = expected.get(x, y, c);
                                    assert!(
                                        (actual - target).abs() <= tolerance,
                                        "seam: depth={depth:?}, clamp={clamp}, selection={}, tile={tile}, pixel=({x},{y}), channel={c}, actual={actual}, expected={target}",
                                        mask.is_some()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn motion_fft_production_unsupported_inputs_keep_upstream_row_fallback() {
    let params = FilterParams::MotionBlur { angle: 31.0, distance: 64.0 };
    let crate::Halo::Radius(reach) = params.halo() else { panic!("local motion halo") };
    for case in 0..5 {
        let origin = if case == 4 { 20_000 } else { -64 };
        let rect = Rect::new(origin, -32, origin + 128, 32);
        let mut image = pattern(rect, 4, true);
        let middle = (32 * rect.width() as usize + 64) * 4;
        match case {
            0 => image.data[middle + 3] = -0.25,
            1 => image.data[middle + 3] = 1.25,
            2 => image.data[middle] = f32::NAN,
            3 => image.data[middle] = f32::INFINITY,
            _ => {}
        }
        let mut source = Surface::new(PixelFormat::RGBA32F);
        source.write_region(rect, &image.data);
        let src = Image::read(&source, rect.inflate(reach));
        let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
        let mut expected = source.clone();
        expected.write_region(rect, &motion_rows(&src, rect, &ctx, 31.0, 64.0));
        let expected = expected.read_region(rect);
        assert!(
            super::motion_apply::apply_for_bench(&source, &params, rect, rect, None, None, 64, &Interrupt::NONE).is_none(),
            "unsupported FFT input case={case}"
        );
        let actual = crate::apply_tiled(&source, &params, rect, rect, None, 64, None).read_region(rect);
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
                "fallback case={case}, sample={i}, actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
fn motion_fft_production_near_alpha_cutoff_retains_upstream_row_summation() {
    let extent = Rect::new(-64, -64, 65, 65);
    let area = Rect::new(-32, -32, 32, 32);
    for alpha in [1e-7_f32, 0.999e-7, 1.001e-7] {
        let mut source = Surface::new(PixelFormat::RGBA32F);
        source.fill_rect(extent, &[0.25, 0.5, 0.75, alpha]);
        for distance in [200.0_f32, 500.0] {
            let params = FilterParams::MotionBlur { angle: 0.0, distance };
            let crate::Halo::Radius(reach) = params.halo() else { panic!("local motion halo") };
            let src = Image::read_clamped(&source, area.inflate(reach), extent);
            let ctx = Ctx { bounds: extent, mode: ColorMode::Rgb, alpha: true };
            let mut expected = source.clone();
            expected.write_region(area, &motion_rows(&src, area, &ctx, 0.0, distance));
            let expected = expected.to_interleaved(area);
            // FFT uses a more accurate sum, but accuracy alone cannot preserve
            // this branch: the upstream rounded alpha chooses whether colour
            // survives. Only pixels near 1e-7 use the upstream row kernel.
            let actual = super::motion_apply::apply_for_bench(&source, &params, area, extent, None, Some(extent), 64, &Interrupt::NONE)
                .expect("the FFT path must exercise its low-alpha fallback")
                .expect("not cancelled");
            assert_eq!(actual.to_interleaved(area), expected, "near-cutoff alpha={alpha}, distance={distance}");
        }
    }
}

#[test]
fn motion_fft_low_alpha_at_large_coordinates_retains_upstream_row_cutoff() {
    let rect = Rect::new(15_950, 15_998, 16_080, 16_002);
    let out = Rect::new(16_000, 16_000, 16_003, 16_001);
    let mut src = Image::new(rect, 4);
    for (i, pixel) in src.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let y = rect.y0 + (i / rect.width() as usize) as i32;
        if y == 15_999 {
            pixel.copy_from_slice(&[0.25, 0.5, 0.75, 1.0]);
        }
    }
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    let expected = motion_rows(&src, out, &ctx, 0.0005, 64.0);
    let actual = fft(&src, out, true, 0.0005, 64.0);
    // The relative upstream kernel does not inherit absolute-coordinate phase
    // rounding. Away from its true alpha cutoff, compare numerical accuracy.
    assert_ideal(&actual, &expected.into_iter().map(f64::from).collect::<Vec<_>>(), "positive low alpha at translated coordinates");
}

#[test]
fn motion_fft_scalar_row_point_matches_upstream_for_layouts_hdr_and_cutoff() {
    let rect = Rect::new(-5, -4, 6, 5);
    let out = Rect::new(-1, 0, 2, 1);
    for ch in 1..=5 {
        for alpha in [false, true] {
            let mut src = pattern(rect, ch, alpha);
            for pixel in src.data.chunks_exact_mut(ch) {
                for value in pixel.iter_mut().take(ch - usize::from(alpha)) {
                    *value = *value * 32.0 - 8.0;
                }
            }
            let ctx = Ctx { bounds: rect, mode: ColorMode::Multichannel, alpha };
            for (angle, distance) in [(30.0, 64.0), (0.0, 256.0), (43.0, 1000.0)] {
                // Read the reference's square window once for all three points.
                // The scalar path itself never allocates that window.
                let expected = motion_rows(&src, out, &ctx, angle, distance);
                for (i, x) in (out.x0..out.x1).enumerate() {
                    let actual = super::motion_fft::row_point_for_test(&src, x, out.y0, alpha, angle, distance).expect("valid scalar row point");
                    assert_eq!(actual, expected[i * ch..(i + 1) * ch], "ch={ch}, alpha={alpha}, angle={angle}, distance={distance}, x={x}");
                }
            }
        }
    }

    let rect = Rect::new(-34, -34, 35, 35);
    for ch in 1..=5 {
        for alpha in [1e-7_f32, f32::from_bits(1e-7_f32.to_bits() + 1), 0.999e-7, 1.001e-7] {
            let mut pixel = vec![1e6; ch];
            pixel[ch - 1] = alpha;
            let src = Image { rect, ch, data: pixel.repeat((rect.width() * rect.height()) as usize) };
            let ctx = Ctx { bounds: rect, mode: ColorMode::Multichannel, alpha: true };
            let expected = motion_rows(&src, Rect::new(0, 0, 1, 1), &ctx, 31.0, 64.0);
            let actual = super::motion_fft::row_point_for_test(&src, 0, 0, true, 31.0, 64.0).expect("valid cutoff scalar point");
            assert_eq!(actual, expected, "cutoff ch={ch}, alpha={alpha}");
        }
    }
}

#[test]
fn motion_fft_output_outside_extent_retains_upstream_boundary_reads() {
    let extent = Rect::new(0, 0, 128, 128);
    let area = Rect::new(0, 200, 64, 264);
    let mut source = Surface::new(PixelFormat::RGBA32F);
    source.fill_rect(extent, &[0.25, 0.5, 0.75, 1.0]);
    let params = FilterParams::MotionBlur { angle: 0.0, distance: 256.0 };
    assert!(super::motion_apply::apply(&source, &params, area, extent, None, Some(extent), 64, &Interrupt::NONE).is_none());
    let crate::Halo::Radius(reach) = params.halo() else { panic!("local motion halo") };
    let src = Image::read_clamped(&source, area.inflate(reach), extent);
    let ctx = Ctx { bounds: extent, mode: ColorMode::Rgb, alpha: true };
    let expected = motion_rows(&src, area, &ctx, 0.0, 256.0);
    let actual = crate::apply_tiled(&source, &params, area, extent, None, 64, Some(extent));
    assert_eq!(actual.read_region(area), expected);
}
