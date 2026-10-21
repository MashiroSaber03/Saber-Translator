use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::{Interrupt, Surface};
use std::time::Instant;

#[path = "reference.rs"]
pub(super) mod reference;

pub(super) fn source(rect: Rect, fmt: PixelFormat) -> Surface {
    let mut s = Surface::new(fmt);
    for y in rect.y0..rect.y1 {
        let data: Vec<f32> = (rect.x0..rect.x1)
            .flat_map(|x| {
                (0..fmt.channels()).map(move |c| {
                    if fmt.alpha && c + 1 == fmt.channels() {
                        if (x + y).rem_euclid(17) == 0 { 0.0 } else { ((x ^ y).rem_euclid(255) + 1) as f32 / 255.0 }
                    } else {
                        let value = ((x * (7 + c as i32) + y * 13).rem_euclid(257)) as f32 / 256.0;
                        if fmt.sample == SampleType::F32 { 3.0 * value - 0.5 } else { value }
                    }
                })
            })
            .collect();
        s.write_region(Rect::new(rect.x0, y, rect.x1, y + 1), &data);
    }
    s
}

// Preserve the surrounding tiled workload as well as the old convolution: halo reads,
// automatic tiles, grouping, selection mixing, writes and pruning are timed in both paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_reference(
    s: &Surface,
    p: &crate::FilterParams,
    area: Rect,
    bounds: Rect,
    sel: Option<&Surface>,
    extent: Option<Rect>,
    tile: i32,
) -> Surface {
    let crate::FilterParams::UnsharpMask { amount, radius, threshold } = *p else { panic!("unsharp params") };
    let crate::Halo::Radius(halo) = p.halo_for(bounds) else { panic!("radius halo") };
    let fmt = s.format();
    let ctx = Ctx { bounds, mode: fmt.mode, alpha: fmt.alpha };
    let tiles: Vec<_> = (area.y0..area.y1)
        .step_by(tile as usize)
        .flat_map(|y| (area.x0..area.x1).step_by(tile as usize).map(move |x| Rect::new(x, y, (x + tile).min(area.x1), (y + tile).min(area.y1))))
        .collect();
    let group = (crate::RESULT_BUDGET / ((tile as usize).pow(2) * fmt.channels() * 4)).max(workers()).max(1);
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let ctl = Interrupt::NONE;
    let run = |t: &Rect| {
        let src = match extent {
            Some(e) => Image::read_clamped(s, t.inflate(halo), e),
            None => Image::read(s, t.inflate(halo)),
        };
        let mut data = reference::unsharp(&src, *t, &ctx, amount, radius, threshold);
        if let Some(sel) = sel {
            crate::mix_selection(&mut data, *t, sel, &src, fmt.alpha);
        }
        let done = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        ctl.progress(0.98 * done as f32 / tiles.len().max(1) as f32);
        (*t, data)
    };
    let mut out = s.clone();
    for chunk in tiles.chunks(group) {
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<_> = {
            use rayon::prelude::*;
            chunk.par_iter().map(run).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let results: Vec<_> = chunk.iter().map(run).collect();
        for (t, data) in results {
            out.write_region(t, &data);
        }
    }
    out.prune();
    out
}

fn workers() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads()
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

#[test]
fn frozen_reference_matches_current_kernel() {
    let out = Rect::new(-3, 2, 16, 15);
    let src = Image { rect: out.inflate(13), ch: 4, data: (0..(out.width() + 26) * (out.height() + 26) * 4).map(|i| (i % 127) as f32 / 126.0).collect() };
    let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha: true };
    for radius in [0.1, 0.6, 1.5, 4.0] {
        assert_eq!(unsharp(&src, out, &ctx, 150.0, radius, 8.0), reference::unsharp(&src, out, &ctx, 150.0, radius, 8.0));
    }
}

fn same_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(), "sample {i}: {a:?} ({:x}) != {b:?} ({:x})", a.to_bits(), b.to_bits());
    }
}

#[test]
fn exact_kernel_across_models_depths_edges_and_thresholds() {
    let modes = [
        ColorMode::Bitmap,
        ColorMode::Grayscale,
        ColorMode::Duotone,
        ColorMode::Indexed,
        ColorMode::Rgb,
        ColorMode::Cmyk,
        ColorMode::Lab,
        ColorMode::Multichannel,
    ];
    for mode in modes {
        for depth in SampleType::ALL {
            for alpha in [false, true] {
                let fmt = PixelFormat::new(mode, depth, alpha);
                let area = Rect::new(-3, 2, 8, 7);
                let mut surface = source(area, fmt);
                if depth == SampleType::F32 {
                    // HDR, negative, signed zero, subnormal and nonfinite samples; colour is
                    // never clamped by this filter. Transparent hidden colour must survive.
                    let px: Vec<f32> = (0..fmt.channels()).map(|c| if alpha && c + 1 == fmt.channels() { 0.0 } else { -4.0 }).collect();
                    surface.fill_rect(Rect::new(0, 3, 1, 4), &px);
                    for (x, value) in [3.5, -0.0, f32::MIN_POSITIVE, f32::INFINITY, f32::NAN].into_iter().enumerate() {
                        let mut px = vec![value; fmt.channels()];
                        if alpha {
                            *px.last_mut().unwrap() = if x == 2 { 1e-8 } else { 1.0 };
                        }
                        surface.fill_rect(Rect::new(x as i32, 5, x as i32 + 1, 6), &px);
                    }
                }
                let ctx = Ctx { bounds: area, mode, alpha };
                for radius in [0.0, 0.049, 0.1, 0.6, 1.5, 4.0, 17.25] {
                    let src = Image::read_clamped(&surface, area.inflate((radius * 3.0_f32).ceil() as i32 + 1), area);
                    for (amount, threshold) in [(0.0, 0.0), (50.0, 0.0), (500.0, 8.0), (300.0, 255.0)] {
                        same_bits(&unsharp(&src, area, &ctx, amount, radius, threshold), &reference::unsharp(&src, area, &ctx, amount, radius, threshold));
                    }
                    // An incomplete halo has zero extension, even with alpha.
                    let src = Image::read(&surface, area);
                    let out = Rect::new(-5, 1, 10, 8);
                    same_bits(&unsharp(&src, out, &ctx, 150.0, radius, 8.0), &reference::unsharp(&src, out, &ctx, 150.0, radius, 8.0));
                }
            }
        }
    }
}

#[test]
fn exact_threshold_boundary_and_degenerate_images() {
    for (w, h, n) in [(1, 1, 1), (1, 7, 2), (9, 1, 3), (7, 5, 4), (2, 3, 5), (5, 2, 7)] {
        let out = Rect::new(-2, 3, w - 2, h + 3);
        let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha: n % 2 == 0 };
        let src = Image { rect: out, ch: n, data: (0..w * h * n as i32).map(|i| if i % 7 == 0 { -0.0 } else { (i % 17) as f32 / 16.0 }).collect() };
        for radius in [0.0, 0.1, 0.75, 8.5, 64.0] {
            let k = reference::gaussian_kernel(radius);
            let blurred = reference::conv_sep(&src, out, &k, &k, ctx.alpha);
            for (&o, &b) in src.data.iter().zip(&blurred).take(8) {
                let boundary = ((o - b).abs() + 1e-6) * 255.0;
                for threshold in [boundary, f32::from_bits(boundary.to_bits().saturating_sub(1)), f32::from_bits(boundary.to_bits() + 1)] {
                    same_bits(&unsharp(&src, out, &ctx, 500.0, radius, threshold), &reference::unsharp(&src, out, &ctx, 500.0, radius, threshold));
                }
            }
        }
    }
    let empty = Image { rect: Rect::EMPTY, ch: 4, data: Vec::new() };
    let ctx = Ctx { bounds: Rect::EMPTY, mode: ColorMode::Rgb, alpha: true };
    assert!(unsharp(&empty, Rect::EMPTY, &ctx, 100.0, 1.0, 0.0).is_empty());
}

#[test]
fn exact_tiled_selected_outputs_and_cancellation() {
    let bounds = Rect::new(-13, -7, 281, 271);
    let area = Rect::new(-3, 5, 273, 265);
    let mut sel = Surface::new(PixelFormat::GRAY8);
    sel.fill_rect(area, &[0.4]);
    sel.fill_rect(Rect::new(250, 250, 270, 265), &[1.0]);
    for depth in SampleType::ALL {
        let surface = source(bounds, PixelFormat::new(ColorMode::Rgb, depth, true));
        for radius in [0.6, 3.0, 10.0] {
            let p = crate::FilterParams::UnsharpMask { amount: 150.0, radius, threshold: 8.0 };
            for extent in [None, Some(bounds)] {
                let expected = apply_reference(&surface, &p, area, bounds, Some(&sel), extent, 256);
                for tile in [63, 256, 512] {
                    let actual = crate::apply_tiled(&surface, &p, area, bounds, Some(&sel), tile, extent);
                    assert_eq!(expected.to_interleaved(bounds), actual.to_interleaved(bounds));
                }
            }
            let cancelled = std::sync::atomic::AtomicBool::new(false);
            let cancel = || cancelled.load(std::sync::atomic::Ordering::Relaxed);
            let progress = |_| cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
            let ctl = Interrupt::new(&cancel, &progress);
            let input = surface.to_interleaved(bounds);
            assert!(crate::apply_in_with(&surface, &p, area, bounds, Some(&sel), bounds, &ctl).is_none());
            assert_eq!(surface.to_interleaved(bounds), input);
        }
    }
}

#[test]
fn preserve_details_resampling_keeps_the_same_sharpening() {
    use crate::resample::{Resample, resize_surface};
    let src = Rect::new(-3, 2, 19, 17);
    let (sx, sy) = (2.0, 1.5);
    let dst = crate::resample::scaled_rect(src, sx, sy);
    for depth in SampleType::ALL {
        for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
            let s = source(src, PixelFormat::new(mode, depth, true));
            let resized = resize_surface(&s, sx, sy, Resample::Lanczos);
            let p = crate::FilterParams::UnsharpMask { amount: 30.0, radius: 0.6 * sx.max(sy) as f32, threshold: 0.0 };
            let area = crate::output_area(&p, resized.content_bounds(), dst, None);
            let expected = apply_reference(&resized, &p, area, dst, None, None, crate::auto_tile(&p, dst));
            let actual = resize_surface(&s, sx, sy, Resample::PreserveDetails);
            assert_eq!(actual.to_interleaved(area), expected.to_interleaved(area));
        }
    }
}

#[test]
fn flat_fields_and_sharp_edges_keep_exact_halos() {
    let out = Rect::new(0, 0, 17, 9);
    for alpha in [false, true] {
        let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha };
        for radius in [0.1, 0.6, 3.25, 64.0] {
            let rect = out.inflate((radius * 3.0_f32).ceil() as i32 + 1);
            for edge in [false, true] {
                let n = 3 + usize::from(alpha);
                let data = (rect.y0..rect.y1)
                    .flat_map(|_| {
                        (rect.x0..rect.x1).flat_map(|x| {
                            (0..n).map(move |c| {
                                if alpha && c + 1 == n {
                                    0.7
                                } else if edge && x >= 8 {
                                    0.75
                                } else {
                                    0.25
                                }
                            })
                        })
                    })
                    .collect();
                let src = Image { rect, ch: n, data };
                same_bits(&unsharp(&src, out, &ctx, 300.0, radius, 0.0), &reference::unsharp(&src, out, &ctx, 300.0, radius, 0.0));
            }
        }
    }
}

#[test]
fn malformed_inputs_and_allocations_stop_gracefully() {
    let rect = Rect::new(0, 0, 2, 2);
    let src = Image { rect, ch: 1, data: vec![0.5; 4] };
    let ctx = Ctx { bounds: rect, mode: ColorMode::Grayscale, alpha: false };
    for (amount, radius, threshold) in [(f32::NAN, 1.0, 0.0), (100.0, f32::INFINITY, 0.0), (100.0, -1.0, 0.0), (100.0, f32::MAX, 0.0), (100.0, 1.0, f32::NAN)] {
        assert!(exact::run(&src, rect, &ctx, amount, radius, threshold, &Interrupt::NONE).is_none());
        let s = source(rect, PixelFormat::GRAY8);
        let p = crate::FilterParams::UnsharpMask { amount, radius, threshold };
        assert!(crate::apply_in_with(&s, &p, rect, rect, None, rect, &Interrupt::NONE).is_none());
        assert_eq!(crate::apply_in(&s, &p, rect, rect, None, rect), s);
    }
    for bad in [Image { ch: 0, ..src.clone() }, Image { data: vec![0.0], ..src.clone() }, Image { rect: Rect::new(5, 0, 0, 2), ..src.clone() }] {
        assert!(exact::run(&bad, rect, &ctx, 100.0, 1.0, 0.0, &Interrupt::NONE).is_none());
    }
    for rect in [Rect::new(i32::MIN, 0, i32::MAX, 2), Rect::new(0, 0, 300_000, 300_000), Rect::new(2, 0, 0, 2)] {
        assert!(exact::run(&src, rect, &ctx, 100.0, 1.0, 0.0, &Interrupt::NONE).is_none());
    }
    let s = source(Rect::new(0, 0, 2, 2), PixelFormat::GRAY8);
    let p = crate::FilterParams::UnsharpMask { amount: 100.0, radius: 1.0, threshold: 0.0 };
    for tile in [i32::MIN, 0, i32::MAX] {
        let r = s.content_bounds();
        assert_eq!(crate::apply_tiled(&s, &p, r, r, None, tile, Some(r)), crate::apply_in(&s, &p, r, r, None, r));
    }
    // Allocation/geometry guard prevents a huge halo read before allocating it.
    let p = crate::FilterParams::UnsharpMask { amount: 100.0, radius: 15_446.0, threshold: 0.0 };
    assert!(crate::apply_in_with(&s, &p, s.content_bounds(), s.content_bounds(), None, s.content_bounds(), &Interrupt::NONE).is_none());
}

#[test]
fn maximum_supported_radius_matches_the_frozen_kernel() {
    let out = Rect::new(0, 0, 1, 1);
    let src = Image { rect: out, ch: 1, data: vec![0.75] };
    let ctx = Ctx { bounds: out, mode: ColorMode::Grayscale, alpha: false };
    same_bits(&unsharp(&src, out, &ctx, 500.0, 1000.0, 8.0), &reference::unsharp(&src, out, &ctx, 500.0, 1000.0, 8.0));
}

#[test]
fn cancellation_stops_inside_a_tile_before_publishing() {
    let rect = Rect::new(0, 0, 16, 16);
    let s = source(rect, PixelFormat::RGBA8);
    let src = Image::read(&s, rect);
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    let checks = std::sync::atomic::AtomicUsize::new(0);
    let cancel = || checks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 10;
    assert!(exact::run(&src, rect, &ctx, 150.0, 30.0, 8.0, &Interrupt::cancel_only(&cancel)).is_none());
    assert_eq!(src, Image::read(&s, rect));
}

#[test]
#[ignore = "release full-operation baseline/candidate comparison; run alone"]
fn unsharp_performance() {
    let number = |name: &str, default: i32| std::env::var(name).ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(default);
    let (w, h, reps) = (number("PHOTOCRAFT_UNSHARP_WIDTH", 1500), number("PHOTOCRAFT_UNSHARP_HEIGHT", 1000), number("PHOTOCRAFT_UNSHARP_REPS", 5));
    assert!((1..=8192).contains(&w) && (1..=8192).contains(&h) && (1..=15).contains(&reps));
    let depth = match number("PHOTOCRAFT_UNSHARP_DEPTH", 8) {
        8 => SampleType::U8,
        16 => SampleType::U16,
        32 => SampleType::F32,
        _ => panic!("depth"),
    };
    let area = Rect::new(0, 0, w, h);
    let channels = number("PHOTOCRAFT_UNSHARP_CHANNELS", 4);
    let (mode, alpha) = match channels {
        1 => (ColorMode::Grayscale, false),
        2 => (ColorMode::Grayscale, true),
        3 => (ColorMode::Rgb, false),
        4 => (ColorMode::Rgb, true),
        5 => (ColorMode::Cmyk, true),
        _ => panic!("channels 1..5"),
    };
    let surface = source(area, PixelFormat::new(mode, depth, alpha));
    let measure = std::env::var("PHOTOCRAFT_UNSHARP_MODE").unwrap_or_else(|_| "paired".into());
    assert!(matches!(measure.as_str(), "paired" | "reference" | "candidate"));
    let cases = std::env::var("PHOTOCRAFT_UNSHARP_CASES").unwrap_or_else(|_| "0.6:150:0,3:150:8,10:300:16,30:150:0".into());
    for case in cases.split(',') {
        let values: Vec<f32> = case.split(':').map(|v| v.parse().unwrap()).collect();
        let (radius, amount, threshold) = (values[0], values[1], values[2]);
        let p = crate::FilterParams::UnsharpMask { amount, radius, threshold };
        let tile = crate::auto_tile(&p, area);
        let before = || apply_reference(&surface, &p, area, area, None, Some(area), tile);
        // Isolate the streamed exact path from the new adaptive FFT dispatcher.
        let after = || crate::apply_tiled_with_impl(&surface, &p, area, area, None, tile, Some(area), &Interrupt::NONE, false).unwrap();
        if measure != "paired" {
            let f = if measure == "reference" { &before as &dyn Fn() -> Surface } else { &after as &dyn Fn() -> Surface };
            let mut times = Vec::new();
            for rep in 0..reps + 2 {
                let start = Instant::now();
                let output = f();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                std::hint::black_box(output);
                if rep >= 2 {
                    times.push(ms);
                }
            }
            println!(
                "{}",
                serde_json::json!({"mode":measure,"width":w,"height":h,"channels":channels,"depth":number("PHOTOCRAFT_UNSHARP_DEPTH",8),"radius":radius,"amount":amount,"threshold":threshold,"workers":workers(),"runs_ms":times})
            );
            continue;
        }
        for _ in 0..2 {
            std::hint::black_box(before());
            std::hint::black_box(after());
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for rep in 0..reps {
            let timed = |f: &dyn Fn() -> Surface| {
                let start = Instant::now();
                let s = f();
                (s, start.elapsed().as_secs_f64() * 1000.0)
            };
            let ((old, old_ms), (new, new_ms)) = if rep % 2 == 0 {
                (timed(&before), timed(&after))
            } else {
                let b = timed(&after);
                (timed(&before), b)
            };
            assert_eq!(old.to_interleaved(area), new.to_interleaved(area));
            a.push(old_ms);
            b.push(new_ms);
        }
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!(
            "{}",
            serde_json::json!({"width":w,"height":h,"channels":channels,"depth":number("PHOTOCRAFT_UNSHARP_DEPTH",8),"radius":radius,"amount":amount,"threshold":threshold,"workers":workers(),"before_ms":a,"after_ms":b,"before_median_ms":median(&a),"after_median_ms":median(&b),"speedup":median(&a)/median(&b),"max_error":0})
        );
    }
}

#[test]
#[ignore = "unchanged-filter stage profile; run alone in release"]
fn unsharp_baseline_profile() {
    let out = Rect::new(0, 0, 256, 256);
    for radius in [0.6, 3.0, 10.0, 30.0] {
        let reach = (radius * 3.0_f32).ceil() as i32 + 1;
        let s = source(out.inflate(reach), PixelFormat::new(ColorMode::Rgb, SampleType::U8, true));
        let src = Image::read(&s, out.inflate(reach));
        let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha: true };
        let k = reference::gaussian_kernel(radius);
        let mut times = [0.0; 4];
        for _ in 0..20 {
            let start = Instant::now();
            std::hint::black_box(reference::gaussian_kernel(radius));
            times[0] += start.elapsed().as_secs_f64();
            let start = Instant::now();
            let bl = reference::conv_sep(&src, out, &k, &k, true);
            times[1] += start.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut res = src.crop(out);
            for (px, b) in res.as_chunks_mut::<4>().0.iter_mut().zip(bl.as_chunks::<4>().0) {
                if px[3] <= 0.0 {
                    continue;
                }
                for c in 0..3 {
                    let d = px[c] - b[c];
                    if d.abs() + 1e-6 >= 8.0 / 255.0 {
                        px[c] += 1.5 * d;
                    }
                }
            }
            std::hint::black_box(res);
            times[2] += start.elapsed().as_secs_f64();
            let start = Instant::now();
            std::hint::black_box(reference::unsharp(&src, out, &ctx, 150.0, radius, 8.0));
            times[3] += start.elapsed().as_secs_f64();
        }
        println!(
            "{}",
            serde_json::json!({"baseline_tile_radius":radius,"kernel_ms":times[0]*50.0,"convolution_ms":times[1]*50.0,"crop_blend_ms":times[2]*50.0,"total_kernel_ms":times[3]*50.0})
        );
    }
}
