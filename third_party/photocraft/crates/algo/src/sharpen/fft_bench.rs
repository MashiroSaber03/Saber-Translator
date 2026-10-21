//! Opt-in benchmark of the adaptive production path against the frozen original.
use super::tests::{apply_reference, source};
use crate::FilterParams;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use std::time::Instant;
#[test]
#[ignore = "adaptive large-radius full-operation quality and performance; run alone"]
fn unsharp_fft_performance() {
    let number = |name: &str, default| std::env::var(name).ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(default);
    let (w, h) = (number("PHOTOCRAFT_UNSHARP_WIDTH", 1920), number("PHOTOCRAFT_UNSHARP_HEIGHT", 1080));
    let reps = number("PHOTOCRAFT_UNSHARP_REPS", 3);
    assert!((1..=8192).contains(&w) && (1..=8192).contains(&h) && (1..=5).contains(&reps));
    let depth = match number("PHOTOCRAFT_UNSHARP_DEPTH", 8) {
        8 => SampleType::U8,
        16 => SampleType::U16,
        32 => SampleType::F32,
        _ => panic!("depth"),
    };
    let area = Rect::new(0, 0, w, h);
    let surface = source(area, PixelFormat::new(ColorMode::Rgb, depth, true));
    let cases = std::env::var("PHOTOCRAFT_UNSHARP_CASES").unwrap_or_else(|_| "100:150:0,300:150:0,500:150:8".into());
    for case in cases.split(',') {
        let v: Vec<f32> = case.split(':').map(|v| v.parse().unwrap()).collect();
        let (radius, amount, threshold) = (v[0], v[1], v[2]);
        let params = FilterParams::UnsharpMask { amount, radius, threshold };
        let baseline = || apply_reference(&surface, &params, area, area, None, Some(area), crate::auto_tile(&params, area));
        let candidate = || crate::apply_in(&surface, &params, area, area, None, area);
        for _ in 0..2 {
            std::hint::black_box(baseline());
            std::hint::black_box(candidate());
        }
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut max_error = 0.0f32;
        let mut changed = 0usize;
        let mut sum_error = 0.0f64;
        for rep in 0..reps {
            let timed = |f: &dyn Fn() -> Surface| {
                let t = Instant::now();
                let output = f();
                (output, t.elapsed().as_secs_f64() * 1000.0)
            };
            let ((old, a), (new, b)) = if rep % 2 == 0 {
                (timed(&baseline), timed(&candidate))
            } else {
                let b = timed(&candidate);
                (timed(&baseline), b)
            };
            before.push(a);
            after.push(b);
            for (a, b) in old.read_region(area).iter().zip(new.read_region(area)) {
                let e = (a - b).abs();
                let limit = match depth {
                    SampleType::U8 => 1.0 / 255.0 + 1e-7,
                    SampleType::U16 => 3.0 / 65535.0 + 1e-7,
                    SampleType::F32 => 3e-5 * (1.0 + a.abs()),
                };
                assert!(e <= limit, "quality regression: {a} != {b} at radius {radius}, threshold {threshold}");
                max_error = max_error.max(e);
                if rep == 0 {
                    sum_error += f64::from(e);
                    changed += usize::from(a.to_bits() != b.to_bits());
                }
            }
        }
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!(
            "{}",
            serde_json::json!({"experiment":"adaptive FFT same Gaussian, changed summation","width":w,"height":h,"depth":number("PHOTOCRAFT_UNSHARP_DEPTH",8),"radius":radius,"amount":amount,"threshold":threshold,"before_ms":before,"after_ms":after,"before_median_ms":median(&before),"after_median_ms":median(&after),"speedup":median(&before)/median(&after),"max_error":max_error,"mean_error":sum_error/(w as f64*h as f64*4.0),"changed_samples":changed})
        );
    }
}

#[test]
#[ignore = "4K wide-radius cancellation latency; run alone in release"]
fn unsharp_fft_cancel_latency() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    let area = Rect::new(0, 0, 3840, 2160);
    let surface = source(area, PixelFormat::RGBA8);
    let input = surface.to_interleaved(area);
    let p = FilterParams::UnsharpMask { amount: 150.0, radius: 1000.0, threshold: 0.0 };
    let mut latencies = Vec::new();
    for _ in 0..5 {
        let flag = AtomicBool::new(false);
        let cancel_at = AtomicU64::new(0);
        let started = Instant::now();
        std::thread::scope(|scope| {
            let timer = scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                cancel_at.store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                flag.store(true, Ordering::Release);
            });
            let cancel = || flag.load(Ordering::Acquire);
            let result = crate::apply_in_with(&surface, &p, area, area, None, area, &photocraft_raster::Interrupt::cancel_only(&cancel));
            let elapsed = started.elapsed().as_micros() as u64;
            timer.join().unwrap();
            assert!(result.is_none());
            latencies.push(elapsed.saturating_sub(cancel_at.load(Ordering::Relaxed)) as f64 / 1000.0);
        });
        assert_eq!(surface.to_interleaved(area), input);
    }
    println!(
        "{}",
        serde_json::json!({"width":3840,"height":2160,"radius":1000,"cancel_after_ms":10,"cancel_return_latencies_ms":latencies,"scope":"native full-resolution algo dispatch, excludes GUI scheduling/proxy creation"})
    );
}
