//! Opt-in release comparisons against the Motion Blur row kernel merged in upstream PR #902.
#![cfg(not(target_arch = "wasm32"))]

use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;
use rayon::prelude::*;
use serde_json::json;
use std::time::Instant;

/// Copy of upstream's tiled apply_in pipeline, with its merged row kernel called directly so
/// the experimental FFT dispatch cannot change the baseline. Source reads, automatic tile
/// sizing, Rayon groups, writes, progress and pruning follow the current upstream implementation.
#[allow(clippy::panic)] // Fixed benchmark parameters, never a production entry point.
fn apply_reference(surface: &Surface, params: &FilterParams, area: Rect) -> Surface {
    let FilterParams::MotionBlur { angle, distance } = params else { panic!("Motion Blur benchmark parameters required") };
    let crate::Halo::Radius(halo) = params.halo_for(area) else { panic!("Motion Blur must declare a radius halo") };
    let tile = crate::auto_tile(params, area);
    let fmt = surface.format();
    let ctx = Ctx { bounds: area, mode: fmt.mode, alpha: fmt.alpha };
    let mut out = surface.clone();
    let mut tiles = Vec::new();
    let mut y = area.y0;
    while y < area.y1 {
        let mut x = area.x0;
        while x < area.x1 {
            tiles.push(Rect::new(x, y, (x + tile).min(area.x1), (y + tile).min(area.y1)));
            x += tile;
        }
        y += tile;
    }
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let total = tiles.len().max(1);
    let ctl = Interrupt::NONE;
    let run = |t: &Rect| {
        if ctl.cancelled() {
            return (*t, Vec::new());
        }
        let src = Image::read_clamped(surface, t.inflate(halo), area);
        let data = motion_rows(&src, *t, &ctx, *angle, *distance);
        let n = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        ctl.progress(0.98 * n as f32 / total as f32);
        (*t, data)
    };
    let per_tile = (tile.max(1) as usize).pow(2) * fmt.channels() * std::mem::size_of::<f32>();
    let group = (crate::RESULT_BUDGET / per_tile.max(1)).max(rayon::current_num_threads()).max(1);
    for chunk in tiles.chunks(group) {
        let results: Vec<_> = chunk.par_iter().map(run).collect();
        for (t, data) in results {
            out.write_region(t, &data);
        }
    }
    out.prune();
    ctl.progress(1.0);
    out
}

fn source(area: Rect, variable_alpha: bool, channels: usize) -> Surface {
    let (mode, alpha) = match channels {
        1 => (ColorMode::Grayscale, false),
        2 => (ColorMode::Grayscale, true),
        3 => (ColorMode::Rgb, false),
        4 => (ColorMode::Rgb, true),
        5 => (ColorMode::Cmyk, true),
        _ => (ColorMode::Rgb, true),
    };
    let mut surface = Surface::new(PixelFormat::new(mode, SampleType::U8, alpha));
    let bytes: Vec<u8> = (area.y0..area.y1)
        .flat_map(|y| {
            (area.x0..area.x1).flat_map(move |x| {
                let (r, g, b) = ((x % 256) as u8, ((x * 3 + y) / 11 % 256) as u8, ((x ^ y) % 256) as u8);
                let alpha = if variable_alpha { (x ^ (y * 7)) as u8 } else { 255 };
                let pixel = match channels {
                    1 => [r, 0, 0, 0, 0],
                    2 => [r, alpha, 0, 0, 0],
                    3 => [r, g, b, 0, 0],
                    4 => [r, g, b, alpha, 0],
                    5 => [r, g, b, ((x * 5 + y * 7) % 256) as u8, alpha],
                    _ => [0; 5],
                };
                pixel.into_iter().take(channels)
            })
        })
        .collect();
    surface.write_interleaved(area, &bytes);
    surface
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.get(sorted.len() / 2) {
        Some(&value) if !sorted.len().is_multiple_of(2) => value,
        Some(&value) => (value + sorted.get(sorted.len() / 2 - 1).copied().unwrap_or(value)) * 0.5,
        None => 0.0,
    }
}

#[allow(clippy::panic)] // Invalid benchmark configuration should fail visibly.
fn env_number(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(value) => match value.parse() {
            Ok(value) => value,
            Err(_) => panic!("invalid benchmark number {name}={value}"),
        },
        Err(_) => default,
    }
}

#[allow(clippy::panic)] // Invalid benchmark configuration should fail visibly.
fn cases() -> Vec<(f32, f32)> {
    let spec = std::env::var("PHOTOCRAFT_MOTION_CASES")
        .unwrap_or_else(|_| "30:10,30:50,30:64,30:128,30:256,30:500,30:1000,30:2000,0:256,0:1000,0:2000,90:256,90:1000,90:2000,45:256,45:1000,45:2000".into());
    spec.split(',')
        .map(|case| {
            let Some((angle, distance)) = case.trim().split_once(':') else { panic!("expected angle:distance in PHOTOCRAFT_MOTION_CASES") };
            let (Ok(angle), Ok(distance)) = (angle.parse::<f32>(), distance.parse::<f32>()) else { panic!("invalid Motion Blur benchmark case {case}") };
            assert!(angle.is_finite() && (0.5..=2000.0).contains(&distance), "out-of-range benchmark case {case}");
            (angle, distance)
        })
        .collect()
}

#[allow(clippy::panic)] // A cancellation with Interrupt::NONE is a benchmark failure.
fn candidate(surface: &Surface, params: &FilterParams, area: Rect, forced: bool) -> (Surface, bool) {
    if forced {
        match super::motion_apply::apply_for_bench(surface, params, area, area, None, Some(area), crate::auto_tile(params, area), &Interrupt::NONE) {
            Some(Some(result)) => return (result, true),
            Some(None) => panic!("uncancellable forced FFT benchmark was cancelled"),
            None => {}
        }
    }
    (crate::apply_in(surface, params, area, area, None, area), false)
}

fn compare(surface: &Surface, area: Rect, angle: f32, distance: f32, runs: usize, forced: bool, variable_alpha: bool) {
    let params = FilterParams::MotionBlur { angle, distance };
    let channels = surface.channels();
    let mut upstream_ms = Vec::with_capacity(runs);
    let mut candidate_ms = Vec::with_capacity(runs);
    let mut largest_error = 0u8;
    let mut changed_samples = 0usize;
    let mut squared_error = 0u64;
    let mut compared_samples = 0usize;
    let mut fft_applied = false;
    for run in 0..runs {
        let upstream = || {
            let start = Instant::now();
            let result = apply_reference(surface, &params, area);
            (result, start.elapsed().as_secs_f64() * 1000.0)
        };
        let after = || {
            let start = Instant::now();
            let (result, used_fft) = candidate(surface, &params, area, forced);
            (result, start.elapsed().as_secs_f64() * 1000.0, used_fft)
        };
        let ((before, before_ms), (after, after_ms, used_fft)) = if run % 2 == 0 {
            (upstream(), after())
        } else {
            let after = after();
            (upstream(), after)
        };
        upstream_ms.push(before_ms);
        candidate_ms.push(after_ms);
        fft_applied |= used_fft;
        let before = before.to_interleaved(area);
        let after = after.to_interleaved(area);
        assert_eq!(before.len(), after.len());
        let mut maximum = 0u8;
        let mut changed = 0usize;
        let mut squared = 0u64;
        for (&before, &after) in before.iter().zip(&after) {
            let error = before.abs_diff(after);
            maximum = maximum.max(error);
            changed += usize::from(error != 0);
            squared += u64::from(error).pow(2);
        }
        assert!(maximum <= 1, "upstream comparison exceeded one U8 level: angle={angle}, distance={distance}, error={maximum}");
        largest_error = largest_error.max(maximum);
        changed_samples += changed;
        squared_error += squared;
        compared_samples += before.len();
        println!(
            "{}",
            json!({"event":"motion_upstream_run", "width":area.width(), "height":area.height(), "channels":channels, "angle":angle, "distance":distance, "run":run, "mode":if forced {"forced_fft"} else {"public"}, "fft_applied":if forced {Some(used_fft)} else {None}, "variable_alpha":variable_alpha, "upstream_ms":before_ms, "candidate_ms":after_ms, "max_u8_error":maximum, "changed_samples":changed, "rms_u8_error":(squared as f64 / before.len() as f64).sqrt()})
        );
    }
    println!(
        "{}",
        json!({"event":"motion_upstream_median", "width":area.width(), "height":area.height(), "channels":channels, "angle":angle, "distance":distance, "runs":runs, "mode":if forced {"forced_fft"} else {"public"}, "fft_applied":if forced {Some(fft_applied)} else {None}, "variable_alpha":variable_alpha, "upstream_median_ms":median(&upstream_ms), "candidate_median_ms":median(&candidate_ms), "speedup":median(&upstream_ms)/median(&candidate_ms), "upstream_runs_ms":upstream_ms, "candidate_runs_ms":candidate_ms, "max_u8_error":largest_error, "changed_samples":changed_samples, "rms_u8_error":(squared_error as f64 / compared_samples as f64).sqrt()})
    );
}

/// Reproduce in release with --ignored --nocapture. PHOTOCRAFT_MOTION_WIDTH/HEIGHT default to
/// 1500/1000; PHOTOCRAFT_MOTION_RUNS defaults to 3; PHOTOCRAFT_MOTION_CASES is comma-separated
/// angle:distance pairs. PHOTOCRAFT_MOTION_MODE=public measures final dispatch, otherwise forced
/// FFT is attempted (normal eligibility/memory guards remain). PHOTOCRAFT_MOTION_CHANNELS=1..5
/// selects Gray, Gray-alpha, RGB, RGBA (default), or CMYK-alpha. PHOTOCRAFT_MOTION_VARIABLE_ALPHA=1
/// enables varying alpha in formats that have alpha. Timings include reads, writes and pruning, excluding
/// source construction, output comparison, proxy construction, compositing and GPU upload.
#[test]
#[ignore = "upstream row kernel versus FFT/full-dispatch release calibration"]
fn motion_upstream_performance() {
    let width = env_number("PHOTOCRAFT_MOTION_WIDTH", 1500);
    let height = env_number("PHOTOCRAFT_MOTION_HEIGHT", 1000);
    let runs = env_number("PHOTOCRAFT_MOTION_RUNS", 3) as usize;
    let channels = env_number("PHOTOCRAFT_MOTION_CHANNELS", 4) as usize;
    assert!((1..=16384).contains(&width) && (1..=16384).contains(&height));
    assert!((1..=31).contains(&runs));
    assert!((1..=5).contains(&channels));
    let area = Rect::new(0, 0, width as i32, height as i32);
    let mode = std::env::var("PHOTOCRAFT_MOTION_MODE").unwrap_or_else(|_| "forced_fft".into());
    assert!(matches!(mode.as_str(), "public" | "forced_fft"), "PHOTOCRAFT_MOTION_MODE must be public or forced_fft");
    let forced = mode == "forced_fft";
    let variable_alpha = matches!(channels, 2 | 4 | 5) && std::env::var("PHOTOCRAFT_MOTION_VARIABLE_ALPHA").is_ok_and(|value| value != "0" && value != "false");
    println!(
        "{}",
        json!({"event":"motion_upstream_config", "width":width, "height":height, "channels":channels, "runs":runs, "mode":if forced {"forced_fft"} else {"public"}, "variable_alpha":variable_alpha, "rayon_workers":rayon::current_num_threads(), "os":std::env::consts::OS, "arch":std::env::consts::ARCH, "baseline":"merged upstream PR902 motion_rows plus current upstream tiled pipeline", "timing_scope":"source decode/halo reads, filtering, writes, pruning; excludes construction/comparison/UI"})
    );
    let surface = source(area, variable_alpha, channels);
    for (angle, distance) in cases() {
        compare(&surface, area, angle, distance, runs, forced, variable_alpha);
    }
}
