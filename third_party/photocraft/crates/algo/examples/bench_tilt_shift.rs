//! Complete Tilt-Shift filtering, including tiled reads and writes.
//! Run alone in release: `bench_tilt_shift 6000 4000 5 1`.
//! Fourth argument models the existing preview proxy scale (e.g. 1500 1000 7 4).
use photocraft_algo::{FilterParams, apply_in};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let number = |i: usize, fallback| args.get(i).and_then(|s| s.parse::<i32>().ok()).unwrap_or(fallback);
    let (w, h, reps) = (number(1, 6000).clamp(1, 6000), number(2, 4000).clamp(1, 6000), number(3, 5).clamp(1, 20));
    let proxy = number(4, 1).clamp(1, 8) as f32;
    let bounds = Rect::new(0, 0, w, h);
    for (sample, name) in [(SampleType::U8, "U8"), (SampleType::U16, "U16"), (SampleType::F32, "F32")] {
        let mut source = Surface::new(PixelFormat::new(ColorMode::Rgb, sample, true));
        for y in 0..h {
            let row: Vec<f32> = (0..w)
                .flat_map(|x| {
                    let a = if (x + y) % 31 == 0 { 0.0 } else { 0.7 + (x % 7) as f32 / 30.0 };
                    let hdr = if sample == SampleType::F32 { 3.0 } else { 1.0 };
                    [hdr * (x % 256) as f32 / 255.0, ((x * 3 + y) % 251) as f32 / 250.0, ((x ^ y) % 256) as f32 / 255.0, a]
                })
                .collect();
            source.write_region(Rect::new(0, y, w, y + 1), &row);
        }
        let mut cases = vec![("medium", 80.0, 0.5, 0.5, 27.0, 0.1, 0.15)];
        if sample == SampleType::U8 {
            cases.extend([
                ("small", 15.0, 0.5, 0.5, 0.0, 0.1, 0.15),
                ("large", 300.0, 0.5, 0.5, -71.0, 0.03, 0.4),
                ("wide_focus", 80.0, 0.5, 0.5, 0.0, 0.4, 0.05),
                ("sharp_transition", 80.0, 0.2, 0.8, 63.0, 0.05, 0.01),
                ("focused", 80.0, 0.5, 0.5, 0.0, 1.0, 0.15),
                ("blurred", 80.0, -5.0, -5.0, 27.0, 0.1, 0.15),
            ]);
        }
        for (label, blur, center_x, center_y, angle, focus, transition) in cases {
            let params = FilterParams::TiltShift { blur: blur / proxy, center_x, center_y, angle, focus, transition };
            for _ in 0..2 {
                std::hint::black_box(apply_in(&source, &params, bounds, bounds, None, bounds));
            }
            let mut times = Vec::new();
            for _ in 0..reps {
                let start = std::time::Instant::now();
                let output = apply_in(&source, &params, bounds, bounds, None, bounds);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&output);
                drop(output);
            }
            println!("{}", serde_json::json!({"width": w, "height": h, "depth": name, "case": label, "proxy": proxy, "ms": times}));
        }
    }
}
