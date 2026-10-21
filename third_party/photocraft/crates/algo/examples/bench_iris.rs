//! End-to-end Iris Blur timings; fixture construction and output drop are untimed.
//! `cargo run --release -p photocraft-algo --example bench_iris -- 6000 4000 3`
//! An optional fourth argument models the UI's proxy factor (e.g. 1500 1000 5 4).
use photocraft_algo::{FilterParams, IrisPin, apply_in};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let number = |i: usize, fallback| args.get(i).and_then(|s| s.parse::<i32>().ok()).unwrap_or(fallback);
    let (w, h, reps) = (number(1, 6000).clamp(1, 6000), number(2, 4000).clamp(1, 6000), number(3, 3).clamp(1, 20));
    let proxy_factor = number(4, 1).clamp(1, 8) as f32;
    let bounds = Rect::new(0, 0, w, h);
    for (sample, name) in [(SampleType::U8, "U8"), (SampleType::U16, "U16"), (SampleType::F32, "F32")] {
        let mut source = Surface::new(PixelFormat::new(ColorMode::Rgb, sample, true));
        // Write rows to avoid a second full-image fixture allocation.
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
        let mut cases = vec![("medium", vec![IrisPin { blur: 80.0, angle: 27.0, ..IrisPin::default() }])];
        if sample == SampleType::U8 {
            cases.extend([
                ("small", vec![IrisPin::default()]),
                ("large", vec![IrisPin { blur: 300.0, ..IrisPin::default() }]),
                (
                    "multiple",
                    vec![
                        IrisPin { x: 0.3, blur: 80.0, roundness: 65.0, ..IrisPin::default() },
                        IrisPin { x: 0.8, angle: 73.0, blur: 40.0, ..IrisPin::default() },
                    ],
                ),
                ("focused", vec![IrisPin { radius_x: 5.0, radius_y: 5.0, blur: 80.0, ..IrisPin::default() }]),
                ("blurred", vec![IrisPin { x: -5.0, blur: 80.0, ..IrisPin::default() }]),
            ]);
        }
        for (label, mut pins) in cases {
            // Top-level one-pin blur scales in preview_document_with. Its nested
            // JSON pin list retains supplied amounts; model that existing path.
            if pins.len() == 1 {
                for pin in &mut pins {
                    pin.blur /= proxy_factor;
                }
            }
            let p = FilterParams::IrisBlur { pins };
            let warm = apply_in(&source, &p, bounds, bounds, None, bounds);
            std::hint::black_box(&warm);
            drop(warm);
            let mut times = Vec::new();
            for _ in 0..reps {
                let start = std::time::Instant::now();
                let output = apply_in(&source, &p, bounds, bounds, None, bounds);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&output);
                drop(output);
            }
            println!("{w}x{h} {name} {label}: {times:?} ms");
        }
    }
}
