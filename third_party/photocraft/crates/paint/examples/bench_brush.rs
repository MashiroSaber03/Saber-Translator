//! Brush engine benchmark: a 500-point stroke with a 200 px textured, scattered brush on a
//! 6016×6016 RGBA8 layer, fed in 16-point chunks (as interactive painting would), compositing the
//! dirty tiles after each chunk.
//!
//! `cargo run --release -p photocraft-paint --example bench_brush`

use std::time::Instant;

use photocraft_color::{PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_paint::*;
use photocraft_raster::Surface;

fn run(fmt: PixelFormat, label: &str, brush: &BrushSettings) {
    let n = 6016;
    let mut s = Surface::with_default(fmt, &photocraft_raster::from_rgba(&fmt, [1.0; 4]));
    // Materialise the layer like a real photo layer.
    s.fill_rect(Rect::new(0, 0, n, n), &photocraft_raster::from_rgba(&fmt, [0.8, 0.7, 0.6, 1.0]));
    let pts: Vec<StrokePoint> = (0..500)
        .map(|i| {
            let t = i as f64 / 499.0;
            StrokePoint {
                time: i as f64 * 8.0,
                ..StrokePoint::new(300.0 + t * 5400.0, 3000.0 + (t * 12.0).sin() * 2000.0, 0.6 + 0.4 * (t * 30.0).sin().abs() as f32)
            }
        })
        .collect();
    let pre = s.clone();
    let start = Instant::now();
    let mut r = StrokeRenderer::new(brush, Some(fmt), 1.0);
    let mut worst = 0.0f64;
    let mut chunks = 0;
    let (mut t_push, mut t_comp) = (0.0f64, 0.0f64);
    for ch in pts.chunks(16) {
        let t = Instant::now();
        r.push(ch);
        let tp = t.elapsed().as_secs_f64() * 1000.0;
        r.composite(&pre, &mut s, None, false, false);
        let tt = t.elapsed().as_secs_f64() * 1000.0;
        t_push += tp;
        t_comp += tt - tp;
        worst = worst.max(tt);
        chunks += 1;
    }
    r.finish();
    r.composite(&pre, &mut s, None, false, false);
    let total = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{label:<34} dabs {:>5}  total {total:>7.1} ms (dabs {t_push:>6.1}, composite {t_comp:>6.1})  mean/chunk {:>6.2} ms  worst/chunk {worst:>6.2} ms",
        r.dab_count(),
        total / chunks as f64
    );
}

fn main() {
    let brush = BrushSettings {
        size: 200.0,
        hardness: 0.5,
        spacing: 0.25,
        pressure_size: false,
        seed: 1,
        shape_dynamics: ShapeDynamics { enabled: true, size: Dynamic::jitter(0.3), angle: Dynamic::jitter(1.0), ..Default::default() },
        scattering: Scattering { enabled: true, scatter: Dynamic::jitter(1.0), both_axes: true, count: 2, ..Default::default() },
        texture: Texture { enabled: true, depth: 0.7, ..Default::default() },
        ..Default::default()
    };
    let rgba8 = PixelFormat::RGBA8;
    run(rgba8, "RGBA8 textured+scattered 200px", &brush);
    run(rgba8, "RGBA8 per-tip texture", &BrushSettings { texture: Texture { each_tip: true, ..brush.texture.clone() }, ..brush.clone() });
    let chalk = presets::find(&presets::builtin(), "Chalk").unwrap().brush.clone();
    run(rgba8, "RGBA8 Chalk (sampled) 200px", &BrushSettings { size: 200.0, ..chalk });
    run(rgba8.with_sample(SampleType::U16), "RGBA16 textured+scattered", &brush);
    run(PixelFormat::RGBA32F, "RGBA32F textured+scattered", &brush);
    run(PixelFormat::CMYKA8, "CMYKA8 textured+scattered", &brush);
}
