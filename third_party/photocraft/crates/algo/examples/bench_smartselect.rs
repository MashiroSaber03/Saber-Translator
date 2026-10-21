//! Timing: smart selection on a large synthetic RGBA8 layer (default 6016×6016): a textured
//! disc (radius = 1/3 of the size) on a noisy background.
//! `cargo run --release -p photocraft-algo --example bench_smartselect [size]`
use std::time::Instant;

use photocraft_algo::matting::{self, RefineParams};
use photocraft_algo::segment::{Rng, SurfaceSampler, focus, grabcut, quick, subject};
use photocraft_algo::selection::{Region, SelectionMode, combine_region};
use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() {
    let n: i32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6016);
    let (c, rad) = (n as f32 / 2.0, n as f32 / 3.0);
    let canvas = Rect::new(0, 0, n, n);
    let t = Instant::now();
    let mut s = Surface::new(PixelFormat::RGBA8);
    let rows: Vec<Vec<u8>> = {
        let make = |y: i32| -> Vec<u8> {
            let mut rng = Rng::new(y as u64 + 1);
            let mut row = Vec::with_capacity(n as usize * 4);
            for x in 0..n {
                let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
                let inside = dx * dx + dy * dy <= rad * rad;
                let noise = 0.05 * rng.normal();
                let p = if inside {
                    let t = ((x + y) as f32 * 0.05).sin() * 0.5 + 0.5;
                    [0.85 + 0.05 * t + noise, 0.2 + 0.3 * t + noise, 0.15 + noise]
                } else {
                    [0.25 + noise, 0.45 + noise, 0.7 + noise]
                };
                row.extend(p.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8));
                row.push(255);
            }
            row
        };
        use rayon::prelude::*;
        (0..n).into_par_iter().map(make).collect()
    };
    for (y, row) in rows.iter().enumerate() {
        s.write_interleaved(Rect::new(0, y as i32, n, y as i32 + 1), row);
    }
    drop(rows);
    println!("built {n}x{n} RGBA8 in {:.2?}", t.elapsed());
    let sampler = SurfaceSampler(&s);
    let truth_iou = |r: &Region| -> f32 {
        let (mut i, mut u) = (0u64, 0u64);
        let step = (n / 1000).max(1);
        for y in (0..n).step_by(step as usize) {
            for x in (0..n).step_by(step as usize) {
                let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
                let a = r.at(x, y) >= 0.5;
                let b = dx * dx + dy * dy <= rad * rad;
                i += (a && b) as u64;
                u += (a || b) as u64;
            }
        }
        i as f32 / u.max(1) as f32
    };

    // Quick selection: small and large brush strokes inside the disc near its edge.
    for size in [30.0f32, 300.0] {
        let x = c + rad - size * 2.0;
        let t = Instant::now();
        let r = quick::quick_select(&sampler, canvas, &[(x, c - size), (x, c + size)], size).expect("region");
        println!("quick selection stroke (brush {size} px): {:.2?}  (region {}x{})", t.elapsed(), r.bbox.width(), r.bbox.height());
    }

    // Object selection (rectangle around the disc).
    let m = (rad * 1.15) as i32;
    let rect = Rect::new(c as i32 - m, c as i32 - m, c as i32 + m, c as i32 + m);
    let t = Instant::now();
    let obj = grabcut::object_select(&sampler, canvas, rect, 160_000).expect("object");
    println!("object selection: {:.2?}  IoU {:.4}", t.elapsed(), truth_iou(&obj));

    // Select subject.
    let t = Instant::now();
    let subj = subject::select_subject(&sampler, canvas).expect("subject");
    println!("select subject: {:.2?}  IoU {:.4}", t.elapsed(), truth_iou(&subj));

    // Refine edge on a 20 px band (selection = the object result as a surface).
    let sel = combine_region(None, Some(&obj), SelectionMode::Replace).expect("selection");
    let content = sel.content_bounds();
    let p = RefineParams { radius: 20.0, smooth: 10.0, feather: 1.0, contrast: 10.0, ..Default::default() };
    let t = Instant::now();
    let refined = matting::refine_mask(&sampler, &matting::surface_reader(&sel), content, canvas, &p).expect("refined");
    println!("refine edge (radius 20): {:.2?}", t.elapsed());
    let t = Instant::now();
    let _ = matting::refine_mask(&sampler, &matting::surface_reader(&sel), content, canvas, &RefineParams { smart_radius: true, ..p }).expect("refined");
    println!("refine edge (radius 20, smart): {:.2?}", t.elapsed());
    let t = Instant::now();
    let dec = matting::decontaminate(&s, &refined, 20.0, 100.0);
    println!("decontaminate: {:.2?}  ({} tiles)", t.elapsed(), dec.tile_count());

    // Focus area.
    let t = Instant::now();
    let _ = focus::focus_area(&sampler, canvas, 0.5, 0.0);
    println!("focus area: {:.2?}", t.elapsed());
}
