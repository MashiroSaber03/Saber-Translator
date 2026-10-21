//! Timing: Gaussian Blur radius 10 on a 6016×6016 RGBA8 layer, as a document layer filling the
//! canvas is filtered (edges repeated at the canvas edge).
//! `cargo run --release -p photocraft-algo --example bench_gaussian [-- <size> <radius>]`
use photocraft_algo::{FilterParams, apply_in, output_area};
use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() {
    let n: i32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6016);
    let radius: f32 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let r = Rect::new(0, 0, n, n);
    let mut s = Surface::new(PixelFormat::RGBA8);
    let row: Vec<u8> = (0..n).flat_map(|x| [(x % 256) as u8, (x / 7 % 256) as u8, 128, 255]).collect();
    let bytes: Vec<u8> = (0..n).flat_map(|_| row.iter().copied()).collect();
    s.write_interleaved(r, &bytes);
    let p = FilterParams::GaussianBlur { radius };
    let area = output_area(&p, s.content_bounds(), r, None).intersect(&r);
    // Best of a few runs (the first also pays for page faults).
    let mut dt = std::time::Duration::MAX;
    let mut out = s.clone();
    for _ in 0..5 {
        let t = std::time::Instant::now();
        out = apply_in(&s, &p, area, r, None, r);
        dt = dt.min(t.elapsed());
    }
    println!("gaussian radius {radius} on {n}x{n} RGBA8: {:.2?} ({} tiles)", dt, out.tile_count());
}
