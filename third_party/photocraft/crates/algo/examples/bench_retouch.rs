//! Timing of the retouch kernels alone (no document plumbing): Poisson seamless clone and
//! PatchMatch completion on a disc-shaped region, RGBA.
//! `cargo run --release -p photocraft-algo --example bench_retouch`
use std::time::Instant;

use photocraft_algo::{inpaint, poisson};

fn main() {
    for d in [50usize, 100, 200, 300] {
        let n = d + 6;
        let (w, h, ch) = (n, n, 4);
        let img: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let v = 0.5 + 0.2 * (x * 0.1).sin() * (y * 0.13).cos();
                [v, v * 0.8, 0.3, 1.0]
            })
            .collect();
        let src: Vec<f32> = img.iter().map(|v| v * 0.7).collect();
        let r = d as f32 / 2.0;
        let mask: Vec<bool> = (0..w * h).map(|i| ((i % w) as f32 - n as f32 / 2.0).hypot((i / w) as f32 - n as f32 / 2.0) < r).collect();
        let t = Instant::now();
        let out = poisson::seamless_clone(w, h, ch, &src, &img, &mask);
        println!("seamless_clone  {d:>3} px disc: {:>8.1?} ({})", t.elapsed(), out.len());
        // Completion on a region with a margin, like Spot Healing.
        let m = d + 16;
        let (cw, chh) = (d + 2 * m, d + 2 * m);
        let cimg: Vec<f32> = (0..cw * chh)
            .flat_map(|i| {
                let (x, y) = ((i % cw) as f32, (i / cw) as f32);
                let v = 0.5 + 0.2 * (x * 0.1).sin() * (y * 0.13).cos();
                [v, v * 0.8, 0.3, 1.0]
            })
            .collect();
        let c = cw as f32 / 2.0;
        let hole: Vec<bool> = (0..cw * chh).map(|i| ((i % cw) as f32 - c).hypot((i / cw) as f32 - c) < r + 2.0).collect();
        let t = Instant::now();
        let _ = inpaint::complete(cw, chh, ch, &cimg, &hole, &inpaint::CompleteParams::default());
        println!("complete        {d:>3} px disc: {:>8.1?}", t.elapsed());
        let t = Instant::now();
        let _ = inpaint::best_offset(cw, chh, ch, &cimg, &hole, (d / 8).clamp(3, 16), m as i32);
        println!("best_offset     {d:>3} px disc: {:>8.1?}", t.elapsed());
    }
}
