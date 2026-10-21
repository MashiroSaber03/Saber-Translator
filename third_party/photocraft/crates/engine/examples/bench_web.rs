//! Timing: File › Export › Save for Web (Legacy) and File › Print on a 6000×4000 (24 MP) image.
//! `cargo run --release -p photocraft-engine --example bench_web [width] [height]`
//!
//! Each line is one full command (flatten, sRGB, quantise/encode), estimate-only (no file).
use std::time::Instant;

use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::json;

fn main() {
    let w: i32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6000);
    let h: i32 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(4000);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h})).unwrap();
    s.edit("texture", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) as u32 % 1000) as f32 / 1000.0 * 0.06;
                data.extend(photocraft_raster::from_rgba(
                    &fmt,
                    [x as f32 / w as f32 + n, 0.5 + ((x + y) as f32 * 0.002).sin() * 0.3, y as f32 / h as f32, 1.0],
                ));
            }
        }
        surf.write_region(Rect::new(0, 0, w, h), &data);
        Ok(())
    })
    .unwrap();
    for (label, p) in [
        ("GIF 128 no dither", json!({"format": "gif", "colors": 128, "dither": "none"})),
        ("GIF 128 pattern", json!({"format": "gif", "colors": 128, "dither": "pattern"})),
        ("GIF 128 diffusion", json!({"format": "gif", "colors": 128, "dither": "diffusion"})),
        ("PNG-8 64 no dither", json!({"format": "png8", "colors": 64, "dither": "none"})),
        ("PNG-24", json!({"format": "png24"})),
        ("JPEG 60", json!({"format": "jpeg", "quality": 60})),
        ("JPEG 60 progressive", json!({"format": "jpeg", "quality": 60, "progressive": true})),
        ("WBMP diffusion", json!({"format": "wbmp"})),
    ] {
        let t = Instant::now();
        let r = s.execute("file.export.saveForWebLegacy", p).unwrap();
        println!("{label:24} {:8.0} ms  {:>10} bytes", t.elapsed().as_secs_f64() * 1000.0, r["bytes"]);
    }
    let t = Instant::now();
    let r = s.execute("file.print", json!({"dryRun": true, "scaleToFit": true})).unwrap();
    println!("{:24} {:8.0} ms  {:>10} bytes", "Print (PDF, dry run)", t.elapsed().as_secs_f64() * 1000.0, r["bytes"]);
    let _ = std::fs::remove_file(r["pdf"].as_str().unwrap_or_default());
}
