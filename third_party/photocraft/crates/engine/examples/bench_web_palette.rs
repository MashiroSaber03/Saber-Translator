//! Compare generated-palette and edited fixed-palette GIF exports on a 24 MP image.
//! cargo run --release -p photocraft-engine --example bench_web_palette
use std::time::Instant;

use photocraft_engine::web_cmds::{WebSettings, optimize};
use photocraft_geom::Rect;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (6000usize, 4000usize);
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(w * h)?;
    for y in 0..h {
        for x in 0..w {
            pixels.push([x as f32 / w as f32, 0.5 + ((x + y) as f32 * 0.002).sin() * 0.3, y as f32 / h as f32, 1.0]);
        }
    }
    let settings = WebSettings::from_params(&json!({"format": "gif", "colors": 128, "dither": "diffusion"}), "bench")?;
    let rect = Rect::new(0, 0, w as i32, h as i32);
    let started = Instant::now();
    let original = optimize(&pixels, w, rect, &settings, None, None, 72.0, false)?;
    println!("24 MP generated palette: {:.1} ms", started.elapsed().as_secs_f64() * 1000.0);
    let mut edited = settings;
    edited.color_table = Some(original.color_table);
    edited.color_table_edits.insert(0, [0, 255, 0]);
    let started = Instant::now();
    let result = optimize(&pixels, w, rect, &edited, None, None, 72.0, false)?;
    println!("24 MP edited fixed palette: {:.1} ms ({} bytes)", started.elapsed().as_secs_f64() * 1000.0, result.bytes.len());
    Ok(())
}
