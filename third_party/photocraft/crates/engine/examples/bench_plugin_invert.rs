//! Timing: the example WebAssembly Invert plug-in vs the built-in Image › Adjustments › Invert,
//! through the full command path, on a 6000 × 4000 (24 MP) layer.
//! `cargo run --release -p photocraft-engine --example bench_plugin_invert [width] [height] [depth]`
use std::time::Instant;

use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::json;

fn main() {
    let mut args = std::env::args().skip(1).map(|v| v.parse::<u64>().unwrap_or(0));
    let w = args.next().filter(|v| *v > 0).unwrap_or(6000);
    let h = args.next().filter(|v| *v > 0).unwrap_or(4000);
    let depth = args.next().filter(|v| *v > 0).unwrap_or(8);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("texture", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let row: Vec<f32> = (0..w).flat_map(|x| photocraft_raster::from_rgba(&fmt, [(x % 256) as f32 / 255.0, 0.5, 0.25, 1.0])).collect();
        for y in 0..h as i32 {
            surf.write_region(Rect::new(0, y, w as i32, y + 1), &row);
        }
        Ok(())
    })
    .unwrap();
    let wasm = include_bytes!("../../plugins/tests/fixtures/invert.wasm");
    let data: String = {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        wasm.chunks(3)
            .flat_map(|c| {
                let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
                (0..4).map(move |i| if i <= c.len() { T[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' })
            })
            .collect()
    };
    s.execute("plugin.install", json!({"data": data})).unwrap();
    println!("{w}x{h} ({:.1} MP), {depth}-bit RGBA layer", (w * h) as f64 / 1e6);
    for _ in 0..3 {
        let t = Instant::now();
        s.execute("image.adjustments.invert", json!({})).unwrap();
        let builtin = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        s.execute("plugin.run", json!({"id": "org.photocraft.example.invert"})).unwrap();
        let plugin = t.elapsed().as_secs_f64() * 1e3;
        println!("built-in invert {builtin:>7.0} ms   plug-in invert {plugin:>7.0} ms");
    }
}
