//! Times the example Invert plug-in on a large image:
//! `cargo run -p photocraft-plugins --release --example bench_invert -- [width] [height]`
//! (default 6000 × 4000 = 24 MP, RGBA 8-bit).

use std::time::Instant;

use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_plugins::{Limits, Plugin};
use photocraft_raster::Surface;

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<i32>().unwrap_or(0));
    let w = args.next().filter(|v| *v > 0).unwrap_or(6000);
    let h = args.next().filter(|v| *v > 0).unwrap_or(4000);
    let mut s = Surface::new(PixelFormat::RGBA8);
    let row: Vec<f32> = (0..w).flat_map(|x| [(x % 256) as f32 / 255.0, 0.5, 0.25, 1.0]).collect();
    for y in 0..h {
        s.write_region(Rect::new(0, y, w, y + 1), &row);
    }
    let p = Plugin::load(include_bytes!("../tests/fixtures/invert.wasm"), Limits::default()).expect("load");
    if std::env::var("NOOP").is_ok() {
        let noop = wat::parse_str(NOOP).unwrap();
        let p = Plugin::load(&noop, Limits::default()).unwrap();
        let t = Instant::now();
        p.apply(&s, Rect::new(0, 0, w, h), None, &serde_json::json!({})).unwrap();
        println!("noop: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    let canvas = Rect::new(0, 0, w, h);
    for run in 0..3 {
        let t = Instant::now();
        let out = p.apply(&s, canvas, None, &serde_json::json!({})).expect("apply");
        println!("run {run}: {w}x{h} ({:.1} MP) in {:.0} ms", (w as f64 * h as f64) / 1e6, t.elapsed().as_secs_f64() * 1e3);
        assert!((out.pixel(1, 1)[0] - (1.0 - 1.0 / 255.0)).abs() < 1e-6);
    }
}

const NOOP: &str = r#"(module (memory (export "memory") 1) (data (i32.const 16) "{\"id\":\"n\",\"name\":\"n\",\"kind\":\"filter\"}")
  (func (export "pc_abi_version") (result i32) (i32.const 1))
  (func (export "pc_manifest") (result i64) (i64.const 0x0000002500000010))
  (func (export "pc_alloc") (param $n i32) (result i32) (local $old i32)
    (local.set $old (memory.grow (i32.shr_u (i32.add (local.get $n) (i32.const 65535)) (i32.const 16))))
    (if (result i32) (i32.eq (local.get $old) (i32.const -1)) (then (i32.const 0)) (else (i32.mul (local.get $old) (i32.const 65536)))))
  (func (export "pc_filter") (param i32 i32 i32 i32 i32 i32 i32 i32) (result i32) (i32.const 0)))"#;
