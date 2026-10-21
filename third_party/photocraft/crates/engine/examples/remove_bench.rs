//! Timing: the Remove Tool (`paint.remove`) on a 6000×4000 (24 MP) document, through the full
//! command path (`Session::execute`: window read, hole, fill, write, undo snapshot).
//! `cargo run --release -p photocraft-engine --example remove_bench [depth]`
//!
//! Rings around objects of 100, 300 and 1000 px (one completion each) and a 3000 px thin line
//! (completed tile by tile), each undone before the next.
use std::time::Instant;

use photocraft_engine::Session;
use serde_json::json;

fn ring(cx: f64, cy: f64, r: f64) -> Vec<[f64; 2]> {
    (0..=96).map(|i| f64::from(i) / 96.0 * std::f64::consts::TAU).map(|t| [cx + r * t.cos(), cy + r * t.sin()]).collect()
}

fn main() {
    let depth: u64 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 6000, "height": 4000, "depth": depth})).unwrap();
    s.execute("filter.render.clouds", json!({"seed": 3})).unwrap();
    s.execute("filter.noise.addNoise", json!({"amount": 8, "seed": 1})).unwrap();
    println!("document 6000x4000, {depth}-bit RGB");
    let cases: Vec<(String, serde_json::Value)> = vec![
        ("ring 100 px".into(), json!({"points": ring(3000.0, 2000.0, 60.0), "size": 16})),
        ("ring 300 px".into(), json!({"points": ring(3000.0, 2000.0, 160.0), "size": 24})),
        ("ring 1000 px".into(), json!({"points": ring(3000.0, 2000.0, 510.0), "size": 32})),
        ("line 3000 px, 4 px wide".into(), json!({"points": [[1500, 1000], [4500, 3000]], "size": 2})),
    ];
    for (name, p) in cases {
        let t = Instant::now();
        s.execute("paint.remove", p).unwrap();
        println!("{name:<26} {:>8.1?}", t.elapsed());
        s.execute("edit.undo", json!({})).unwrap();
    }
}
