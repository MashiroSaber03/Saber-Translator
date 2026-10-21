//! Timing: Pattern Stamp on a 24 MP (6016×4000) RGBA8 document.
//!
//! ```sh
//! cargo run --release -p photocraft-engine --example bench_pattern_stamp
//! ```
//!
//! One aligned, hard 200 px stroke along y=2000 from x=100 to x=5900, impressionist off, then
//! optionally the same stroke with impressionist on.

use std::time::Instant;

use photocraft_engine::Session;
use serde_json::json;

fn main() {
    let mut s = Session::new();
    if let Err(e) = s.execute("file.new", json!({"width": 6016, "height": 4000, "depth": 8, "mode": "rgb"})) {
        eprintln!("file.new: {e}");
        std::process::exit(1);
    }
    println!("document 6016x4000, 8-bit RGB (24.06 MP)");
    let stroke = json!({
        "points": [[100, 2000], [5900, 2000]],
        "size": 200,
        "hardness": 100,
        "pattern": "Checkerboard",
        "scale": 100,
        "aligned": true,
        "impressionist": false,
    });
    let t = Instant::now();
    match s.execute("paint.patternStamp", stroke.clone()) {
        Ok(r) => {
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("patternStamp aligned hardness 100 size 200, 5800 px stroke: {ms:.1} ms");
            println!("damage={}", r.get("damage").unwrap_or(&json!(null)));
        }
        Err(e) => {
            eprintln!("paint.patternStamp: {e}");
            std::process::exit(1);
        }
    }
    let mut impressionist = stroke;
    impressionist["impressionist"] = json!(true);
    impressionist["points"] = json!([[100, 2400], [5900, 2400]]);
    let t = Instant::now();
    match s.execute("paint.patternStamp", impressionist) {
        Ok(_) => {
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("patternStamp impressionist hardness 100 size 200, 5800 px stroke: {ms:.1} ms");
        }
        Err(e) => {
            eprintln!("paint.patternStamp impressionist: {e}");
            std::process::exit(1);
        }
    }
}
