//! End-to-end timing of `select.rect` (rectangular and elliptical marquee) on a 6000×4000
//! document. `cargo run --release -p photocraft-engine --example bench_marquee [reps]`
use std::time::{Duration, Instant};

use photocraft_engine::Session;
use serde_json::{Value, json};

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(5);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 6000, "height": 4000})).unwrap();
    let base = json!({"x": 1000, "y": 500, "width": 3000, "height": 2000});
    let rect = |x: i32, y: i32, w: i32, h: i32, extra: Value| {
        let mut p = json!({"x": x, "y": y, "width": w, "height": h});
        if let (Some(m), Some(e)) = (p.as_object_mut(), extra.as_object()) {
            m.extend(e.clone());
        }
        p
    };
    let cases: Vec<(&str, Option<Value>, Value)> = vec![
        ("rect replace 5800x3800", None, rect(100, 100, 5800, 3800, json!({}))),
        ("ellipse replace 5800x3800 aa", None, rect(100, 100, 5800, 3800, json!({"ellipse": true}))),
        ("ellipse replace 5800x3800 no aa", None, rect(100, 100, 5800, 3800, json!({"ellipse": true, "antiAlias": false}))),
        ("ellipse replace 1000x800 aa", None, rect(2000, 1500, 1000, 800, json!({"ellipse": true}))),
        ("ellipse off-canvas 8000x6000 aa", None, rect(-1000, -1000, 8000, 6000, json!({"ellipse": true}))),
        ("ellipse feather 20", None, rect(100, 100, 5800, 3800, json!({"ellipse": true, "feather": 20}))),
        ("rect add", Some(base.clone()), rect(2500, 1500, 3000, 2000, json!({"mode": "add"}))),
        ("ellipse add", Some(base.clone()), rect(2500, 1500, 3000, 2000, json!({"ellipse": true, "mode": "add"}))),
        ("ellipse subtract", Some(base.clone()), rect(2500, 1500, 3000, 2000, json!({"ellipse": true, "mode": "subtract"}))),
        ("ellipse intersect", Some(base.clone()), rect(2500, 1500, 3000, 2000, json!({"ellipse": true, "mode": "intersect"}))),
    ];
    for (label, pre, p) in cases {
        let mut best = Duration::MAX;
        for _ in 0..reps {
            s.execute("select.deselect", json!({})).ok();
            if let Some(pre) = &pre {
                s.execute("select.rect", pre.clone()).unwrap();
            }
            let t = Instant::now();
            s.execute("select.rect", p.clone()).unwrap();
            best = best.min(t.elapsed());
        }
        println!("{label:34} best of {reps}: {best:.2?}");
    }
}
