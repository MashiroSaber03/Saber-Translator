//! End-to-end timing of the smart selection commands (including sampling the layer and storing
//! the selection / layer) on a 6016×6016 8-bit document: a textured disc on a noisy background.
//! `cargo run --release -p photocraft-engine --example bench_smartselect_cmds [size]`
use std::time::Instant;

use photocraft_algo::segment::Rng;
use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::json;

fn main() {
    let n: i32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6016);
    let (c, rad) = (n as f32 / 2.0, n as f32 / 3.0);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": n, "height": n})).unwrap();
    s.edit("fill", |doc, _| {
        let bg = doc.layers[0].surface_mut().unwrap();
        let ch = bg.channels();
        for band in (0..n).step_by(256) {
            let y1 = (band + 256).min(n);
            let mut data = Vec::with_capacity((n * (y1 - band)) as usize * ch);
            for y in band..y1 {
                let mut rng = Rng::new(y as u64 + 1);
                for x in 0..n {
                    let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
                    let noise = 0.05 * rng.normal();
                    let p = if dx * dx + dy * dy <= rad * rad {
                        let t = ((x + y) as f32 * 0.05).sin() * 0.5 + 0.5;
                        [0.85 + 0.05 * t + noise, 0.2 + 0.3 * t + noise, 0.15 + noise]
                    } else {
                        [0.25 + noise, 0.45 + noise, 0.7 + noise]
                    };
                    data.extend(p.iter().map(|v| v.clamp(0.0, 1.0)).take(ch.min(3)));
                    if ch == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(Rect::new(0, band, n, y1), &data);
        }
        Ok(())
    })
    .unwrap();
    let time = |s: &mut Session, label: &str, id: &str, p: serde_json::Value| {
        let t = Instant::now();
        let r = s.execute(id, p).unwrap();
        println!("{label}: {:.2?}  {r}", t.elapsed());
    };
    let x = c + rad - 60.0;
    time(&mut s, "select.quick (brush 30)", "select.quick", json!({"points": [[x, c - 30.0], [x, c + 30.0]], "size": 30}));
    time(&mut s, "select.quick (brush 30, add)", "select.quick", json!({"points": [[x - 100.0, c]], "size": 30}));
    let m = (rad * 1.15) as i32;
    time(&mut s, "select.object", "select.object", json!({"rect": [c as i32 - m, c as i32 - m, 2 * m, 2 * m]}));
    time(&mut s, "select.refineEdge (radius 20)", "select.refineEdge", json!({"radius": 20}));
    s.execute("edit.undo", json!({})).unwrap();
    time(&mut s, "select.refineEdge (radius 20, newLayerWithMask + decontaminate)", "select.refineEdge", json!({"radius": 20, "decontaminate": true}));
    time(&mut s, "select.subject", "select.subject", json!({}));
    time(&mut s, "select.focusArea", "select.focusArea", json!({}));
}
