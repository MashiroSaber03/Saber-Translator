//! Timing: Healing Brush and Spot Healing dab cost, and Patch and Content-Aware Move cost, on a
//! 6016×6016 document.
//! `cargo run --release -p photocraft-engine --example retouch_bench [side] [depth]`
//!
//! Each measurement is one single-dab stroke through the full command path (`Session::execute`:
//! undo snapshot, source sampling, solve/fill, composite), on a textured area of the canvas.
use std::time::Instant;

use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::json;

fn main() {
    let side: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6016);
    let depth: u64 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": side, "height": side, "depth": depth})).unwrap();
    // Texture a 2048² area in the middle (the rest stays white) with a gradient, noise and blemishes.
    let area = Rect::new(2000, 2000, 4048, 4048).intersect(&Rect::new(0, 0, side as i32, side as i32));
    s.edit("texture", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let mut data = Vec::with_capacity(area.width() as usize * area.height() as usize * 4);
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) as u32 % 1000) as f32 / 1000.0 * 0.08 - 0.04;
                let blemish = (x % 400 - 200).abs() < 8 && (y % 400 - 200).abs() < 8;
                let v =
                    if blemish { [0.05, 0.05, 0.05, 1.0] } else { [0.3 + 0.0002 * x as f32 + n, 0.4 + ((x + y) as f32 * 0.01).sin() * 0.1 + n, 0.5 + n, 1.0] };
                data.extend(photocraft_raster::from_rgba(&fmt, v));
            }
        }
        surf.write_region(area, &data);
        Ok(())
    })
    .unwrap();
    println!("document {side}x{side}, {depth}-bit RGB");
    let mut k = 0;
    let mut spot = || {
        k += 1;
        (2200 + (k % 4) * 400, 2200 + (k / 4 % 4) * 400)
    };
    for size in [50.0, 100.0, 200.0, 300.0] {
        let (x, y) = spot();
        let t = Instant::now();
        s.execute("paint.healingBrush", json!({"points": [[x, y]], "offset": [-150, -130], "size": size, "hardness": 50})).unwrap();
        println!("healingBrush                size {size:>4}: {:>8.1?}", t.elapsed());
    }
    for kind in ["contentAware", "proximityMatch", "createTexture"] {
        for size in [50.0, 100.0, 200.0, 300.0] {
            let (x, y) = spot();
            let t = Instant::now();
            s.execute("paint.spotHealing", json!({"points": [[x, y]], "size": size, "hardness": 50, "type": kind})).unwrap();
            println!("spotHealing {kind:<15} size {size:>4}: {:>8.1?}", t.elapsed());
        }
    }
    // A short multi-dab heal stroke (spacing 25 %): the whole stroke is solved at once.
    let t = Instant::now();
    s.execute("paint.healingBrush", json!({"points": [[2300, 3500], [2700, 3500]], "offset": [0, -300], "size": 100, "hardness": 50})).unwrap();
    println!("healingBrush stroke 400 px long, size 100: {:.1?}", t.elapsed());
    // Patch: an elliptical selection of each size, dragged 300 px; the whole selection is solved.
    for size in [100, 250, 500, 1000] {
        let (x, y) = (2300, 2300);
        s.execute("select.rect", json!({"x": x, "y": y, "width": size, "height": size, "ellipse": true})).unwrap();
        let t = Instant::now();
        s.execute("paint.patch", json!({"offset": [300, 200]})).unwrap();
        println!("patch ellipse {size:>4}×{size:<4}: {:>8.1?}", t.elapsed());
        // The Patch Tool's live preview of the same patch (coarse solve, 128² cells, at 100 % zoom).
        let t = Instant::now();
        photocraft_engine::retouch_cmds::patch_preview(&s, &json!({"offset": [300, 200]}), 128 * 128, 1).unwrap();
        println!("patch preview {size:>4}×{size:<4}: {:>8.1?}", t.elapsed());
    }
    // Content-Aware Move: the same selections dragged 300 px, then undone (both fills: the edge band
    // at the new place and the old place), and Extend at Color 5 (one fill, plus the colour fit).
    for size in [100, 250, 500, 1000] {
        for (mode, color) in [("move", 0), ("extend", 5)] {
            s.execute("select.rect", json!({"x": 2300, "y": 2300, "width": size, "height": size, "ellipse": true})).unwrap();
            let t = Instant::now();
            s.execute("paint.contentAwareMove", json!({"offset": [300, 200], "mode": mode, "color": color})).unwrap();
            println!("contentAwareMove {mode:<6} {size:>4}×{size:<4}: {:>8.1?}", t.elapsed());
            s.execute("edit.undo", json!({})).unwrap();
        }
    }
}
