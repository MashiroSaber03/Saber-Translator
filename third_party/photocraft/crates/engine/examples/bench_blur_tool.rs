//! Reproducible soft Blur strokes on a 24 MP striped document. Run in release mode.
//! Setup, Undo and the output fingerprint are excluded from the timing.
use std::time::Instant;

use photocraft_engine::{EngineError, Session};
use photocraft_geom::Rect;
use serde_json::json;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    for depth in [8, 16, 32] {
        let mut session = Session::new();
        session.execute("file.new", json!({"width":6000,"height":4000,"depth":depth,"mode":"rgb"}))?;
        session.execute("layer.new.layer", json!({}))?;
        session.edit("stripes", |doc, active| {
            let id = active.ok_or_else(|| EngineError::Other("no layer".into()))?;
            let surface = doc.layer_mut(id).and_then(|l| l.surface_mut()).ok_or_else(|| EngineError::Other("no surface".into()))?;
            for x in (0..6000).step_by(64) {
                let color = if x % 128 == 0 { [0.9, 0.3, 0.1, 1.0] } else { [0.1, 0.4, 0.8, 1.0] };
                surface.fill_rect(Rect::new(x, 0, (x + 64).min(6000), 4000), &color);
            }
            Ok(())
        })?;
        for size in [100, 500, 1000] {
            let params = json!({"points":[[2500,2000],[3500,2000]],"size":size,"hardness":20,"strength":50,"spacing":25});
            let mut times = Vec::new();
            let mut fingerprint = 0u64;
            for sample in 0..6 {
                let start = Instant::now();
                session.execute("paint.blur", params.clone())?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if sample > 0 {
                    times.push(ms);
                }
                if sample == 5 {
                    let st = session.active().ok_or("no document")?;
                    let surface = st.doc.layer(st.active_layer.ok_or("no layer")?).and_then(|l| l.surface()).ok_or("no surface")?;
                    for y in (1500..2500).step_by(17) {
                        for x in (2000..4000).step_by(13) {
                            for channel in surface.rgba(x, y) {
                                fingerprint = fingerprint.wrapping_mul(16777619) ^ u64::from(channel.to_bits());
                            }
                        }
                    }
                }
                session.execute("edit.undo", json!({}))?;
            }
            times.sort_by(f64::total_cmp);
            println!(
                "depth={depth} size={size} median_ms={:.3} min_ms={:.3} max_ms={:.3} sampled_fingerprint={fingerprint:016x}",
                times[2], times[0], times[4]
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
