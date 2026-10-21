//! Compare plain, single-axis and dual-axis live painting on a 24 MP document.
//! Run with `cargo run --release -p photocraft-engine --example bench_symmetry`.
use photocraft_engine::{Session, brush_cmds::LiveStroke};
use photocraft_paint::StrokePoint;
use serde_json::json;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for mode in ["off", "vertical", "dual"] {
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut s = Session::new();
            s.execute("file.new", json!({"width":6000,"height":4000,"depth":8,"mode":"rgb"}))?;
            if mode != "off" {
                s.execute("paint.setSymmetry", json!({"mode":mode}))?;
            }
            let params = json!({"points":[[1800,1200]],"size":60,"hardness":0.8,"opacity":0.5,"smoothing":0.0,"seed":7});
            let start = Instant::now();
            let mut live = LiveStroke::begin(&s, &params)?;
            for i in 1..=120 {
                let t = f64::from(i);
                live.push(&[StrokePoint::new(1800.0 + t * 2.0, 1200.0 + 20.0 * (t / 15.0).sin(), 1.0)])?;
            }
            std::hint::black_box(live.doc);
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        if let Some(median) = samples.get(2) {
            println!("{mode}: median {median:.2} ms / 121 live samples (24 MP, 60 px brush, 5 runs)");
        }
    }
    Ok(())
}
