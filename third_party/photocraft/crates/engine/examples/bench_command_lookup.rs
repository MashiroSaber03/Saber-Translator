//! Registry lookup cost, separate from command execution and canvas refresh.
//! Run with `cargo run --release -p photocraft-engine --example bench_command_lookup`.
use photocraft_engine::commands;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let specs = commands::command_specs();
    let mut ids: Vec<_> = specs.iter().map(|s| s.id).collect();
    ids.extend(std::iter::repeat_n("missing.command", 100));
    for (name, indexed) in [("linear reference", false), ("commands::find", true)] {
        let mut samples = Vec::new();
        for _ in 0..31 {
            let t = Instant::now();
            for _ in 0..100 {
                for id in &ids {
                    let id = black_box(*id);
                    black_box(if indexed { commands::find(id) } else { specs.iter().find(|s| s.id == id) });
                }
            }
            samples.push(t.elapsed().as_secs_f64() * 10.0);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{name}: {} ids/batch, p50={:.6} ms p95={:.6} ms",
            ids.len(),
            samples.get(15).copied().unwrap_or_default(),
            samples.get(29).copied().unwrap_or_default()
        );
    }
}
