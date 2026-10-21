//! A dense 24 MP document with 50 COW snapshots: trim alone and record + trim.
//! Run with `cargo run --release -p photocraft-ops --example bench_history_trim`.
//! Add `-- --layered` for 150 layers sharing the initial photo (about 2.8 M tile visits).
use photocraft_doc::{Color, ColorMode, Document, Layer, SampleType, Size};
use photocraft_ops::{History, LayerTarget};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let layers = if std::env::args().any(|a| a == "--layered") { 150 } else { 1 };
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut current = Arc::new(Document::with_background("benchmark", Size::new(6000, 4000), ColorMode::Rgb, depth, Color::WHITE));
        let doc = Arc::make_mut(&mut current);
        if let Some(surface) = doc.layers.first_mut().and_then(|l| l.surface_mut()) {
            // Make every photo tile unique, rather than benchmarking a shared solid fill.
            for y in (0..4000).step_by(256) {
                for x in (0..6000).step_by(256) {
                    surface.write_pixel(x, y, &[0.25, 0.5, 0.75, 1.0]);
                }
            }
            let photo = surface.clone();
            for _ in 1..layers {
                let mut layer = Layer::raster("Photo", photo.format());
                if let Some(surface) = layer.surface_mut() {
                    *surface = photo.clone();
                }
                doc.layers.push(layer);
            }
        }
        let mut history = History::new(50);
        history.max_bytes = usize::try_from(8u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX);
        for i in 0..50 {
            let before = current.clone();
            let doc = Arc::make_mut(&mut current);
            if let Some(surface) = doc.layers.first_mut().and_then(|l| l.surface_mut()) {
                surface.write_pixel(i * 100, 10, &[0.0, 0.0, 0.0, 1.0]);
            }
            history.record("Paint", before, LayerTarget::default());
            history.trim(&current);
        }
        for record in [false, true] {
            let mut samples = Vec::new();
            for _ in 0..31 {
                let mut h = history.clone();
                let t = Instant::now();
                if record {
                    h.record("Rename", current.clone(), LayerTarget::default());
                }
                black_box(h.trim(black_box(&current)));
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{depth:?} {layers} layer(s) {}: p50={:.6} ms p95={:.6} ms",
                if record { "record + trim" } else { "trim" },
                samples.get(15).copied().unwrap_or_default(),
                samples.get(29).copied().unwrap_or_default()
            );
        }
    }
}
