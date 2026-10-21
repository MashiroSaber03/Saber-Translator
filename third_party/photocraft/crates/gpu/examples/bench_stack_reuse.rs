//! Dense 24-MP stack, synchronized GPU rendering with/without lower-stack reuse.
//! `cargo run --release -p photocraft-gpu --example bench_stack_reuse [-- --quick]`
use std::time::Instant;

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::Size;
use photocraft_gpu::{Compositor, Stats};
use photocraft_raster::Surface;

fn render(comp: &mut Compositor, device: &wgpu::Device, queue: &wgpu::Queue, doc: &Document) -> Result<Stats, Box<dyn std::error::Error>> {
    let stats = comp.render(device, queue, doc, doc.bounds(), |_, _| {})?;
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
    Ok(stats)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let quick = std::env::args().any(|a| a == "--quick");
    let size = if quick { Size::new(1024, 768) } else { Size::new(6000, 4000) };
    let adapter = pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
    let format = Compositor::preferred_acc_format(&adapter);
    println!("adapter={:?}, accumulation={format:?}, size={}x{}", adapter.get_info(), size.width, size.height);
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut doc = Document::new("stack", size, ColorMode::Rgb, depth);
        let mut surface = Surface::new(doc.pixel_format());
        surface.fill_rect(doc.bounds(), &[0.6, 0.2, 0.4, 0.5]);
        for i in 0..12 {
            // The immutable dense starting tiles are shared on the CPU, but each layer has
            // its own resident GPU pages and composite passes. No raster-generation timing.
            let mut layer = Layer::new(format!("layer {i}"), LayerContent::Raster(surface.clone()));
            layer.opacity = 0.8;
            doc.layers.push(layer);
        }
        for target in [11, 5] {
            for budget in [0, 512 << 20] {
                let mut comp = Compositor::try_new_with_format(&device, format)?;
                comp.set_composite_cache_budget(budget);
                let cold = Instant::now();
                render(&mut comp, &device, &queue, &doc)?;
                let cold_ms = cold.elapsed().as_secs_f64() * 1000.0;
                doc.layers.get_mut(target).ok_or("missing target layer")?.opacity = 0.45;
                let adapt = Instant::now();
                render(&mut comp, &device, &queue, &doc)?;
                let adapt_ms = adapt.elapsed().as_secs_f64() * 1000.0;
                let mut samples = Vec::new();
                let mut stats = Stats::default();
                for i in 0..11 {
                    doc.layers.get_mut(target).ok_or("missing target layer")?.opacity = if i % 2 == 0 { 0.7 } else { 0.4 };
                    let start = Instant::now();
                    stats = render(&mut comp, &device, &queue, &doc)?;
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                samples.sort_by(f64::total_cmp);
                println!(
                    "{depth:?}, target={target}, reuse={}, cold={cold_ms:.3} ms adapt={adapt_ms:.3} ms p50={:.3} ms p95={:.3} ms cache={} bytes hits={} skipped={}",
                    budget != 0,
                    samples.get(5).copied().unwrap_or_default(),
                    samples.get(10).copied().unwrap_or_default(),
                    comp.composite_cache_bytes(),
                    stats.prefix_hits,
                    stats.prefix_passes_skipped,
                );
            }
        }
    }
    Ok(())
}
