//! Chisel bevel benchmark: the cost of changing the bevel size, the way a dragged slider does.
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example bevel_bench -- [--size 4000x3000] [--reps 5] [--scene ellipse|text] [--no-cpu] [--json out.json]
//! ```
//!
//! Two synthetic scenes on a document of the given size: one big anti-aliased ellipse (a smooth
//! outline, few edge pixels) and a paragraph of text (many thin, anti-aliased edges). The layer
//! carries a Chisel Hard inner bevel; every case first renders it one pixel smaller (untimed,
//! warm caches), then times `--reps` refreshes of the whole document, each with the size one
//! pixel larger than the last, so the GPU canvas rebuilds its distance fields every time (it
//! keeps them while the size shrinks) and the CPU compositor misses its map cache. Sizes 8, 50
//! and 250 px. Rows are named `bevel size change to N, <scene> (GPU|CPU)`; `cargo xtask perf`
//! reads them through `perf/budgets.toml`.

use std::time::Instant;

use eframe::wgpu;
use photocraft_color::{BlendMode, Color};
use photocraft_doc::{Bevel, BevelStyle, BevelTechnique, Contour, Document, Effect, FxCommon, LayerContent};
use photocraft_geom::Rect;
use serde_json::json;

const SIZES: [u32; 3] = [8, 50, 250];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// A Chisel Hard inner bevel of the given size (the light comes from the effect's own angle).
fn bevel(size: f32) -> Effect {
    Effect::BevelEmboss(Bevel {
        enabled: true,
        style: BevelStyle::InnerBevel,
        technique: BevelTechnique::ChiselHard,
        depth: 1.5,
        up: true,
        size,
        soften: 0.0,
        angle: 120.0,
        altitude: 32.0,
        use_global_light: false,
        gloss_contour: Contour::Linear,
        highlight: FxCommon::new(BlendMode::Screen, 0.75),
        highlight_color: Color::WHITE,
        shadow: FxCommon::new(BlendMode::Multiply, 0.75),
        shadow_color: Color::BLACK,
        contour: None,
        texture: None,
    })
}

/// The scene's document through the engine, so the shape and the glyphs are the real thing.
fn scene(kind: &str, w: u32, h: u32) -> Result<Document, String> {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "background": "#1E1813"})).map_err(|e| format!("file.new: {e}"))?;
    let (w, h) = (w as i64, h as i64);
    match kind {
        "ellipse" => {
            s.execute("shape.create", json!({"kind": "ellipse", "rect": [w / 16, h / 12, w * 7 / 8, h * 5 / 6], "fill": "#C9A45C"}))
                .map_err(|e| format!("shape.create: {e}"))?;
        }
        _ => {
            let text = "Chisel bevels follow every curve of a letter, ".repeat(14);
            s.execute(
                "type.create",
                json!({"x": w / 25, "y": h / 10, "box": [w / 25, h / 25, w * 23 / 25, h * 23 / 25], "text": text, "size": (h / 20).max(12), "color": "#E8D9B0"}),
            )
            .map_err(|e| format!("type.create: {e}"))?;
        }
    }
    s.active().map(|a| (*a.doc).clone()).ok_or_else(|| "no active document".to_string())
}

/// Sets the bevel on the scene's one effect layer.
fn set_bevel(doc: &mut Document, size: f32) {
    for l in &mut doc.layers {
        if matches!(l.content, LayerContent::Shape(_) | LayerContent::Text(_)) {
            l.effects.items = vec![bevel(size)];
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    comp: photocraft_gpu::Compositor,
    adapter: String,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
            .ok()?;
    eprintln!("adapter: {:?}", adapter.get_info().name);
    // The limits the app requests (see `gpu_canvas::use_adapter_limits`).
    let limits = photocraft_ui_egui::gpu_canvas::device_limits(&adapter);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).ok()?;
    let comp = photocraft_gpu::Compositor::new(&device);
    Some(Gpu { device, queue, comp, adapter: adapter.get_info().name })
}

fn gpu_time(g: &mut Gpu, doc: &Document, region: Rect) -> Result<f64, String> {
    let t = Instant::now();
    g.comp.render(&g.device, &g.queue, doc, region, |_, _| {}).map_err(|e| e.to_string())?;
    let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    Ok(t.elapsed().as_secs_f64() * 1000.0)
}

fn cpu_time(doc: &Document, region: Rect) -> f64 {
    let t = Instant::now();
    let b = photocraft_compose::render(doc, region);
    std::hint::black_box(&b);
    t.elapsed().as_secs_f64() * 1000.0
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) = arg(&args, "--size").and_then(|s| s.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).unwrap_or((4000u32, 3000u32));
    let reps: usize = arg(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let no_cpu = args.iter().any(|a| a == "--no-cpu");
    let only = arg(&args, "--scene");
    let mut g = gpu();
    let adapter = g.as_ref().map(|g| g.adapter.clone());
    println!("document {w}x{h} ({:.1} MP), {reps} reps per case, adapter {}", f64::from(w) * f64::from(h) / 1e6, adapter.as_deref().unwrap_or("none"));
    println!("{:<44} {:>18} {:>18}   (median ms)", "case", "GPU", "CPU");

    // `--json out.json` (`cargo xtask perf`): one row per case and compositor, with samples and the peak RSS.
    let rss = photocraft_testkit::perf::RssSampler::start(std::time::Duration::from_millis(2));
    let mut json_rows: Vec<serde_json::Value> = Vec::new();
    let full = Rect::new(0, 0, w as i32, h as i32);
    for kind in ["ellipse", "text"] {
        if only.as_deref().is_some_and(|o| o != kind) {
            continue;
        }
        let mut doc = match scene(kind, w, h) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{kind}: {e}");
                std::process::exit(1);
            }
        };
        for size in SIZES {
            let what = format!("bevel size change to {size}, {kind}");
            // Warm up one pixel smaller, untimed, so only the size change is measured.
            set_bevel(&mut doc, (size - 1) as f32);
            if let Some(g) = g.as_mut() {
                let _ = gpu_time(g, &doc, full);
            }
            if !no_cpu {
                let _ = cpu_time(&doc, full);
            }
            rss.reset();
            let (mut gs, mut cs, mut gerr) = (Vec::new(), Vec::new(), None);
            for i in 0..reps {
                set_bevel(&mut doc, (size + i as u32) as f32);
                if let Some(g) = g.as_mut() {
                    match gpu_time(g, &doc, full) {
                        Ok(v) => gs.push(v),
                        Err(e) => gerr = Some(e),
                    }
                }
                if !no_cpu {
                    cs.push(cpu_time(&doc, full));
                }
            }
            let peak = rss.peak();
            if !gs.is_empty() {
                json_rows.push(photocraft_testkit::perf::row(&format!("{what} (GPU)"), &gs, peak, None));
            }
            if !cs.is_empty() {
                json_rows.push(photocraft_testkit::perf::row(&format!("{what} (CPU)"), &cs, peak, None));
            }
            let show = |v: &mut Vec<f64>, err: Option<&String>| match (v.is_empty(), err) {
                (false, _) => format!("{:.1}", median(v)),
                (true, Some(e)) => format!("unsupported ({e})"),
                (true, None) => "-".to_string(),
            };
            println!("{what:<44} {:>18} {:>18}", show(&mut gs, gerr.as_ref()), show(&mut cs, None));
        }
    }
    if let Some(out) = arg(&args, "--json") {
        let context = json!({"width": w, "height": h, "gpu_adapter": adapter});
        let report = photocraft_testkit::perf::report("bevel_bench", context, json_rows, Some(&rss));
        if let Err(e) = photocraft_testkit::perf::write_report(&out, &report) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
