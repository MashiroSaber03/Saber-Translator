//! Layer-effect compositing benchmark: CPU (`photocraft-compose`) vs GPU (`photocraft-gpu`).
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example fx_bench -- [--size 7360x4912] [--psd file.psd]
//! ```
//!
//! Builds a synthetic document (default 36 MP) with text layers carrying drop shadow + stroke +
//! bevel, a painted raster layer with effects and an adjustment layer on top, then times a full
//! refresh (cold and warm effect caches) and the incremental cases the canvas sees: an adjustment
//! tweak (full refresh, maps cached), a brush dab on a plain layer and a brush dab on the effect
//! layer (damage rect grown by the effect reach). `--psd` times a real file instead.

use std::time::Instant;

use eframe::wgpu;
use photocraft_color::{BlendMode, Color, ColorMode, SampleType};
use photocraft_doc::{
    Adjustment, Bevel, BevelStyle, BevelTechnique, Contour, Document, Effect, FxCommon, FxPaint, Layer, LayerContent, StrokeFx, StrokePosition,
};
use photocraft_geom::{Rect, Size};
use serde_json::json;

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

fn effects() -> Vec<Effect> {
    let Effect::DropShadow(mut ds) = Effect::default_drop_shadow() else { unreachable!() };
    ds.distance = 12.0;
    ds.size = 16.0;
    vec![
        Effect::DropShadow(ds),
        Effect::Stroke(StrokeFx {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            size: 6.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::rgb(0.95, 0.85, 0.2)),
        }),
        Effect::BevelEmboss(Bevel {
            enabled: true,
            style: BevelStyle::InnerBevel,
            technique: BevelTechnique::Smooth,
            depth: 1.0,
            up: true,
            size: 10.0,
            soften: 2.0,
            angle: 120.0,
            altitude: 30.0,
            use_global_light: true,
            gloss_contour: Contour::Linear,
            highlight: FxCommon::new(BlendMode::Screen, 0.75),
            highlight_color: Color::WHITE,
            shadow: FxCommon::new(BlendMode::Multiply, 0.75),
            shadow_color: Color::BLACK,
            contour: None,
            texture: None,
        }),
    ]
}

fn synthetic(w: u32, h: u32) -> Document {
    let mut d = Document::with_background("fx", Size::new(w, h), ColorMode::Rgb, SampleType::U8, Color::rgb(0.82, 0.86, 0.9));
    // Some colour blocks on the background.
    for i in 0..12 {
        let x = (i * 613 % w as i32).max(0);
        let y = (i * 389 % h as i32).max(0);
        let c = [(i as f32 * 0.13) % 1.0, (i as f32 * 0.29) % 1.0, (i as f32 * 0.41) % 1.0, 1.0];
        d.layers[0].surface_mut().unwrap().fill_rect(Rect::new(x, y, x + w as i32 / 5, y + h as i32 / 6), &c);
    }
    let mut paint = Layer::raster("paint", d.pixel_format());
    paint.surface_mut().unwrap().fill_rect(Rect::new(10, 10, 20, 20), &[0.0, 0.0, 0.0, 1.0]);
    d.layers.push(paint);
    let mut blob = Layer::raster("blob", d.pixel_format());
    let (bw, bh) = (w as i32 / 4, h as i32 / 4);
    for k in 0..40 {
        let r = Rect::new(w as i32 / 2 - bw / 2 + k * 9, h as i32 / 2 - bh / 2 + k * 5, w as i32 / 2 + bw / 2 - k * 4, h as i32 / 2 + bh / 2 - k * 7);
        blob.surface_mut().unwrap().fill_rect(r, &[0.2 + k as f32 * 0.01, 0.4, 0.8, 1.0]);
    }
    blob.effects.items = effects();
    d.layers.push(blob);

    // Text layers through the engine (real glyph rasterization).
    let mut s = photocraft_engine::Session::new();
    s.add_document(d, None);
    let lines = ["Photocraft", "Layer Effects", "on the GPU", "Drop Shadow", "Stroke", "Bevel & Emboss", "36 megapixels", "interactive"];
    let rows = lines.len() as u32;
    for (i, t) in lines.iter().enumerate() {
        let y = (h / (rows + 1)) * (i as u32 + 1);
        let r = s.execute(
            "type.create",
            json!({"x": (w / 12) as i32 + (i as i32 % 3) * 300, "y": y, "text": t, "size": (h / rows / 2).max(12), "color": "#d04020"}),
        );
        if let Err(e) = r {
            eprintln!("type.create: {e}");
        }
    }
    let mut doc = (*s.active().unwrap().doc).clone();
    for l in &mut doc.layers {
        if matches!(l.content, LayerContent::Text(_)) {
            l.effects.items = effects();
        }
    }
    let mut adj = Layer::new(
        "curves",
        LayerContent::Adjustment(Adjustment::HueSaturation {
            hue: 10.0,
            saturation: 10.0,
            lightness: 0.0,
            colorize: false,
            ranges: photocraft_doc::adjust::HueRange::defaults(),
        }),
    );
    adj.opacity = 0.9;
    doc.layers.push(adj);
    doc
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

/// `--compare a.psd b.psd …`: GPU vs CPU max difference (premultiplied, /255) per file.
fn compare(files: &[String]) {
    let Some(mut g) = gpu() else { return };
    let mut worst_all = 0.0f32;
    for p in files {
        let Ok(bytes) = std::fs::read(p) else { continue };
        let Ok(r) = photocraft_io::import(p, &bytes) else {
            println!("{p}: import failed");
            continue;
        };
        let doc = r.document;
        let fx = doc.walk().iter().filter(|(_, _, l)| photocraft_compose::effects::has_effects(l)).count();
        let cpu = photocraft_compose::flatten(&doc);
        match photocraft_gpu::render_to_vec(&mut g.comp, &g.device, &g.queue, &doc, doc.bounds()) {
            Ok(out) => {
                let mut worst = (0.0f32, 0usize);
                for (i, (c, o)) in cpu.px.iter().zip(&out).enumerate() {
                    for k in 0..4 {
                        let (a, b) = if k == 3 { (c[3], o[3]) } else { (c[k] * c[3], o[k] * o[3]) };
                        let d = (a - b).abs();
                        if d > worst.0 || d.is_nan() {
                            worst = (if d.is_nan() { 9.0 } else { d }, i);
                        }
                    }
                }
                let w = doc.size.width as usize;
                worst_all = worst_all.max(worst.0);
                println!("{:<70} {fx:>2} fx layers  max diff {:6.2}/255 at ({},{})", p, worst.0 * 255.0, worst.1 % w, worst.1 / w);
                if std::env::var_os("FX_DUMP").is_some() && worst.0 > 1.0 / 255.0 {
                    println!("    cpu {:?} gpu {:?}", cpu.px[worst.1], out[worst.1]);
                    for (_, depth, l) in doc.walk() {
                        println!(
                            "    {}{} {:?} op {} fill {} clipped {} visible {} fx {:?}",
                            "  ".repeat(depth),
                            l.name,
                            l.blend,
                            l.opacity,
                            l.fill_opacity,
                            l.clipped,
                            l.visible,
                            l.effects.items.iter().map(|e| e.label()).collect::<Vec<_>>()
                        );
                    }
                }
            }
            Err(e) => println!("{p:<70} {fx:>2} fx layers  CPU fallback: {e}"),
        }
    }
    println!("worst over all files: {:.2}/255", worst_all * 255.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--compare") {
        compare(&args[i + 1..]);
        return;
    }
    let no_cpu = args.iter().any(|a| a == "--no-cpu");
    let mut doc = if let Some(p) = arg(&args, "--psd") {
        let bytes = std::fs::read(&p).expect("read --psd");
        photocraft_io::import(&p, &bytes).expect("import").document
    } else {
        let (w, h) = arg(&args, "--size").and_then(|s| s.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).unwrap_or((7360, 4912));
        synthetic(w, h)
    };
    let full = doc.bounds();
    println!("document {}x{} ({} layers)", doc.size.width, doc.size.height, doc.walk().len());
    let mut g = gpu();
    let adapter = g.as_ref().map(|g| g.adapter.clone());
    let mut gt = |doc: &Document, r: Rect| match g.as_mut() {
        Some(g) => gpu_time(g, doc, r),
        None => Err("no adapter".into()),
    };

    let reps: usize = arg(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let stats = |v: &mut Vec<f64>| -> Option<(f64, f64)> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (!v.is_empty()).then(|| (v[0], v[v.len() / 2]))
    };
    let show = |what: &str, r: Rect, c: Option<(f64, f64)>, g: Result<(f64, f64), String>| {
        let mp = r.width() as f64 * r.height() as f64 / 1e6;
        let c = c.map_or("-".to_string(), |(m, med)| format!("{m:8.1} / {med:8.1} ms"));
        let g = match g {
            Ok((m, med)) => format!("{m:7.1} / {med:7.1} ms"),
            Err(e) => format!("unsupported ({e})"),
        };
        println!("{what:<42} {mp:6.2} MP   cpu {c:>22}   gpu {g}   (min / median)");
    };
    // `--json out.json` (`cargo xtask perf`): one row per case and compositor, with samples and
    // the peak RSS while the case ran.
    let rss = photocraft_testkit::perf::RssSampler::start(std::time::Duration::from_millis(2));
    let mut json_rows: Vec<serde_json::Value> = Vec::new();
    // Each case: `edit` mutates the document, then both compositors refresh `region`.
    let mut case = |doc: &mut Document, what: &str, n: usize, edit: &mut dyn FnMut(&mut Document, usize) -> Rect, cpu_too: bool| {
        rss.reset();
        let (mut cs, mut gs, mut gerr) = (Vec::new(), Vec::new(), None);
        let mut region = full;
        for i in 0..n {
            region = edit(doc, i).intersect(&full);
            if cpu_too && !no_cpu {
                cs.push(cpu_time(doc, region));
            }
            match gt(doc, region) {
                Ok(v) => gs.push(v),
                Err(e) => gerr = Some(e),
            }
        }
        let peak = rss.peak();
        if !gs.is_empty() {
            json_rows.push(photocraft_testkit::perf::row(&format!("{what} (GPU)"), &gs, peak, None));
        }
        if !cs.is_empty() {
            json_rows.push(photocraft_testkit::perf::row(&format!("{what} (CPU)"), &cs, peak, None));
        }
        show(
            what,
            region,
            stats(&mut cs),
            match gerr {
                Some(e) => Err(e),
                None => Ok(stats(&mut gs).unwrap_or((0.0, 0.0))),
            },
        );
    };

    if args.iter().any(|a| a == "--baseline") {
        let mut plain = doc.clone();
        for l in &mut plain.layers {
            l.effects.items.clear();
        }
        case(&mut plain, "baseline: no effects, full refresh", reps, &mut |_, _| full, false);
    }
    case(
        &mut doc,
        "full refresh, cold effect caches",
        1,
        &mut |_, _| {
            photocraft_compose::purge_effect_cache();
            full
        },
        true,
    );
    case(&mut doc, "full refresh, warm caches", reps.min(3), &mut |_, _| full, true);
    // Adjustment tweak (maps unaffected).
    case(
        &mut doc,
        "adjustment tweak (full refresh)",
        reps.min(3),
        &mut |d, _| {
            if let Some(LayerContent::Adjustment(Adjustment::HueSaturation { hue, .. })) = d.layers.last_mut().map(|l| &mut l.content) {
                *hue += 5.0;
            }
            full
        },
        true,
    );
    // Brush dabs on a plain layer and on the effect layer: the canvas refreshes the damage rect
    // grown by the effect reach (`canvas::effect_reach`).
    let margin = doc.walk().iter().map(|(_, _, l)| photocraft_compose::effects::margin(l)).max().unwrap_or(0);
    let (w, h) = (full.width() as i32, full.height() as i32);
    case(
        &mut doc,
        "dab on a plain layer (damage + reach)",
        reps,
        &mut |d, i| {
            let r = Rect::new(w / 3 + i as i32 * 70, h / 3, w / 3 + i as i32 * 70 + 64, h / 3 + 64);
            if let Some(l) = d.layers.iter_mut().find(|l| l.name == "paint") {
                l.surface_mut().unwrap().fill_rect(r, &[0.1, 0.9, 0.1, 1.0]);
            }
            r.inflate(margin)
        },
        true,
    );
    // Moving a text layer by whole pixels: its effect maps move with it.
    case(
        &mut doc,
        "move a text layer 7 px (old + new bounds)",
        reps,
        &mut |d, _| {
            let Some(l) = d.layers.iter_mut().find(|l| matches!(l.content, LayerContent::Text(_))) else { return Rect::EMPTY };
            let LayerContent::Text(t) = &mut l.content else { return Rect::EMPTY };
            let Some(src) = t.cache.as_ref() else { return Rect::EMPTY };
            let b = src.content_bounds();
            let to = Rect::new(b.x0 + 7, b.y0, b.x1 + 7, b.y1);
            let mut moved = photocraft_raster::Surface::new(src.format());
            moved.write_region(to, &src.read_region(b));
            t.cache = Some(moved);
            b.union(&to).inflate(margin)
        },
        true,
    );
    case(
        &mut doc,
        "dab on the effect layer (damage + reach)",
        reps,
        &mut |d, i| {
            let r = Rect::new(w / 2 - 32 + i as i32 * 300, h / 2 - 32, w / 2 + 32 + i as i32 * 300, h / 2 + 32);
            if let Some(l) = d.layers.iter_mut().find(|l| l.name == "blob") {
                l.surface_mut().unwrap().fill_rect(r, &[0.9, 0.1, 0.1, 1.0]);
            }
            r.inflate(margin)
        },
        true,
    );
    if let Some(out) = arg(&args, "--json") {
        let context = json!({"width": w, "height": h, "gpu_adapter": adapter});
        let report = photocraft_testkit::perf::report("fx_bench", context, json_rows, Some(&rss));
        if let Err(e) = photocraft_testkit::perf::write_report(&out, &report) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
