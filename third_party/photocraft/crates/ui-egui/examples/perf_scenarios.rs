//! Scorecard performance scenarios (#221): the measurable rows of the targets in #209, #210 and
//! #211, each timed through the command path the UI, CLI and MCP share (`Session::execute`) plus
//! the canvas refresh the app does after it (the app's own `GpuCanvas`, on a device created the
//! way the app creates it; the CPU compositor without an adapter).
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example perf_scenarios -- [--quick] [--json out.json] [--only text] [--group layered] [--reps N] [--cpu] [--no-composite-reuse]
//! ```
//!
//! `cargo xtask perf` runs this and maps the rows to scenario ids in `perf/budgets.toml`, so
//! keep row names stable. `--quick` uses small synthetic documents (seconds, for smoke runs);
//! its numbers are only comparable with other `--quick` runs. Every row records p50 / p95 / max
//! over its samples, the peak resident memory measured in-process while it ran, and the GPU
//! bytes the canvas holds for the document (texture + compositor) when there is a GPU.

use std::time::Instant;

use eframe::egui_wgpu::RenderState;
use photocraft_algo::liquify::{LiquifyField, LiquifyStroke, LiquifyTool, ProxyImage, auto_cell};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_engine::Session;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use photocraft_testkit::perf::{RssSampler, report, row, write_report};
use photocraft_ui_egui::gpu_canvas::GpuCanvas;
use serde_json::{Value, json};

type Res<T> = Result<T, String>;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn exec(s: &mut Session, id: &str, p: Value) -> Res<Value> {
    s.execute(id, p).map_err(|e| format!("{id}: {e}"))
}

fn active_layer(s: &Session) -> Res<u64> {
    s.active().and_then(|d| d.active_layer).map(|l| l.0).ok_or_else(|| "no active layer".to_string())
}

/// A photo-like RGB image (gradients, soft blobs, fine noise) with a high-contrast disc as the
/// subject, encoded as JPEG so it opens like a real photo.
fn photo_jpeg(w: u32, h: u32) -> Res<Vec<u8>> {
    use rayon::prelude::*;
    let mut px = vec![0u8; w as usize * h as usize * 3];
    let (cx, cy, r) = (w as f32 * 0.55, h as f32 * 0.5, w.min(h) as f32 / 3.0);
    px.par_chunks_mut(w as usize * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w as usize {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 997) as f32 / 997.0 - 0.5;
            let blob = (-(((fx - 0.4) * 3.0).powi(2) + ((fy - 0.55) * 4.0).powi(2))).exp();
            let inside = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() < r;
            let (r0, g0, b0) = if inside { (0.85, 0.3 + 0.2 * (fx * 40.0).sin(), 0.15) } else { (0.35, 0.45, 0.55) };
            let rgb = [r0 + 0.1 * blob + n * 0.08, g0 + 0.1 * blob + n * 0.08, b0 + 0.2 * fy + n * 0.08];
            for (c, v) in rgb.into_iter().enumerate() {
                row[x * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    let img =
        photocraft_codecs::Image::from_raw(w, h, photocraft_codecs::ChannelLayout::Rgb, photocraft_codecs::SampleType::U8, px).map_err(|e| e.to_string())?;
    photocraft_codecs::encode(&img, photocraft_codecs::Format::Jpeg, &Default::default()).map_err(|e| e.to_string())
}

fn open_photo(w: u32, h: u32) -> Res<Session> {
    let jpeg = photo_jpeg(w, h)?;
    let doc = photocraft_io::import("photo.jpg", &jpeg).map_err(|e| e.to_string())?.document;
    let mut s = Session::new();
    s.open_document(doc, Some("photo.jpg".into()));
    Ok(s)
}

/// How far an edit's composite change reaches beyond its damage (layer effects); mirrors
/// `canvas::effect_reach`.
fn reach(layers: &[photocraft_doc::Layer]) -> i32 {
    layers
        .iter()
        .map(|l| {
            let own = if photocraft_compose::effects::has_effects(l) { photocraft_compose::effects::margin(l) } else { 0 };
            let kids = match &l.content {
                photocraft_doc::LayerContent::Group(g) => reach(&g.children),
                _ => 0,
            };
            own + kids
        })
        .max()
        .unwrap_or(0)
}

struct Bench {
    gpu: Option<(GpuCanvas, RenderState)>,
    sampler: RssSampler,
    reps: usize,
    only: Option<String>,
    rows: Vec<Value>,
    /// Rows that couldn't run: (name, error).
    errors: Vec<(String, String)>,
    /// The row being timed, so a panic inside it is reported against that row.
    current: Option<String>,
}

impl Bench {
    fn wanted(&self, name: &str) -> bool {
        self.only.as_ref().is_none_or(|o| name.contains(o.as_str()))
    }

    /// The canvas refresh after an edit (`canvas::ensure_gpu`): the damage rect grown by the
    /// effect reach when the session reports one, else everything; waits for the GPU.
    /// Re-composite only `r` of the active document (what the canvas hadn't shown yet).
    fn refresh_rect(&self, s: &Session, r: photocraft_geom::Rect) -> Res<f64> {
        let st = s.active().ok_or("no document")?;
        let doc = &st.doc;
        let r = r.intersect(&doc.bounds());
        let t = Instant::now();
        if !r.is_empty() {
            match &self.gpu {
                Some((g, rs)) => {
                    g.refresh(doc.id.0, doc, Some(r), None);
                    let _ = rs.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None });
                }
                None => {
                    std::hint::black_box(photocraft_compose::render(doc, r));
                }
            }
        }
        Ok(ms(t))
    }

    fn refresh(&self, s: &Session, full: bool) -> Res<f64> {
        let st = s.active().ok_or("no document")?;
        let doc = &st.doc;
        let damage = if full { None } else { st.last_damage.map(|r| r.inflate(reach(&doc.layers)).intersect(&doc.bounds())) };
        let t = Instant::now();
        match &self.gpu {
            Some((g, rs)) => {
                g.refresh(doc.id.0, doc, damage, None);
                let _ = rs.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
            None => {
                let r = damage.unwrap_or(doc.bounds()).intersect(&doc.bounds());
                if !r.is_empty() {
                    std::hint::black_box(photocraft_compose::render(doc, r));
                }
            }
        }
        Ok(ms(t))
    }

    /// Frame `i` of a Move tool drag of `layer` (#128): the canvas shows the document with the
    /// layer at the pointer (`layer_multi_cmds::moved`) and recomposites where it was in the
    /// previous frame and where it is now (`move_ui::damage`).
    fn drag_frame(&self, s: &Session, layer: u64, i: usize) -> Res<f64> {
        let doc = s.active().ok_or("no document")?.doc.clone();
        let id = photocraft_doc::LayerId(layer);
        let all = photocraft_geom::Rect::new(i32::MIN / 2, i32::MIN / 2, i32::MAX / 2, i32::MAX / 2);
        let was = doc.layer(id).and_then(|l| photocraft_compose::change_bounds(l, all)).ok_or("no bounds")?;
        // Frame i shows offset (d, d / 2); the canvas showed the previous frame's.
        let (d, prev) = (3 * i as i32 + 1, if i == 0 { 0 } else { 3 * i as i32 - 2 });
        let t = Instant::now();
        let shown = photocraft_engine::layer_multi_cmds::moved(&doc, &[id], d, d / 2).map_err(|e| e.to_string())?;
        let damage = was.translate(prev, prev / 2).union(&was.translate(d, d / 2)).inflate(1).intersect(&doc.bounds());
        match &self.gpu {
            Some((g, rs)) => {
                g.refresh(doc.id.0, &shown, Some(damage), None);
                let _ = rs.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
            None => {
                std::hint::black_box(photocraft_compose::render(&shown, damage));
            }
        }
        Ok(ms(t))
    }

    fn gpu_bytes(&self, s: &Session) -> Option<u64> {
        let (g, _) = self.gpu.as_ref()?;
        let tex = s.active().and_then(|d| g.texture_info(d.doc.id.0)).map_or(0, |(_, b)| b);
        let (resident, fx) = g.compositor_bytes().unwrap_or((0, 0));
        Some(tex + resident + fx as u64)
    }

    /// Times `f` `reps` times after one untimed warm-up run; `f` returns its own time in ms.
    fn time(&mut self, name: &str, s: &mut Session, reps: usize, warm: bool, mut f: impl FnMut(&Self, &mut Session, usize) -> Res<f64>) {
        if !self.wanted(name) {
            return;
        }
        self.current = Some(name.to_string());
        self.sampler.reset();
        let mut samples = Vec::new();
        let mut run = || -> Res<()> {
            if warm {
                f(self, s, 0)?;
            }
            for i in 0..reps.max(1) {
                samples.push(f(self, s, i + 1)?);
            }
            Ok(())
        };
        let result = run();
        self.record(name, s, &samples, result);
    }

    fn record(&mut self, name: &str, s: &Session, samples: &[f64], result: Res<()>) {
        match result {
            Ok(()) => {
                let r = row(name, samples, self.sampler.peak(), self.gpu_bytes(s));
                println!(
                    "{name:<52} p50 {:>9.2} ms  p95 {:>9.2} ms  max {:>9.2} ms  (n {}, peak RSS {} MB)",
                    r["p50_ms"].as_f64().unwrap_or(f64::NAN),
                    r["p95_ms"].as_f64().unwrap_or(f64::NAN),
                    r["max_ms"].as_f64().unwrap_or(f64::NAN),
                    samples.len(),
                    r["peak_rss_bytes"].as_u64().unwrap_or(0) >> 20,
                );
                self.rows.push(r);
            }
            Err(e) => {
                println!("{name:<52} ERROR {e}");
                self.errors.push((name.to_string(), e));
            }
        }
    }
}

/// A headless GPU canvas on a device created like the app's.
fn canvas() -> Option<(GpuCanvas, RenderState)> {
    let setup = photocraft_ui_egui::gpu_canvas::wgpu_setup();
    let rs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(setup, Default::default()))).ok()?;
    eprintln!("adapter: {}", rs.adapter.get_info().name);
    Some((GpuCanvas::new(&rs), rs))
}

/// Sizes per mode.
struct Sizes {
    /// The layered document of #209 (20 MP).
    layered: (u32, u32),
    /// A layer of 12 MP to duplicate and paste.
    big_layer: (u32, u32),
    /// Layers in the "150-layer document".
    many_layers: usize,
    /// The 16-bit document (15000 × 10000).
    huge16: (u32, u32),
    /// The 32-bit document with 20 layers.
    float32: (u32, u32),
    /// 10 MP for Content-Aware Fill and Spot Healing.
    ten_mp: (u32, u32),
    /// 24 MP for the selection algorithms and Content-Aware Scale.
    big_photo: (u32, u32),
    /// A4 at 300 ppi.
    a4: (u32, u32),
}

const FULL: Sizes = Sizes {
    layered: (5472, 3648),
    big_layer: (4243, 2828),
    many_layers: 150,
    huge16: (15000, 10000),
    float32: (4000, 3000),
    ten_mp: (3872, 2592),
    big_photo: (6000, 4000),
    a4: (2480, 3508),
};

const QUICK: Sizes = Sizes {
    layered: (1600, 1066),
    big_layer: (1200, 800),
    many_layers: 40,
    huge16: (2400, 1600),
    float32: (1200, 900),
    ten_mp: (1200, 800),
    big_photo: (1600, 1066),
    a4: (1240, 1754),
};

/// The layered document of #209: a background, six pixel layers with a drop shadow and a stroke,
/// four type layers with a drop shadow, and a Levels adjustment on top. Returns the session and
/// the id of the styled layer the edits target.
fn layered_doc(w: u32, h: u32) -> Res<(Session, u64)> {
    let mut s = Session::new();
    exec(&mut s, "file.new", json!({"width": w, "height": h, "background": "white"}))?;
    let mut target = 0;
    for i in 0..6u32 {
        exec(&mut s, "layer.new.layer", json!({"name": format!("panel {i}")}))?;
        let (x, y) = ((i % 3) * w / 3 + w / 24, (i / 3) * h / 2 + h / 16);
        exec(&mut s, "select.rect", json!({"x": x, "y": y, "width": w / 4, "height": h / 3, "ellipse": i % 2 == 1}))?;
        exec(&mut s, "edit.fill", json!({"color": format!("#{:02x}6040", 40 + i * 30)}))?;
        exec(&mut s, "select.deselect", json!({}))?;
        exec(&mut s, "layer.layerStyle.dropShadow", json!({"distance": 12, "size": 16}))?;
        exec(&mut s, "layer.layerStyle.stroke", json!({"size": 3, "color": "#202020", "add": true}))?;
        if i == 1 {
            target = active_layer(&s)?;
        }
    }
    for i in 0..4u32 {
        exec(&mut s, "type.create", json!({"x": w / 10 + i * w / 5, "y": h * 9 / 10, "text": format!("Headline {i}"), "size": (h / 30).max(12)}))?;
        exec(&mut s, "layer.layerStyle.dropShadow", json!({"distance": 6, "size": 8}))?;
    }
    exec(&mut s, "layer.newAdjustmentLayer.levels", json!({"inBlack": 8, "inWhite": 245, "gamma": 1.05}))?;
    exec(&mut s, "layer.select", json!({"layer": target}))?;
    Ok((s, target))
}

/// A document with `n` layers: small painted rects, every fifth a type layer, every seventh
/// styled.
fn many_layers_doc(w: u32, h: u32, n: usize) -> Res<(Session, u64)> {
    let mut s = Session::new();
    exec(&mut s, "file.new", json!({"width": w, "height": h, "background": "white"}))?;
    let mut mid = 0;
    for i in 0..n as u32 {
        let (x, y) = ((i * 397) % (w - w / 8), (i * 211) % (h - h / 8));
        if i % 5 == 4 {
            exec(&mut s, "type.create", json!({"x": x, "y": y + 40, "text": format!("Label {i}"), "size": 28}))?;
        } else {
            exec(&mut s, "layer.new.layer", json!({"name": format!("item {i}")}))?;
            exec(&mut s, "select.rect", json!({"x": x, "y": y, "width": w / 10, "height": h / 10}))?;
            exec(&mut s, "edit.fill", json!({"color": format!("#{:02x}{:02x}80", (i * 37) % 256, (i * 91) % 256)}))?;
            exec(&mut s, "select.deselect", json!({}))?;
        }
        if i % 7 == 3 {
            exec(&mut s, "layer.layerStyle.dropShadow", json!({"distance": 6, "size": 8}))?;
        }
        if i as usize == n / 2 {
            mid = active_layer(&s)?;
        }
    }
    exec(&mut s, "layer.select", json!({"layer": mid}))?;
    Ok((s, mid))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let sz = if quick { &QUICK } else { &FULL };
    let reps: usize = arg(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(if quick { 7 } else { 15 }).max(1);
    let gpu = if args.iter().any(|a| a == "--cpu") { None } else { canvas() };
    let no_reuse = args.iter().any(|a| a == "--no-composite-reuse");
    if no_reuse && let Some((canvas, _)) = &gpu {
        canvas.set_composite_cache_budget(0);
    }
    let adapter = gpu.as_ref().map(|(_, rs)| rs.adapter.get_info().name);
    let mut b = Bench {
        gpu,
        sampler: RssSampler::start(std::time::Duration::from_millis(2)),
        reps,
        only: arg(&args, "--only"),
        rows: Vec::new(),
        errors: Vec::new(),
        current: None,
    };
    println!("perf_scenarios: {} mode, {} reps, {}", if quick { "quick" } else { "full" }, reps, adapter.as_deref().unwrap_or("CPU canvas"));
    let t_all = Instant::now();

    // A panic in one group (a crash the scenario found) is reported against the row that was
    // running, and the other groups still run.
    type Group = fn(&mut Bench, &Sizes);
    let groups: [(&str, Group); 5] = [
        ("layered", layered_scenarios),
        ("many layers", many_layer_scenarios),
        ("bit depths", depth_scenarios),
        ("long operations", job_scenarios),
        ("algorithms", algorithm_scenarios),
    ];
    let use_gpu = b.gpu.is_some();
    if let Some(wanted) = arg(&args, "--group")
        && !groups.iter().any(|(group, _)| *group == wanted)
    {
        eprintln!("unknown scenario group: {wanted}");
        std::process::exit(2);
    }
    for (i, (group, run)) in groups.into_iter().enumerate() {
        if arg(&args, "--group").is_some_and(|wanted| wanted != group) {
            continue;
        }
        b.current = None;
        // Each group starts on a fresh canvas device (one document open, as in the app), so
        // GPU memory or an error left by one group never carries into the next.
        if use_gpu && i > 0 {
            b.gpu = None;
            b.gpu = canvas();
            if no_reuse && let Some((canvas, _)) = &b.gpu {
                canvas.set_composite_cache_budget(0);
            }
        }
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut b, sz))) {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into());
            let msg = format!("panicked: {}", msg.lines().filter(|l| !l.trim().is_empty()).collect::<Vec<_>>().join(" / "));
            let row = b.current.take().unwrap_or_else(|| group.to_string());
            println!("{row:<52} ERROR {msg}");
            b.errors.push((row, msg));
        }
    }

    println!("total {:.1} s", t_all.elapsed().as_secs_f64());
    if let Some(out) = arg(&args, "--json") {
        let context = json!({
            "mode": if quick { "quick" } else { "full" },
            "reps": reps,
            "gpu_adapter": adapter,
            "composite_reuse": !no_reuse,
            "group": arg(&args, "--group"),
            "errors": b.errors.iter().map(|(n, e)| json!({"name": n, "error": e})).collect::<Vec<_>>(),
        });
        let rep = report("perf_scenarios", context, std::mem::take(&mut b.rows), Some(&b.sampler));
        if let Err(e) = write_report(&out, &rep) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

/// #209 rows on the layered 20 MP document.
fn layered_scenarios(b: &mut Bench, sz: &Sizes) {
    let (w, h) = sz.layered;
    let (mut s, target) = match layered_doc(w, h) {
        Ok(v) => v,
        Err(e) => {
            b.errors.push(("layered document".into(), e));
            return;
        }
    };
    let _ = b.refresh(&s, true);
    let reps = b.reps;
    b.time("move styled layer 10 px + refresh", &mut s, reps, true, |b, s, i| {
        let d = if i % 2 == 0 { 10 } else { -10 };
        let t = Instant::now();
        exec(s, "layer.translate", json!({"layer": target, "dx": d, "dy": d / 2}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("move tool drag frame, styled layer", &mut s, reps, true, |b, s, i| b.drag_frame(s, target, i));
    b.time("set layer opacity + refresh", &mut s, reps, true, |b, s, i| {
        let t = Instant::now();
        exec(s, "layer.setProps", json!({"layer": target, "opacity": if i % 2 == 0 { 0.6 } else { 0.9 }}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("set layer fill + refresh", &mut s, reps, true, |b, s, i| {
        let t = Instant::now();
        exec(s, "layer.setProps", json!({"layer": target, "fill": if i % 2 == 0 { 0.5 } else { 1.0 }}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("set blend mode + refresh", &mut s, reps, true, |b, s, i| {
        let t = Instant::now();
        exec(s, "layer.setProps", json!({"layer": target, "blend": if i % 2 == 0 { "Multiply" } else { "Normal" }}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("toggle visibility + refresh", &mut s, reps, true, |b, s, i| {
        let t = Instant::now();
        exec(s, "layer.setProps", json!({"layer": target, "visible": i % 2 == 0}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    let _ = exec(&mut s, "layer.setProps", json!({"layer": target, "visible": true}));
    b.time("undo move + refresh", &mut s, reps, true, |b, s, _| {
        exec(s, "layer.translate", json!({"layer": target, "dx": 7, "dy": 3}))?;
        b.refresh(s, false)?;
        let t = Instant::now();
        exec(s, "edit.undo", json!({}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("redo opacity change + refresh", &mut s, reps, true, |b, s, i| {
        exec(s, "layer.setProps", json!({"layer": target, "opacity": 0.5 + 0.02 * i as f64}))?;
        exec(s, "edit.undo", json!({}))?;
        b.refresh(s, false)?;
        let t = Instant::now();
        exec(s, "edit.redo", json!({}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });

    // A layer filling the canvas, with a drop shadow (#761) and without (#128): dragged, it
    // overhangs the canvas edge.
    let full = (|| -> Res<u64> {
        exec(&mut s, "layer.new.layer", json!({"name": "full"}))?;
        exec(&mut s, "select.all", json!({}))?;
        exec(&mut s, "edit.fill", json!({"color": "#4080c0"}))?;
        exec(&mut s, "select.deselect", json!({}))?;
        exec(&mut s, "layer.layerStyle.dropShadow", json!({"distance": 12, "size": 16}))?;
        active_layer(&s)
    })();
    match full {
        Ok(full) => {
            let _ = b.refresh(&s, true);
            b.time("move tool drag frame, full-canvas layer with drop shadow", &mut s, reps, true, |b, s, i| b.drag_frame(s, full, i));
            let _ = exec(&mut s, "layer.layerStyle.clear", json!({"layer": full}));
            let _ = b.refresh(&s, true);
            b.time("move tool drag frame, full-canvas layer", &mut s, reps, true, |b, s, i| b.drag_frame(s, full, i));
            let _ = exec(&mut s, "layer.delete", json!({"layer": full}));
        }
        Err(e) => b.errors.push(("full-canvas styled layer".into(), e)),
    }

    // A 12 MP layer, duplicated and pasted: the first time (a fresh layer, cold caches) and
    // again (the same source).
    let (lw, lh) = sz.big_layer;
    let fresh_layer = |s: &mut Session| -> Res<u64> {
        exec(s, "layer.new.layer", json!({"name": "12 MP"}))?;
        exec(s, "select.rect", json!({"x": (w - lw) / 2, "y": (h - lh) / 2, "width": lw, "height": lh}))?;
        exec(s, "edit.fill", json!({"color": "#3070a0"}))?;
        exec(s, "select.deselect", json!({}))?;
        active_layer(s)
    };
    let reps_big = reps.min(5);
    let mut src = 0;
    b.time("duplicate 12 MP layer (first) + refresh", &mut s, reps_big, false, |b, s, _| {
        src = fresh_layer(s)?;
        b.refresh(s, false)?;
        let t = Instant::now();
        exec(s, "layer.duplicate", json!({"layer": src}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("duplicate 12 MP layer (repeat) + refresh", &mut s, reps_big, false, |b, s, _| {
        let t = Instant::now();
        exec(s, "layer.duplicate", json!({"layer": src}))?;
        let v = ms(t) + b.refresh(s, false)?;
        exec(s, "edit.undo", json!({}))?;
        b.refresh(s, false)?;
        Ok(v)
    });
    b.time("paste 12 MP layer (first) + refresh", &mut s, reps_big, false, |b, s, _| {
        let id = fresh_layer(s)?;
        exec(s, "layer.select", json!({"layer": id}))?;
        exec(s, "select.all", json!({}))?;
        exec(s, "edit.copy", json!({}))?;
        exec(s, "select.deselect", json!({}))?;
        b.refresh(s, false)?;
        let t = Instant::now();
        exec(s, "edit.paste", json!({}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    b.time("paste 12 MP layer (repeat) + refresh", &mut s, reps_big, false, |b, s, _| {
        let t = Instant::now();
        exec(s, "edit.paste", json!({}))?;
        let v = ms(t) + b.refresh(s, false)?;
        exec(s, "edit.undo", json!({}))?;
        b.refresh(s, false)?;
        Ok(v)
    });
}

/// #209: arrow-key nudge in the 150-layer document; #1771: a new layer there.
fn many_layer_scenarios(b: &mut Bench, sz: &Sizes) {
    // Row names stay stable across modes: the layer count is in the mode, not the key.
    let name = "nudge 1 px in the many-layer document + refresh".to_string();
    let new_layer = "new layer in the many-layer document + refresh";
    if !b.wanted(&name) && !b.wanted(new_layer) {
        return;
    }
    // The document's first refresh is part of this row (a crash there is reported against it).
    b.current = Some(name.clone());
    let (w, h) = sz.layered;
    let (mut s, mid) = match many_layers_doc(w, h, sz.many_layers) {
        Ok(v) => v,
        Err(e) => {
            b.errors.push((name, e));
            return;
        }
    };
    let _ = b.refresh(&s, true);
    let reps = b.reps;
    b.time(&name, &mut s, reps, true, |b, s, i| {
        let t = Instant::now();
        exec(s, "layer.translate", json!({"layer": mid, "dx": if i % 2 == 0 { 1 } else { -1 }, "dy": 0}))?;
        Ok(ms(t) + b.refresh(s, false)?)
    });
    // Layer › New › Layer: as fast at any layer count. Undone (untimed) so the count stays put.
    b.time(new_layer, &mut s, reps, true, |b, s, _| {
        let t = Instant::now();
        exec(s, "layer.new.layer", json!({}))?;
        let took = ms(t) + b.refresh(s, false)?;
        exec(s, "edit.undo", json!({}))?;
        b.refresh(s, false)?;
        Ok(took)
    });
}

/// #209: edits on the 15000 × 10000 16-bit document and the 32-bit document with 20 layers.
fn depth_scenarios(b: &mut Bench, sz: &Sizes) {
    let reps = b.reps;
    for (label, (w, h), depth, layers) in [("16-bit 15000x10000", sz.huge16, 16, 1u32), ("32-bit 20 layers", sz.float32, 32, 20)] {
        let dab = format!("brush dab 60 px on {label} + refresh");
        let mv = format!("move layer on {label} + refresh");
        if !b.wanted(&dab) && !b.wanted(&mv) {
            continue;
        }
        let built = (|| -> Res<(Session, u64)> {
            let mut s = Session::new();
            exec(&mut s, "file.new", json!({"width": w, "height": h, "depth": depth, "background": "white"}))?;
            let mut first = 0;
            for i in 0..layers {
                exec(&mut s, "layer.new.layer", json!({"name": format!("layer {i}")}))?;
                let (x, y) = ((i * 173) % (w * 3 / 4), (i * 97) % (h * 3 / 4));
                exec(&mut s, "select.rect", json!({"x": x, "y": y, "width": w / 4, "height": h / 4}))?;
                exec(&mut s, "edit.fill", json!({"color": format!("#{:02x}8060", (i * 41) % 256)}))?;
                exec(&mut s, "select.deselect", json!({}))?;
                if i == 0 {
                    first = active_layer(&s)?;
                }
            }
            Ok((s, first))
        })();
        let (mut s, layer) = match built {
            Ok(v) => v,
            Err(e) => {
                b.errors.push((dab, e));
                continue;
            }
        };
        let _ = b.refresh(&s, true);
        let _ = exec(&mut s, "layer.select", json!({"layer": layer}));
        b.time(&dab, &mut s, reps, true, |b, s, i| {
            let (x, y) = (200 + (i as u32 * 97) % (w - 400), 200 + (i as u32 * 61) % (h - 400));
            let t = Instant::now();
            exec(s, "paint.stroke", json!({"points": [[x, y]], "size": 60, "hardness": 0.8, "color": "#c03020"}))?;
            Ok(ms(t) + b.refresh(s, false)?)
        });
        b.time(&mv, &mut s, reps, true, |b, s, i| {
            let t = Instant::now();
            exec(s, "layer.translate", json!({"layer": layer, "dx": if i % 2 == 0 { 10 } else { -10 }, "dy": 0}))?;
            Ok(ms(t) + b.refresh(s, false)?)
        });
    }
}

/// #210 rows that are measurable before the job system exists.
fn job_scenarios(b: &mut Bench, sz: &Sizes) {
    let reps = b.reps.min(5);
    let open = "open 20 MP layered PSD (decode + first composite)";
    if b.wanted(open) {
        let (w, h) = sz.layered;
        match layered_doc(w, h).and_then(|(s, _)| {
            let d = s.active().ok_or("no document")?.doc.clone();
            photocraft_io::export(&d, "layered.psd", &Default::default()).map(|o| o.bytes).map_err(|e| e.to_string())
        }) {
            Ok(psd) => {
                let mut holder = Session::new();
                b.time(open, &mut holder, reps, true, |b, s, _| {
                    let t = Instant::now();
                    let d = photocraft_io::import("layered.psd", &psd).map_err(|e| e.to_string())?.document;
                    *s = Session::new();
                    s.open_document(d, Some("layered.psd".into()));
                    Ok(ms(t) + b.refresh(s, true)?)
                });
            }
            Err(e) => b.errors.push((open.into(), e)),
        }
    }
    let caf = "Content-Aware Fill 10 MP (200 px hole)";
    let heal = "Spot Healing stroke 10 MP";
    if b.wanted(caf) || b.wanted(heal) {
        let (w, h) = sz.ten_mp;
        match open_photo(w, h) {
            Ok(mut s) => {
                b.time(caf, &mut s, reps, false, |_, s, i| {
                    exec(s, "select.rect", json!({"x": w / 3 + i as u32 * 40, "y": h / 3, "width": 200, "height": 200, "ellipse": true}))?;
                    let t = Instant::now();
                    exec(s, "edit.contentAwareFill", json!({"seed": i}))?;
                    let v = ms(t);
                    exec(s, "edit.undo", json!({}))?;
                    exec(s, "select.deselect", json!({}))?;
                    Ok(v)
                });
                b.time(heal, &mut s, reps, false, |b, s, i| {
                    let (x, y) = (w / 4 + i as u32 * 60, h / 4);
                    let pts: Vec<Value> = (0..10).map(|k| json!([x + k * 12, y + k * 4])).collect();
                    let t = Instant::now();
                    exec(s, "paint.spotHealing", json!({"points": pts, "size": 60}))?;
                    Ok(ms(t) + b.refresh(s, false)?)
                });
            }
            Err(e) => b.errors.push((caf.into(), e)),
        }
    }
    background_job_scenarios(b, sz);
}

/// Wait (polling like the UI's frame loop) until job `id` has ended.
fn wait_job(s: &mut Session, id: photocraft_engine::jobs::JobId) -> Res<()> {
    let t = Instant::now();
    loop {
        if s.poll_jobs().iter().any(|e| e.id == id) {
            return Ok(());
        }
        if t.elapsed().as_secs() > 120 {
            return Err("job did not finish in 120 s".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn start_job(s: &mut Session, id: &str, p: Value) -> Res<photocraft_engine::jobs::JobId> {
    match s.start(id, p).map_err(|e| format!("{id}: {e}"))? {
        photocraft_engine::jobs::Started::Job(j) => Ok(j),
        photocraft_engine::jobs::Started::Done(_) => Err(format!("{id} did not run as a background job")),
    }
}

/// #210 background jobs: cancel latency (from `cancel_job` until the worker has exited) at
/// several points of a 24 MP Gaussian Blur and Content-Aware Fill; the UI thread's longest step
/// while 500 brushes import; the app's frame time while a 24 MP Gaussian Blur runs.
fn background_job_scenarios(b: &mut Bench, sz: &Sizes) {
    let reps = b.reps.min(8);
    let (w, h) = sz.big_photo;
    let blur = "cancel Gaussian Blur r 50 job on 24 MP";
    let caf = "cancel Content-Aware Fill job on 24 MP";
    if b.wanted(blur) || b.wanted(caf) {
        match open_photo(w, h) {
            Ok(mut s) => {
                for (name, cmd, params, select) in
                    [(blur, "filter.blur.gaussianBlur", json!({"radius": 50}), false), (caf, "edit.contentAwareFill", json!({}), true)]
                {
                    if select && let Err(e) = exec(&mut s, "select.rect", json!({"x": w / 3, "y": h / 3, "width": w / 5, "height": h / 4})) {
                        b.errors.push((name.into(), e));
                        continue;
                    }
                    b.time(name, &mut s, reps, false, |_, s, i| {
                        let job = start_job(s, cmd, params.clone())?;
                        // Cancel early, midway and late in the run.
                        std::thread::sleep(std::time::Duration::from_millis(20 + (i as u64 * 97) % 700));
                        let t = Instant::now();
                        s.cancel_job(job);
                        s.join_cancelled_jobs();
                        let v = ms(t);
                        wait_job(s, job)?;
                        Ok(v)
                    });
                }
            }
            Err(e) => b.errors.push((blur.into(), e)),
        }
    }
    let brushes = "import 500 brushes: longest UI-thread step";
    if b.wanted(brushes) {
        use photocraft_psd::abr::{AbrSample, LegacyBrush, LegacyTip, write_v12};
        let list: Vec<LegacyBrush> = (0..500)
            .map(|i| {
                let side = 40 + (i % 60) as u32;
                let data = (0..side * side).map(|k| ((k * 7 + i as u32) % 255) as u8).collect();
                LegacyBrush {
                    name: format!("Brush {i}"),
                    spacing: 25,
                    anti_alias: true,
                    tip: LegacyTip::Sampled(AbrSample { id: String::new(), width: side, height: side, depth: 8, data }),
                }
            })
            .collect();
        match write_v12(2, &list, true) {
            Ok(abr) => {
                let data = photocraft_engine::paint::tile::b64_encode(&abr);
                let mut s = Session::new();
                b.time(brushes, &mut s, reps.min(5), true, |_, s, _| {
                    // The UI thread's work: starting the job, then each frame's poll (the last one
                    // applies the 500 presets).
                    let t = Instant::now();
                    let job = start_job(s, "brush.presets.importAbr", json!({"data": data, "group": "Bench"}))?;
                    let mut worst = ms(t);
                    let t0 = Instant::now();
                    loop {
                        let t = Instant::now();
                        let done = s.poll_jobs().iter().any(|e| e.id == job);
                        worst = worst.max(ms(t));
                        if done {
                            break;
                        }
                        if t0.elapsed().as_secs() > 120 {
                            return Err("import did not finish".into());
                        }
                        std::thread::sleep(std::time::Duration::from_millis(4));
                    }
                    Ok(worst)
                });
            }
            Err(e) => b.errors.push((brushes.into(), e.to_string())),
        }
    }
    let frame = "app frame during 24 MP Gaussian Blur r 50 job";
    if b.wanted(frame) {
        frame_scenario(b, frame, w, h);
    }
}

/// The whole app (kittest harness, CPU canvas) with a 24 MP document: frame times while a
/// Gaussian Blur r 50 job runs, one sample per frame.
fn frame_scenario(b: &mut Bench, name: &str, w: u32, h: u32) {
    let mut s = match open_photo(w, h) {
        Ok(s) => s,
        Err(e) => return b.errors.push((name.into(), e)),
    };
    let doc = std::mem::take(&mut s);
    let mut harness = egui_kittest::Harness::builder().with_size(egui::vec2(1440.0, 900.0)).with_max_steps(4).build_eframe(move |cc| {
        photocraft_ui_egui::PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = photocraft_ui_egui::PhotocraftApp::new(doc, photocraft_ui_egui::Services::default());
        app.background_jobs = true;
        app
    });
    // Warm up: first frames build the canvas cache of the 24 MP document.
    harness.run_steps(6);
    b.current = Some(name.to_string());
    b.sampler.reset();
    let mut samples = Vec::new();
    let run = (|| -> Res<()> {
        let r = harness.state_mut().run("filter.blur.gaussianBlur", json!({"radius": 50}))?;
        if r.get("pending").is_none() {
            return Err("the blur did not start as a job".into());
        }
        let t0 = Instant::now();
        while harness.state().session.has_jobs() {
            let t = Instant::now();
            harness.step();
            samples.push(ms(t));
            if t0.elapsed().as_secs() > 120 {
                return Err("the job did not finish".into());
            }
        }
        if samples.len() < 10 {
            return Err(format!("only {} frames while the job ran", samples.len()));
        }
        Ok(())
    })();
    // The rows' document handle: the harness's session.
    let st = std::mem::take(&mut harness.state_mut().session);
    b.record(name, &st, &samples, run);
}

/// #211 rows: selection algorithms, Content-Aware Scale and Select Subject on 24 MP, and a brush
/// dab on an A4 300 ppi CMYK document.
fn algorithm_scenarios(b: &mut Bench, sz: &Sizes) {
    let reps = b.reps.min(5);
    let names = [
        "Select > Modify > Smooth r 50 on 24 MP",
        "Select > Modify > Feather r 50 on 24 MP",
        "Select Subject 24 MP (repeat)",
        "Content-Aware Scale 24 MP to 90 % width",
    ];
    if names.iter().any(|n| b.wanted(n)) {
        let (w, h) = sz.big_photo;
        match open_photo(w, h) {
            Ok(mut s) => {
                for (name, op) in [(names[0], "select.modify.smooth"), (names[1], "select.modify.feather")] {
                    b.time(name, &mut s, reps, false, |_, s, _| {
                        exec(s, "select.rect", json!({"x": w / 6, "y": h / 6, "width": w * 2 / 3, "height": h * 2 / 3, "ellipse": true}))?;
                        let t = Instant::now();
                        exec(s, op, json!({"radius": 50}))?;
                        let v = ms(t);
                        exec(s, "select.deselect", json!({}))?;
                        Ok(v)
                    });
                }
                b.time(names[2], &mut s, reps, true, |_, s, _| {
                    let t = Instant::now();
                    exec(s, "select.subject", json!({}))?;
                    let v = ms(t);
                    exec(s, "select.deselect", json!({}))?;
                    Ok(v)
                });
                // One run: today this takes far longer than its budget.
                b.time(names[3], &mut s, 1, false, |_, s, _| {
                    let t = Instant::now();
                    exec(s, "edit.contentAwareScale", json!({"scaleX": 90, "scaleY": 100}))?;
                    let v = ms(t);
                    exec(s, "edit.undo", json!({}))?;
                    Ok(v)
                });
            }
            Err(e) => b.errors.push((names[0].into(), e)),
        }
    }
    liquify_preview_scenario(b);
    let cmyk = "brush dab on A4 300 ppi CMYK + refresh";
    if b.wanted(cmyk) {
        let (w, h) = sz.a4;
        let mut s = Session::new();
        let setup = exec(&mut s, "file.new", json!({"width": w, "height": h, "mode": "cmyk", "resolution": 300, "background": "white"}))
            .and_then(|_| exec(&mut s, "layer.new.layer", json!({"name": "paint"})));
        if let Err(e) = setup {
            b.errors.push((cmyk.into(), e));
            return;
        }
        let _ = b.refresh(&s, true);
        let reps = b.reps;
        b.time(cmyk, &mut s, reps, true, |b, s, i| {
            let (x, y) = (200 + (i as u32 * 97) % (w - 400), 200 + (i as u32 * 61) % (h - 400));
            let t = Instant::now();
            exec(s, "paint.stroke", json!({"points": [[x, y]], "size": 40, "hardness": 0.8, "color": "#c03020"}))?;
            Ok(ms(t) + b.refresh(s, false)?)
        });
    }
    // P32: releasing a long stroke. The canvas draws it live (untimed here); the release offers
    // those pixels to the commit and re-composites only what the canvas hadn't shown yet.
    let release = "brush release: 600-point stroke, 200 px soft, 24 MP";
    if b.wanted(release) {
        let (w, h) = sz.big_photo;
        let mut s = Session::new();
        if let Err(e) = exec(&mut s, "file.new", json!({"width": w, "height": h, "background": "white"}))
            .and_then(|_| exec(&mut s, "layer.new.layer", json!({"name": "paint"})))
        {
            b.errors.push((release.into(), e));
            return;
        }
        let _ = b.refresh(&s, true);
        let reps = b.reps.min(5);
        b.time(release, &mut s, reps, false, |b, s, i| {
            use photocraft_engine::paint::StrokePoint;
            let (fw, fh) = (f64::from(w), f64::from(h));
            let pts: Vec<[f64; 3]> = (0..600)
                .map(|k| {
                    let t = k as f64 / 599.0;
                    [fw * (0.1 + 0.8 * t), fh * (0.5 + 0.35 * (t * 12.0 + i as f64).sin()), 0.6 + 0.4 * (t * 7.0).sin().abs()]
                })
                .collect();
            let mut p = json!({"points": [pts[0]], "size": 200, "hardness": 0.0, "color": "#c03020", "smoothing": 0.0});
            let mut live = photocraft_engine::brush_cmds::LiveStroke::begin(s, &p).map_err(|e| e.to_string())?;
            for c in pts[1..].chunks(4) {
                let sp: Vec<StrokePoint> = c.iter().map(|q| StrokePoint::new(q[0], q[1], q[2] as f32)).collect();
                live.push(&sp).map_err(|e| e.to_string())?;
            }
            p["points"] = json!(pts);
            p["seed"] = json!(live.seed);
            let t = Instant::now();
            s.offer_live_stroke(live).map_err(|e| e.to_string())?;
            let r = exec(s, "paint.stroke", p);
            s.clear_live_stroke();
            if r?.get("fromLive").and_then(Value::as_bool) != Some(true) {
                return Err("the release rendered the stroke again".into());
            }
            // The canvas already shows every step of the stroke: nothing left to re-composite.
            Ok(ms(t) + b.refresh_rect(s, photocraft_geom::Rect::EMPTY)?)
        });
    }
}

/// A source-resolution Liquify brush update on the visible part of a 3500×2200 layer.
/// Reports the old whole-document proxy cost alongside the full-resolution cropped update.
fn liquify_preview_scenario(b: &mut Bench) {
    let name = "Liquify 800 px brush on 3500x2200 at full resolution";
    if !b.wanted(name) {
        return;
    }
    let canvas = Rect::new(0, 0, 3500, 2200);
    let view = Rect::new(1150, 700, 2350, 1500);
    let source = Surface::with_default(PixelFormat::new(ColorMode::Rgb, SampleType::U8, true), &[0.32, 0.48, 0.64, 1.0]);
    let proxy = ProxyImage::new(&source, canvas, 1600);
    let mut low_out = vec![[0u8; 4]; proxy.w * proxy.h];
    let (vw, vh) = (view.width() as usize, view.height() as usize);
    let mut full_out = vec![[0u8; 4]; vw * vh];
    let mut source_scratch = Vec::new();
    let mut stroke = LiquifyStroke::new(LiquifyTool::ForwardWarp, 800.0);
    stroke.density = 100.0;
    stroke.points = vec![vec![1470.0, 1100.0, 1.0], vec![1530.0, 1100.0, 1.0]];
    let base = LiquifyField::new(canvas, auto_cell(canvas));
    let update = |field: &mut LiquifyField, out: &mut [[u8; 4]], scratch: &mut Vec<[f32; 4]>, full_res: bool| -> Res<f64> {
        let t = Instant::now();
        let dirty = field.stroke_segment(&stroke, [1470.0, 1100.0, 1.0], [1530.0, 1100.0, 1.0]);
        let ok = if full_res {
            ProxyImage::render_full_resolution(&source, field, view, dirty, out, scratch)
        } else {
            proxy.render(field, proxy.proxy_rect(dirty), out);
            true
        };
        if !ok {
            return Err("full-resolution preview could not allocate or render its visible region".into());
        }
        Ok(ms(t))
    };

    // Warm both paths, then compare identical field updates. The old renderer shades a downsampled
    // proxy; the new one produces one source pixel per preview texel at 100% zoom.
    let mut warm_low = base.clone();
    let _ = update(&mut warm_low, &mut low_out, &mut source_scratch, false);
    let mut warm_full = base.clone();
    let warm = update(&mut warm_full, &mut full_out, &mut source_scratch, true);
    if let Err(e) = warm {
        return b.errors.push((name.into(), e));
    }

    let mut low_samples = Vec::new();
    let mut samples = Vec::new();
    b.current = Some(name.into());
    b.sampler.reset();
    for _ in 0..b.reps.clamp(1, 5) {
        let mut field = base.clone();
        match update(&mut field, &mut low_out, &mut source_scratch, false) {
            Ok(t) => low_samples.push(t),
            Err(e) => return b.errors.push((name.into(), e)),
        }
        let mut field = base.clone();
        match update(&mut field, &mut full_out, &mut source_scratch, true) {
            Ok(t) => samples.push(t),
            Err(e) => return b.errors.push((name.into(), e)),
        }
    }
    let low_sorted = {
        let mut v = low_samples.clone();
        v.sort_by(f64::total_cmp);
        v
    };
    if let Some(median) = low_sorted.get(low_sorted.len() / 2) {
        println!("  existing 1600 px proxy update p50 {median:.2} ms (lower-resolution output)");
    }
    b.record(name, &Session::new(), &samples, Ok(()));
}
