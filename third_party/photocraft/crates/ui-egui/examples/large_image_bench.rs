//! Large-image benchmark (issue #49): a 14000×14000 (or `--size`) 8-bit RGB photo through open,
//! canvas refresh, panel thumbnails, a filter, a brush stroke, save and export, with the GPU
//! device created the way the app creates it (so its texture limits are the app's).
//!
//! One operation per run, so the process's peak RSS is that operation's (plus opening):
//!
//! ```sh
//! cargo build --release -p photocraft-ui-egui --example large_image_bench
//! for op in open refresh thumbs filter brush psd png jpeg pcraft; do
//!   /usr/bin/time -l target/release/examples/large_image_bench --size 14000x14000 --op $op 2>&1 | grep -E 'ms|maximum resident'
//! done
//! ```
//!
//! The source JPEG is generated once and cached (`--cache <dir>`, default the system temp dir).
//! `--op open` is the baseline every other operation includes. `--cpu` skips the GPU.
//! `--depth 16` or `--depth 32` converts the document to that bit depth after opening (the
//! conversion is part of "open"), e.g. a 36 MP 16-bit refresh: `--size 7360x4912 --depth 16 --op refresh`.

use std::time::Instant;

use eframe::egui_wgpu::RenderState;
use photocraft_engine::Session;
use photocraft_ui_egui::gpu_canvas::GpuCanvas;
use serde_json::{Value, json};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// A photo-like RGB image: smooth gradients, soft blobs, fine noise (so JPEG has real work).
fn photo(w: u32, h: u32) -> photocraft_codecs::Image {
    use rayon::prelude::*;
    let mut px = vec![0u8; w as usize * h as usize * 3];
    px.par_chunks_mut(w as usize * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w as usize {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 997) as f32 / 997.0 - 0.5;
            let blob = (-(((fx - 0.4) * 3.0).powi(2) + ((fy - 0.55) * 4.0).powi(2))).exp();
            let r = 0.55 + 0.35 * (fx * 6.0).sin() * 0.5 + 0.3 * blob + n * 0.06;
            let g = 0.45 + 0.25 * (fy * 9.0 + fx * 2.0).cos() * 0.5 + 0.2 * blob + n * 0.06;
            let b = 0.35 + 0.4 * fy - 0.2 * blob + n * 0.06;
            for (c, v) in [r, g, b].into_iter().enumerate() {
                row[x * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    photocraft_codecs::Image::from_raw(w, h, photocraft_codecs::ChannelLayout::Rgb, photocraft_codecs::SampleType::U8, px).expect("image")
}

/// A headless GPU canvas on a device created like the app's.
fn canvas() -> Option<(GpuCanvas, RenderState)> {
    let setup = photocraft_ui_egui::gpu_canvas::wgpu_setup();
    let rs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(setup, Default::default()))).ok()?;
    let l = rs.device.limits();
    eprintln!("adapter: {} (max texture {}, max buffer {} MB)", rs.adapter.get_info().name, l.max_texture_dimension_2d, l.max_buffer_size >> 20);
    Some((GpuCanvas::new(&rs), rs))
}

fn exec(s: &mut Session, id: &str, p: Value) {
    if let Err(e) = s.execute(id, p) {
        panic!("{id}: {e}");
    }
}

/// The canvas refresh the app does after an edit (`canvas::ensure_gpu`): the damage rect, or
/// everything, then waits for the GPU (as presenting the next frame would, which also releases
/// the frame's transient GPU memory). Returns the path taken.
fn refresh(gpu: Option<&(GpuCanvas, RenderState)>, s: &Session, full: bool) -> &'static str {
    let st = s.active().expect("document");
    let damage = if full { None } else { st.last_damage };
    match gpu {
        Some((g, rs)) => {
            let kind = g.refresh(st.doc.id.0, &st.doc, damage, None).kind;
            let _ = rs.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None });
            kind
        }
        None => {
            let r = damage.unwrap_or(st.doc.bounds()).intersect(&st.doc.bounds());
            std::hint::black_box(photocraft_compose::render(&st.doc, r));
            "cpu"
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) =
        arg(&args, "--size").and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap_or(14000), b.parse().unwrap_or(14000)))).unwrap_or((14000, 14000));
    let op = arg(&args, "--op").unwrap_or_else(|| "open".into());
    let dir = arg(&args, "--cache").map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    let src = dir.join(format!("photocraft-bench-{w}x{h}.jpg"));
    if !src.exists() {
        let t = Instant::now();
        let img = photo(w, h);
        let jpeg = photocraft_codecs::encode(&img, photocraft_codecs::Format::Jpeg, &Default::default()).expect("jpeg");
        std::fs::write(&src, jpeg).expect("write cache");
        eprintln!("generated {} in {:.0} ms (run again to measure)", src.display(), ms(t));
        // `--json` (`cargo xtask perf`) measures in the same run.
        if arg(&args, "--json").is_none() {
            return;
        }
    }
    let gpu = if args.iter().any(|a| a == "--cpu") { None } else { canvas() };
    let gpu = gpu.as_ref();
    println!("document {w}×{h} ({:.0} MP), op {op}", (w as f64 * h as f64) / 1e6);

    let t = Instant::now();
    let bytes = std::fs::read(&src).expect("read");
    let doc = photocraft_io::import("photo.jpg", &bytes).expect("import").document;
    drop(bytes);
    let mut s = Session::new();
    s.open_document(doc, Some("photo.jpg".into()));
    match arg(&args, "--depth").as_deref() {
        Some("16") => exec(&mut s, "image.mode.bits16", json!({})),
        Some("32") => exec(&mut s, "image.mode.bits32", json!({})),
        _ => {}
    }
    let t_open = ms(t);
    let t = Instant::now();
    let path = refresh(gpu, &s, true);
    let t_first = ms(t);
    println!("open (decode + document)      {t_open:>9.0} ms");
    println!("first refresh ({path:<8})       {t_first:>9.0} ms");
    // `--json out.json` (`cargo xtask perf`): open and first refresh (every op includes them),
    // the canvas texture size and the process's peak RSS.
    let json_out = arg(&args, "--json");
    let mut json_rows = vec![
        photocraft_testkit::perf::row("open (decode + document)", &[t_open], None, None),
        photocraft_testkit::perf::row("first refresh", &[t_first], None, None),
    ];
    if let Some((g, _)) = gpu
        && let Some((format, bytes)) = g.texture_info(s.active().expect("doc").doc.id.0)
    {
        println!("canvas texture {format:?}, {} MB with mips", bytes >> 20);
        if let Some(r) = json_rows.get_mut(1) {
            r["gpu_bytes"] = json!(bytes);
        }
    }

    let doc = || s.active().expect("doc").doc.clone();
    match op.as_str() {
        "open" => {}
        "refresh" => {
            let mut v = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                let path = refresh(gpu, &s, true);
                v.push(ms(t));
                println!("full refresh ({path:<8})        {:>9.0} ms", ms(t));
            }
            json_rows.push(photocraft_testkit::perf::row("full refresh", &v, None, None));
        }
        "thumbs" => {
            let d = doc();
            let t = Instant::now();
            std::hint::black_box(photocraft_compose::thumbnail(&d, 512));
            println!("navigator thumbnail (512)     {:>9.0} ms", ms(t));
            let t = Instant::now();
            std::hint::black_box(photocraft_compose::thumbnail(&d, 56));
            println!("channel thumbnails (56)       {:>9.0} ms", ms(t));
            let t = Instant::now();
            std::hint::black_box(photocraft_compose::thumbnail(&d, 384));
            println!("histogram source (384)        {:>9.0} ms", ms(t));
        }
        "filter" => {
            let t = Instant::now();
            exec(&mut s, "filter.blur.gaussianBlur", json!({"radius": 10.0}));
            let t1 = ms(t);
            let t = Instant::now();
            let path = refresh(gpu, &s, false);
            println!("Gaussian Blur r 10            {t1:>9.0} ms");
            println!("  refresh ({path:<8})           {:>9.0} ms", ms(t));
        }
        "invert" => {
            // Image › Adjustments on the whole layer (#1772), then on a half-transparent layer.
            let t = Instant::now();
            exec(&mut s, "image.adjustments.invert", json!({}));
            let t1 = ms(t);
            let t = Instant::now();
            let path = refresh(gpu, &s, false);
            println!("Invert                        {t1:>9.0} ms   refresh ({path}) {:>6.0} ms", ms(t));
            exec(&mut s, "select.rect", json!({"x": 0, "y": 0, "width": w / 2, "height": h}));
            exec(&mut s, "layer.new.layerViaCopy", json!({}));
            let t = Instant::now();
            exec(&mut s, "image.adjustments.invert", json!({}));
            println!("Invert, half-transparent layer {:>8.0} ms", ms(t));
            json_rows.push(photocraft_testkit::perf::row("invert", &[t1], None, None));
        }
        "brush" => {
            exec(&mut s, "layer.new.layer", json!({"name": "paint"}));
            refresh(gpu, &s, true);
            for k in 0..3 {
                let (x, y) = (300 + k * 2500, 300 + k * 1500);
                let pts: Vec<Value> = (0..40).map(|i| json!([x + i * 30, y + ((i as f32 * 0.4).sin() * 80.0) as i32])).collect();
                let t = Instant::now();
                exec(&mut s, "paint.stroke", json!({"points": pts, "size": 120, "hardness": 0.5, "color": "#2040c0", "spacing": 0.25}));
                let t1 = ms(t);
                let t = Instant::now();
                let path = refresh(gpu, &s, false);
                println!("brush stroke 40 dabs          {t1:>9.0} ms   refresh ({path}) {:>6.0} ms", ms(t));
            }
        }
        "psd" | "png" | "jpeg" | "pcraft" => {
            let ext = if op == "jpeg" { "jpg" } else { op.as_str() };
            let d = doc();
            let t = Instant::now();
            let out = photocraft_io::export(&d, &format!("x.{ext}"), &Default::default()).expect("export");
            println!("save {ext:<6} ({:>5} MB)          {:>9.0} ms", out.bytes.len() >> 20, ms(t));
        }
        other => eprintln!("unknown --op {other}"),
    }
    if let Some(out) = json_out {
        let peak = photocraft_testkit::perf::process_peak_rss_bytes().or_else(photocraft_testkit::perf::current_rss_bytes);
        let context = json!({"width": w, "height": h, "op": op, "gpu_adapter": gpu.map(|(_, rs)| rs.adapter.get_info().name)});
        let mut report = photocraft_testkit::perf::report("large_image_bench", context, json_rows, None);
        report["process_peak_rss_bytes"] = json!(peak);
        if let Err(e) = photocraft_testkit::perf::write_report(&out, &report) {
            eprintln!("{e}");
        }
    }
}
