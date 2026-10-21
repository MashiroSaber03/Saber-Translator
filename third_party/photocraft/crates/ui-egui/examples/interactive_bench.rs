//! End-to-end timings of the interactive paths on a large photo (default 7360×4912 = 36 MP):
//! open JPEG / PSD, brush dab, Gaussian Blur, adjustment layer tweak, layer move, undo,
//! save .pcraft / .psd, export PNG / JPEG, plus the canvas refresh each edit triggers.
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example interactive_bench -- [--size 7360x4912] [--reps 5] [--json out.json] [--only text] [--cpu]
//! ```
//!
//! Every edit goes through `Session::execute` (the command path the UI, CLI and MCP share:
//! history snapshot, command, damage), then the canvas refresh the app would do: the GPU
//! compositor over the edit's damage rect (or the whole document), waited on so the time is the
//! real latency. Without a GPU adapter the CPU compositor stands in. Zoom and pan don't
//! recomposite (the canvas samples the composited texture's mip levels); "viewport (CPU)" times
//! what the CPU fallback pays for one 1920×1080 view. Each row is the median of `--reps` runs
//! (min / max shown); the first run of an operation is reported separately as "cold".

use std::time::Instant;

use eframe::wgpu;
use photocraft_doc::{Document, LayerContent};
use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::{Value, json};

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

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    comp: photocraft_gpu::Compositor,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
            .ok()?;
    eprintln!("adapter: {}", adapter.get_info().name);
    // The limits the app requests (see `gpu_canvas::use_adapter_limits`).
    let limits = photocraft_ui_egui::gpu_canvas::device_limits(&adapter);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).ok()?;
    let comp = photocraft_gpu::Compositor::new(&device);
    Some(Gpu { device, queue, comp })
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

struct Bench {
    gpu: Option<Gpu>,
    reps: usize,
    /// `--only text`: time only rows whose name contains it (others run once, untimed, for the
    /// state later rows need).
    only: Option<String>,
    rows: Vec<(String, f64, f64, f64, f64)>,
    /// Per row (same order as `rows`): the timed samples and the peak RSS while they ran, for
    /// `--json` (`cargo xtask perf`).
    extra: Vec<(Vec<f64>, Option<u64>)>,
    rss: photocraft_testkit::perf::RssSampler,
}

impl Bench {
    /// Canvas refresh after an edit: the damage rect when the session reports one for this
    /// revision, else the whole document (what `canvas::ensure_gpu` does).
    fn refresh(&mut self, s: &Session, full: bool) -> f64 {
        let st = s.active().expect("document");
        let doc = st.doc.clone();
        let region = if full { doc.bounds() } else { st.last_damage.map_or(doc.bounds(), |r| r.inflate(reach(&doc.layers)).intersect(&doc.bounds())) };
        self.render(&doc, region)
    }

    /// GPU (or CPU fallback) composite of `region` of `doc`, waited on.
    fn render(&mut self, doc: &Document, region: Rect) -> f64 {
        if region.is_empty() {
            return 0.0;
        }
        let t = Instant::now();
        match &mut self.gpu {
            Some(g) => {
                if g.comp.render(&g.device, &g.queue, doc, region, |_, _| {}).is_err() {
                    std::hint::black_box(photocraft_compose::render(doc, region));
                }
                let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
            None => {
                std::hint::black_box(photocraft_compose::render(doc, region));
            }
        }
        t.elapsed().as_secs_f64() * 1000.0
    }

    /// Times `f` (`reps` runs after a cold one); `f` returns its own breakdown-free total in ms.
    fn time(&mut self, name: &str, mut f: impl FnMut(&mut Self) -> f64) {
        if self.only.as_ref().is_some_and(|o| !name.contains(o.as_str())) {
            f(self);
            return;
        }
        self.rss.reset();
        let cold = f(self);
        let mut v: Vec<f64> = (0..self.reps).map(|_| f(self)).collect();
        self.extra.push((v.clone(), self.rss.peak()));
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        println!("{name:<44} {med:>9.1} ms   (min {:>8.1}, max {:>8.1}, cold {:>8.1})", v[0], v[v.len() - 1], cold);
        self.rows.push((name.to_string(), med, v[0], v[v.len() - 1], cold));
    }
}

/// How far an edit's composite change reaches beyond its damage (layer effects).
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

fn select(s: &mut Session, index: usize) {
    let id = s.active().expect("doc").doc.layers[index].id;
    exec(s, "layer.select", json!({"layer": id.0}));
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn exec(s: &mut Session, id: &str, p: Value) {
    if let Err(e) = s.execute(id, p) {
        panic!("{id}: {e}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) =
        arg(&args, "--size").and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap_or(7360), b.parse().unwrap_or(4912)))).unwrap_or((7360, 4912));
    let reps: usize = arg(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    let mut b = Bench {
        gpu: if args.iter().any(|a| a == "--cpu") { None } else { gpu() },
        reps,
        only: arg(&args, "--only"),
        rows: Vec::new(),
        extra: Vec::new(),
        rss: photocraft_testkit::perf::RssSampler::start(std::time::Duration::from_millis(2)),
    };
    println!("document {w}×{h} ({:.1} MP), 8-bit RGB, {} reps, {}", (w * h) as f64 / 1e6, reps, if b.gpu.is_some() { "GPU canvas" } else { "CPU canvas" });

    // ---- open / save / export ----------------------------------------------------------------
    let img = photo(w, h);
    let mut jpeg = Vec::new();
    b.time("encode JPEG (codecs, q 90)", |_| {
        let t = Instant::now();
        jpeg = photocraft_codecs::encode(&img, photocraft_codecs::Format::Jpeg, &Default::default()).expect("jpeg");
        ms(t)
    });
    let mut s = Session::new();
    b.time("open JPEG (decode + document + refresh)", |b| {
        let t = Instant::now();
        let doc = photocraft_io::import("photo.jpg", &jpeg).expect("import").document;
        s = Session::new();
        s.open_document(doc, Some("photo.jpg".into()));
        let t = ms(t);
        t + b.refresh(&s, true)
    });
    // A working document: the photo, a painting layer above it, an adjustment on top later.
    exec(&mut s, "layer.new.layer", json!({"name": "paint"}));
    let doc = |s: &Session| -> std::sync::Arc<Document> { s.active().expect("doc").doc.clone() };
    b.time("full refresh (2 layers)", |b| b.refresh(&s, true));
    b.time("viewport 1920×1080 (CPU compose)", |_| {
        let d = doc(&s);
        let t = Instant::now();
        std::hint::black_box(photocraft_compose::render(&d, Rect::from_xywh(w as i32 / 3, h as i32 / 3, 1920, 1080)));
        ms(t)
    });

    // ---- painting --------------------------------------------------------------------------
    let mut k = 0;
    b.time("brush dab 60 px (command + refresh)", |b| {
        k += 1;
        let (x, y) = (200 + (k * 97) % (w as i32 - 400), 200 + (k * 61) % (h as i32 - 400));
        let t = Instant::now();
        exec(&mut s, "paint.stroke", json!({"points": [[x, y]], "size": 60, "hardness": 0.8, "color": "#c03020"}));
        ms(t) + b.refresh(&s, false)
    });
    b.time("brush stroke 40 dabs 120 px", |b| {
        k += 1;
        let (x, y) = (300 + (k * 131) % (w as i32 - 2000), 300 + (k * 71) % (h as i32 - 600));
        let pts: Vec<Value> = (0..40).map(|i| json!([x + i * 30, y + ((i as f32 * 0.4).sin() * 80.0) as i32])).collect();
        let t = Instant::now();
        exec(&mut s, "paint.stroke", json!({"points": pts, "size": 120, "hardness": 0.5, "color": "#2040c0", "spacing": 0.25}));
        ms(t) + b.refresh(&s, false)
    });
    b.time("undo (stroke)", |b| {
        exec(&mut s, "paint.stroke", json!({"points": [[500, 500], [900, 700]], "size": 80, "color": "#20a040"}));
        let t = Instant::now();
        exec(&mut s, "edit.undo", json!({}));
        ms(t) + b.refresh(&s, false)
    });

    // ---- filters / adjustments / layers ------------------------------------------------------
    select(&mut s, 0);
    for radius in [4.0, 40.0, 250.0] {
        b.time(&format!("Gaussian Blur r {radius} (photo layer) + refresh"), |b| {
            let t = Instant::now();
            exec(&mut s, "filter.blur.gaussianBlur", json!({"radius": radius}));
            let v = ms(t) + b.refresh(&s, false);
            exec(&mut s, "edit.undo", json!({}));
            b.refresh(&s, false);
            v
        });
    }
    // Filter dialog live preview, per radius change (#464): the command on the proxy plus its
    // flatten for upload.
    for radius in [40.0, 250.0] {
        b.time(&format!("Gaussian Blur dialog preview r {radius}"), |_| {
            let st = s.active().expect("doc");
            let k = photocraft_ui_egui::proxy::factor(&st.doc);
            let t = Instant::now();
            let p = photocraft_ui_egui::filter_dialog::preview_document(&st.doc, st.active_layer, "filter.blur.gaussianBlur", &json!({"radius": radius}), k)
                .expect("preview");
            std::hint::black_box(photocraft_compose::flatten(&p));
            ms(t)
        });
    }
    // Image › Adjustments dialog preview, per settings change (#77). Before: the command re-run
    // on a downsampled proxy (plus its flatten for upload). After: the document with a temporary
    // clipped adjustment layer, recomposited over the target's area by the canvas compositor.
    for selection in [false, true] {
        if selection {
            exec(&mut s, "select.rect", json!({"x": w / 4, "y": h / 4, "width": w / 2, "height": h / 2, "ellipse": true}));
        }
        let sel = if selection { ", selection" } else { "" };
        let mut v = 0;
        b.time(&format!("Curves dialog change, CPU proxy preview{sel}"), |_| {
            v = (v + 7) % 120;
            let st = s.active().expect("doc");
            let k = photocraft_ui_egui::proxy::factor(&st.doc);
            let t = Instant::now();
            let p = photocraft_ui_egui::filter_dialog::preview_document(
                &st.doc,
                st.active_layer,
                "image.adjustments.curves",
                &json!({"points": [[0, 0], [100, 100 + v], [255, 255]]}),
                k,
            )
            .expect("preview");
            std::hint::black_box(photocraft_compose::flatten(&p));
            ms(t)
        });
        // Once per dialog session (the app caches both per document revision).
        let (base, region) = {
            let st = s.active().expect("doc");
            let target = st.active_layer.expect("layer");
            let t = Instant::now();
            let base = photocraft_ui_egui::adjust_preview::base_document(&st.doc, target).expect("base");
            let region = photocraft_ui_egui::adjust_preview::region(&st.doc, target);
            println!("  (layer preview session setup: {:.1} ms, region {region:?})", ms(t));
            (base, region)
        };
        b.time(&format!("Curves dialog change, layer preview{sel}"), |b| {
            v = (v + 7) % 120;
            let t = Instant::now();
            let p =
                photocraft_ui_egui::adjust_preview::with_settings(&base, "curves", &json!({"points": [[0, 0], [100, 100 + v], [255, 255]]})).expect("preview");
            let build = ms(t);
            let t = Instant::now();
            match &mut b.gpu {
                Some(g) => {
                    if g.comp.render(&g.device, &g.queue, &p, region, |_, _| {}).is_err() {
                        std::hint::black_box(photocraft_compose::render(&p, region));
                    }
                    let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }
                None => {
                    std::hint::black_box(photocraft_compose::render(&p, region));
                }
            }
            build + ms(t)
        });
        // Zoomed in, the canvas composites the region's visible part (here a 2560×1440 view at
        // 100 %) and catches up elsewhere when the view moves.
        let view = Rect::from_xywh(w as i32 / 3, h as i32 / 3, 2560, 1440).intersect(&region);
        b.time(&format!("Curves dialog change, layer preview 100 % view{sel}"), |b| {
            v = (v + 7) % 120;
            let t = Instant::now();
            let p =
                photocraft_ui_egui::adjust_preview::with_settings(&base, "curves", &json!({"points": [[0, 0], [100, 100 + v], [255, 255]]})).expect("preview");
            match &mut b.gpu {
                Some(g) => {
                    if g.comp.render(&g.device, &g.queue, &p, view, |_, _| {}).is_err() {
                        std::hint::black_box(photocraft_compose::render(&p, view));
                    }
                    let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }
                None => {
                    std::hint::black_box(photocraft_compose::render(&p, view));
                }
            }
            ms(t)
        });
        // Zoomed out (fit to a 1600-pixel-wide view on a 2× display: k = 2 for 36 MP), the canvas
        // previews on a reduced copy built once per session.
        let zoom = 1600.0 / w as f32 * 2.0;
        let k = photocraft_ui_egui::adjust_preview::proxy_factor(zoom);
        let t = Instant::now();
        let proxy = photocraft_ui_egui::adjust_preview::proxy_base(&base, k);
        println!("  (zoomed-out proxy k {k} setup: {:.1} ms)", ms(t));
        let kk = k as i32;
        let pregion = Rect::new(region.x0 / kk, region.y0 / kk, region.x1 / kk + 1, region.y1 / kk + 1).intersect(&proxy.bounds());
        b.time(&format!("Curves dialog change, layer preview zoomed out{sel}"), |b| {
            v = (v + 7) % 120;
            let t = Instant::now();
            let p = photocraft_ui_egui::adjust_preview::proxy_with_settings(&proxy, "curves", &json!({"points": [[0, 0], [100, 100 + v], [255, 255]]}))
                .expect("preview");
            match &mut b.gpu {
                Some(g) => {
                    if g.comp.render(&g.device, &g.queue, &p, pregion, |_, _| {}).is_err() {
                        std::hint::black_box(photocraft_compose::render(&p, pregion));
                    }
                    let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }
                None => {
                    std::hint::black_box(photocraft_compose::render(&p, pregion));
                }
            }
            ms(t)
        });
        if selection {
            exec(&mut s, "select.deselect", json!({}));
        }
    }
    exec(&mut s, "layer.newAdjustmentLayer.levels", json!({"inBlack": 10, "inWhite": 240, "gamma": 1.1}));
    let mut g = 1.0;
    b.time("Levels layer tweak (gamma) + full refresh", |b| {
        g = if g > 1.5 { 0.8 } else { g + 0.07 };
        let t = Instant::now();
        exec(&mut s, "layer.setAdjustment", json!({"gamma": g}));
        ms(t) + b.refresh(&s, false)
    });
    select(&mut s, 1);
    let mut d = 1;
    b.time("move paint layer 10 px + refresh", |b| {
        d = -d;
        let t = Instant::now();
        exec(&mut s, "layer.translate", json!({"dx": 10 * d, "dy": 5 * d}));
        ms(t) + b.refresh(&s, false)
    });
    b.time("undo (move)", |b| {
        exec(&mut s, "layer.translate", json!({"dx": 7, "dy": 3}));
        let t = Instant::now();
        exec(&mut s, "edit.undo", json!({}));
        ms(t) + b.refresh(&s, false)
    });

    // ---- save / export ---------------------------------------------------------------------
    let mut psd = Vec::new();
    b.time("save .pcraft (bytes, full)", |_| {
        let d = doc(&s);
        let t = Instant::now();
        std::hint::black_box(photocraft_format::save_to_bytes(&d, &photocraft_format::SaveOptions::default()).expect("pcraft"));
        ms(t)
    });
    b.time("save .psd", |_| {
        let d = doc(&s);
        let t = Instant::now();
        psd = photocraft_io::export(&d, "x.psd", &Default::default()).expect("psd").bytes;
        ms(t)
    });
    b.time("open .psd (3 layers) + refresh", |b| {
        let t = Instant::now();
        let d = photocraft_io::import("x.psd", &psd).expect("psd").document;
        let mut s2 = Session::new();
        s2.open_document(d, None);
        ms(t) + b.refresh(&s2, true)
    });
    b.time("flatten (CPU compositor, 3 layers)", |_| {
        let d = doc(&s);
        let t = Instant::now();
        std::hint::black_box(photocraft_compose::flatten(&d));
        ms(t)
    });
    b.time("export PNG (flatten + encode)", |_| {
        let d = doc(&s);
        let t = Instant::now();
        std::hint::black_box(photocraft_io::export(&d, "x.png", &Default::default()).expect("png"));
        ms(t)
    });
    b.time("export JPEG (flatten + encode)", |_| {
        let d = doc(&s);
        let t = Instant::now();
        std::hint::black_box(photocraft_io::export(&d, "x.jpg", &Default::default()).expect("jpg"));
        ms(t)
    });

    // ---- live gradient ------------------------------------------------------------------------
    // A Gradient tool drag in live mode: each pointer move previews a shallow copy of the document
    // with the edited fill (gradient_ui::display_doc, through gradient_fill_cmds::apply_set) and
    // recomposites the whole canvas, since the fill covers it. Release commits one command.
    exec(&mut s, "gradient.fill.create", json!({"from": [w as f32 * 0.2, h as f32 * 0.5], "to": [w as f32 * 0.8, h as f32 * 0.5]}));
    let gid = s.active().and_then(|st| st.active_layer).expect("gradient layer");
    let mut k = 0;
    b.time("live gradient preview doc (no refresh)", |_| {
        k += 1;
        let to = [w as f32 * (0.6 + 0.03 * (k % 10) as f32), h as f32 * (0.3 + 0.04 * (k % 7) as f32)];
        let t = Instant::now();
        let d = doc(&s);
        let l = d.layer(gid).expect("layer");
        let LayerContent::Fill(f) = &l.content else { panic!("not a fill") };
        let f = photocraft_engine::gradient_fill_cmds::apply_set(l, f, d.bounds(), &json!({"to": to}), [0.0, 0.0, 0.0, 1.0], [1.0; 4]).expect("set");
        let mut shown = (*d).clone();
        shown.layer_mut(gid).expect("layer").content = LayerContent::Fill(f);
        std::hint::black_box(&shown);
        ms(t)
    });
    b.time("live gradient drag update (preview + full refresh)", |b| {
        k += 1;
        let to = [w as f32 * (0.6 + 0.03 * (k % 10) as f32), h as f32 * (0.3 + 0.04 * (k % 7) as f32)];
        let t = Instant::now();
        let d = doc(&s);
        let l = d.layer(gid).expect("layer");
        let LayerContent::Fill(f) = &l.content else { panic!("not a fill") };
        let f = photocraft_engine::gradient_fill_cmds::apply_set(l, f, d.bounds(), &json!({"to": to}), [0.0, 0.0, 0.0, 1.0], [1.0; 4]).expect("set");
        let mut shown = (*d).clone();
        shown.layer_mut(gid).expect("layer").content = LayerContent::Fill(f);
        ms(t) + b.render(&shown, shown.bounds())
    });
    b.time("live gradient release (gradient.fill.set + refresh)", |b| {
        k += 1;
        let t = Instant::now();
        exec(&mut s, "gradient.fill.set", json!({"layer": gid.0, "to": [w as f32 * (0.6 + 0.03 * (k % 10) as f32), h as f32 * 0.4]}));
        ms(t) + b.refresh(&s, false)
    });

    if let Some(out) = arg(&args, "--json") {
        let rows: Vec<Value> = b
            .rows
            .iter()
            .zip(&b.extra)
            .map(|((n, med, min, max, cold), (samples, rss))| {
                let mut r = json!({"name": n, "median_ms": med, "min_ms": min, "max_ms": max, "cold_ms": cold});
                // For `cargo xtask perf`: raw samples (it computes p50 / p95) and peak RSS.
                r["samples_ms"] = json!(samples);
                r["peak_rss_bytes"] = json!(rss);
                r
            })
            .collect();
        std::fs::write(&out, serde_json::to_string_pretty(&json!({"width": w, "height": h, "gpu": b.gpu.is_some(), "rows": rows})).unwrap())
            .expect("write json");
    }
}
