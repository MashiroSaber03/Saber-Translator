//! Huge-document benchmark (issue #49): canvas refreshes of a 20000×20000 (or `--size`) 8-bit
//! RGB document with eleven layers (full-canvas pixel layers, soft blobs, an adjustment, type
//! layers with effects, a full-width band whose effect region is wider than the GPU texture
//! limit, a group with a clipped layer), on a device created the way the app creates it.
//!
//! One operation per run, so the process's peak memory is that operation's (plus building the
//! document):
//!
//! ```sh
//! cargo build --release -p photocraft-ui-egui --example huge_doc_bench
//! for op in full view; do
//!   /usr/bin/time -l target/release/examples/huge_doc_bench --size 20000x20000 --op $op 2>&1 | grep -E 'ms|MB|maximum resident|peak memory'
//! done
//! # The CPU fallback, for comparison:
//! PHOTOCRAFT_CPU_COMPOSE=1 /usr/bin/time -l target/release/examples/huge_doc_bench --op full
//! ```
//!
//! `full`: the first refresh (every page uploaded, every effect map built) and three repeats.
//! `view`: refreshes of a 2560×1440 viewport (zoomed in) at three places, after the first
//! refresh, then the same after a brush dab in each.

use std::time::Instant;

use eframe::egui_wgpu::RenderState;
use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Adjustment, Document, Effect, FxCommon, FxPaint, Layer, LayerContent, Shadow, StrokeFx, StrokePosition, TextLayer};
use photocraft_geom::{Rect, Size};
use photocraft_ui_egui::gpu_canvas::GpuCanvas;
use rayon::prelude::*;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// A layer over `r` whose RGBA8 pixels are `f(x, y)`, written in parallel-generated bands.
fn layer_from(name: &str, r: Rect, f: impl Fn(i32, i32) -> [u8; 4] + Sync) -> Layer {
    let mut l = Layer::raster(name, PixelFormat::RGBA8);
    let s = l.surface_mut().expect("raster");
    let mut y = r.y0;
    while y < r.y1 {
        let band = Rect::new(r.x0, y, r.x1, (y + 256).min(r.y1));
        let w = band.width() as usize;
        let mut bytes = vec![0u8; w * band.height() as usize * 4];
        bytes.par_chunks_mut(w * 4).enumerate().for_each(|(row, px)| {
            for (i, p) in px.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                p.copy_from_slice(&f(band.x0 + i as i32, band.y0 + row as i32));
            }
        });
        s.write_interleaved(band, &bytes);
        y = band.y1;
    }
    l
}

fn hash(x: i32, y: i32) -> u8 {
    ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) as u32 % 251) as u8
}

/// A soft-edged disc of colour `c` centred in `r`.
fn blob(name: &str, r: Rect, c: [u8; 3]) -> Layer {
    let (cx, cy, rad) = ((r.x0 + r.x1) as f32 / 2.0, (r.y0 + r.y1) as f32 / 2.0, r.width().min(r.height()) as f32 / 2.0 - 8.0);
    layer_from(name, r, |x, y| {
        let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
        let a = ((rad - d) / 6.0).clamp(0.0, 1.0);
        [c[0], c[1].wrapping_add(hash(x, y) / 16), c[2], (a * 255.0) as u8]
    })
}

/// A type layer: rows of glyph-like blocks rasterised into its cache.
fn text(name: &str, r: Rect) -> Layer {
    let glyphs = layer_from(name, r, |x, y| {
        let (gx, gy) = ((x - r.x0) % 90, (y - r.y0) % 160);
        let on = gx < 70 && gy < 120 && !(gx > 20 && gx < 50 && gy > 30 && gy < 90);
        [20, 30, 200, if on { 255 } else { 0 }]
    });
    let mut l = Layer::new(name, LayerContent::Text(TextLayer { cache: glyphs.surface().cloned(), ..Default::default() }));
    l.effects.items = vec![
        Effect::DropShadow(Shadow {
            common: FxCommon::new(BlendMode::Multiply, 0.75),
            color: Color::rgb(0.0, 0.0, 0.0),
            angle: 120.0,
            use_global_light: false,
            distance: 12.0,
            spread: 0.0,
            size: 16.0,
            contour: photocraft_doc::Contour::Linear,
            anti_alias: false,
            noise: 0.0,
            knocks_out: true,
        }),
        Effect::Stroke(StrokeFx {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            size: 4.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::rgb(1.0, 1.0, 1.0)),
        }),
    ];
    l
}

fn build(w: u32, h: u32) -> Document {
    let (wi, hi) = (w as i32, h as i32);
    let mut d = Document::new("huge", Size::new(w, h), ColorMode::Rgb, SampleType::U8);
    d.layers.push(layer_from("background", Rect::new(0, 0, wi, hi), |x, y| {
        let (fx, fy) = (x as f32 / wi as f32, y as f32 / hi as f32);
        let n = hash(x, y) / 8;
        [(80.0 + 120.0 * fx) as u8 + n, (60.0 + 140.0 * fy) as u8 + n, (150.0 - 60.0 * fx * fy) as u8 + n, 255]
    }));
    let mut over = layer_from("texture", Rect::new(0, 0, wi * 3 / 5, hi), |x, y| [200, 180u8.wrapping_add(hash(x, y) / 4), 120, 90 + hash(y, x) / 2]);
    over.blend = BlendMode::Multiply;
    over.opacity = 0.7;
    d.layers.push(over);
    let mut adj = Layer::new("exposure", LayerContent::Adjustment(Adjustment::Exposure { exposure: 0.3, offset: 0.0, gamma: 1.05 }));
    adj.opacity = 0.8;
    d.layers.push(adj);
    for (i, (x, y)) in [(0.15, 0.2), (0.55, 0.35), (0.3, 0.7)].into_iter().enumerate() {
        let (x, y) = ((wi as f32 * x) as i32, (hi as f32 * y) as i32);
        let mut b = blob(&format!("blob {i}"), Rect::new(x, y, x + wi / 6, y + wi / 6), [220, 60 + 60 * i as u8, 40]);
        b.blend = if i == 1 { BlendMode::Screen } else { BlendMode::Normal };
        d.layers.push(b);
    }
    // A full-width band with effects: its effect region is wider than a 16384 px texture.
    let mut band = layer_from("band", Rect::new(40, hi / 2, wi - 40, hi / 2 + 1200), |x, y| [240, 240, 230, if (x / 64 + y / 64) % 5 == 0 { 0 } else { 230 }]);
    band.effects.items = vec![
        Effect::DropShadow(Shadow {
            common: FxCommon::new(BlendMode::Multiply, 0.6),
            color: Color::rgb(0.0, 0.0, 0.1),
            angle: 90.0,
            use_global_light: false,
            distance: 20.0,
            spread: 0.0,
            size: 24.0,
            contour: photocraft_doc::Contour::Linear,
            anti_alias: false,
            noise: 0.0,
            knocks_out: true,
        }),
        Effect::Stroke(StrokeFx {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            size: 3.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::rgb(0.1, 0.1, 0.1)),
        }),
    ];
    d.layers.push(band);
    d.layers.push(text("title", Rect::new(wi / 10, hi / 12, wi / 10 + 6000, hi / 12 + 900)));
    d.layers.push(text("caption", Rect::new(wi / 2, hi * 5 / 6, wi / 2 + 4000, hi * 5 / 6 + 500)));
    let a = blob("group base", Rect::new(wi * 2 / 3, hi / 10, wi * 2 / 3 + wi / 4, hi / 10 + wi / 4), [40, 160, 90]);
    let mut c = layer_from("clipped", Rect::new(wi * 2 / 3, hi / 10, wi, hi / 10 + wi / 8), |x, y| [250, 230, 40, 120 + hash(x, y) / 3]);
    c.clipped = true;
    c.blend = BlendMode::Overlay;
    let mut g = Layer::group("group", vec![a, c]);
    g.opacity = 0.9;
    d.layers.push(g);
    d
}

/// A headless GPU canvas on a device created like the app's.
fn canvas() -> Option<(GpuCanvas, RenderState)> {
    let setup = photocraft_ui_egui::gpu_canvas::wgpu_setup();
    let rs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(setup, Default::default()))).ok()?;
    let l = rs.device.limits();
    eprintln!("adapter: {} (max texture {})", rs.adapter.get_info().name, l.max_texture_dimension_2d);
    Some((GpuCanvas::new(&rs), rs))
}

/// One canvas refresh (everything, or `damage`), waiting for the GPU. Returns the path taken.
fn refresh(g: &GpuCanvas, rs: &RenderState, doc: &Document, damage: Option<Rect>) -> (String, Option<String>) {
    let r = g.refresh(doc.id.0, doc, damage, None);
    let _ = rs.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None });
    let kind = if r.kind.starts_with("gpu") { format!("{}, {} tiles up", r.kind, r.uploads) } else { r.kind.to_string() };
    (kind, r.fallback)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) =
        arg(&args, "--size").and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap_or(20000), b.parse().unwrap_or(20000)))).unwrap_or((20000, 20000));
    let op = arg(&args, "--op").unwrap_or_else(|| "full".into());
    let Some((g, rs)) = canvas() else {
        eprintln!("no GPU adapter");
        return;
    };
    let t = Instant::now();
    let mut doc = build(w, h);
    println!("document {w}×{h} ({:.0} MP), {} layers, built in {:.0} ms", (w as f64 * h as f64) / 1e6, doc.walk().len(), ms(t));

    // The compositor budget the app would use: Memory Usage (default 8 GB, `--memory-mb`) less
    // the document's pixels, within a quarter of RAM (`PHOTOCRAFT_RAM_MB` simulates a smaller
    // machine). The view is the first viewport (`view`) or the whole document (`full`).
    let allowance = arg(&args, "--memory-mb").and_then(|v| v.parse::<u64>().ok()).unwrap_or(8192) << 20;
    let pixels: u64 = doc
        .walk()
        .iter()
        .flat_map(|(_, _, l)| l.surface().into_iter().chain(l.mask.as_ref().map(|m| &m.surface)))
        .map(|s| s.tiles().map(|(_, t)| t.bytes().len() as u64).sum::<u64>())
        .sum();
    let ram = photocraft_ui_egui::gpu_canvas::physical_memory();
    let budget = photocraft_ui_egui::gpu_canvas::memory_budget(allowance, pixels, ram);
    g.set_memory_budget(budget);
    println!("RAM {} MB, document pixels {} MB, Memory Usage {} MB -> GPU budget {} MB", ram.unwrap_or(0) >> 20, pixels >> 20, allowance >> 20, budget >> 20);
    let first_view = Rect::from_xywh((w as f32 * 0.1) as i32, (h as f32 * 0.1) as i32, 2560, 1440);
    g.set_focus(Some(if op == "view" { first_view } else { doc.bounds() }));

    let t = Instant::now();
    let (kind, fallback) = refresh(&g, &rs, &doc, None);
    println!("first refresh ({kind:<8})       {:>9.0} ms{}", ms(t), fallback.map_or(String::new(), |f| format!("  [fallback: {f}]")));
    let report = |g: &GpuCanvas| {
        if let Some((format, bytes)) = g.texture_info(doc.id.0) {
            println!("canvas texture {format:?}: {} MB with mips", bytes >> 20);
        }
        if let Some((pages, fx)) = g.compositor_bytes() {
            println!("compositor: {} MB resident pages, {} MB effect maps", pages >> 20, fx >> 20);
        }
    };
    report(&g);
    match op.as_str() {
        "full" => {
            for _ in 0..3 {
                let t = Instant::now();
                let (kind, _) = refresh(&g, &rs, &doc, None);
                println!("full refresh ({kind:<8})        {:>9.0} ms", ms(t));
            }
        }
        "view" => {
            let views: Vec<Rect> = [(0.1, 0.1), (0.45, 0.48), (0.7, 0.8)]
                .into_iter()
                .map(|(x, y)| Rect::from_xywh((w as f32 * x) as i32, (h as f32 * y) as i32, 2560, 1440).intersect(&doc.bounds()))
                .collect();
            for v in &views {
                let t = Instant::now();
                let (kind, _) = refresh(&g, &rs, &doc, Some(*v));
                println!("viewport refresh ({kind:<8})    {:>9.0} ms  at {v:?}", ms(t));
            }
            // A dab on the band (an effect layer) inside each viewport, then its refresh.
            let band = doc.layers.iter().position(|l| l.name == "band").unwrap_or(0);
            for v in &views {
                let c = Rect::from_xywh(v.x0 + 600, h as i32 / 2 + 300, 120, 120);
                if let Some(s) = doc.layers.get_mut(band).and_then(Layer::surface_mut) {
                    s.fill_rect(c, &[0.1, 0.3, 0.9, 1.0]);
                }
                let t = Instant::now();
                let (kind, _) = refresh(&g, &rs, &doc, Some(*v));
                println!("dab + viewport refresh ({kind:<8}) {:>6.0} ms", ms(t));
            }
        }
        other => eprintln!("unknown --op {other}"),
    }
    report(&g);
}
