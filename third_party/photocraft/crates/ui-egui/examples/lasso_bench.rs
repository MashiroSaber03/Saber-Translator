//! Frame times of a freehand Lasso drag on a large document (#1773): the real `PhotocraftApp` in
//! an offscreen egui_kittest harness on wgpu, a 6336×7920 RGB document, the view at 125 %, then a
//! spiral drag of many pointer moves, one per frame, and the release (the `select.lasso` commit)
//! and the frame after it (the canvas refresh the new selection causes).
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example lasso_bench -- [--size 6336x7920] [--moves 600]
//!     [--layers 4] [--full 3] [--selection] [--zoom 1.25] [--render]
//! ```
//!
//! `--layers` layers, the first `--full` of them canvas-sized (the rest 1024×1024 patches);
//! `--selection` starts from an elliptical selection; `--render` also tessellates and paints
//! each frame on the GPU (with a readback, about 7 ms here), otherwise only the app's logic runs.
//! `PHOTOCRAFT_GPU_SYNC=1` makes a canvas refresh wait for the GPU.

use std::time::Instant;

use egui::{Event, Modifiers, PointerButton, Pos2};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer};
use photocraft_geom::{Rect, Size};
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::state::Tool;
use photocraft_ui_egui::{PhotocraftApp, Services};
use rayon::prelude::*;
use serde_json::json;

type H = egui_kittest::Harness<'static, PhotocraftApp>;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

static RENDER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One frame, timed: the app's logic, plus tessellation and the GPU paint (with a readback) under `--render`.
fn frame(h: &mut H) -> f64 {
    let t = Instant::now();
    h.step();
    if RENDER.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = h.render();
    }
    ms(t)
}

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

fn screen(h: &H, x: f64, y: f64) -> Pos2 {
    let app = h.state();
    let idx = app.session.active_index().unwrap_or(0);
    let v = &app.ui.views[idx];
    let rect = photocraft_ui_egui::rulers::content_rect(app, app.last_canvas_rect);
    ViewXform { rect, zoom: v.zoom, center: v.center, flip: app.ui.view.flip_horizontal, rotation: v.rotation, aspect: 1.0 }.to_screen(x as f32, y as f32)
}

fn stats(name: &str, mut v: Vec<f64>) {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return;
    }
    let (med, p90, max) = (v[v.len() / 2], v[(v.len() * 9 / 10).min(v.len() - 1)], v[v.len() - 1]);
    let first: f64 = v.iter().sum::<f64>() / v.len() as f64;
    println!("{name:<40} median {med:>8.2} ms  mean {first:>8.2}  p90 {p90:>8.2}  max {max:>8.2}");
}

fn select_ellipse(h: &mut H, w: u32, hh: u32) {
    let (cx, cy, rx, ry) = (w as f64 / 2.0, hh as f64 / 2.0, w as f64 / 2.0 - 500.0, hh as f64 / 2.0 - 500.0);
    let pts: Vec<[f64; 2]> = (0..720)
        .map(|k| {
            let a = k as f64 / 720.0 * std::f64::consts::TAU;
            [cx + rx * a.cos(), cy + ry * a.sin()]
        })
        .collect();
    let r = h.state_mut().run("select.lasso", json!({"points": pts, "mode": "replace", "antiAlias": true, "feather": 0.0}));
    println!("selection: {:?}", r.is_ok());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, hh) =
        arg(&args, "--size").and_then(|s| s.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).unwrap_or((6336u32, 7920u32));
    let moves: usize = arg(&args, "--moves").and_then(|v| v.parse().ok()).unwrap_or(600);
    let zoom: f32 = arg(&args, "--zoom").and_then(|v| v.parse().ok()).unwrap_or(1.25);
    RENDER.store(args.iter().any(|a| a == "--render"), std::sync::atomic::Ordering::Relaxed);
    let full: usize = arg(&args, "--full").and_then(|v| v.parse().ok()).unwrap_or(3);
    let layers: usize = arg(&args, "--layers").and_then(|v| v.parse().ok()).unwrap_or(4);
    let t = Instant::now();
    let (wi, hi) = (w as i32, hh as i32);
    let mut doc = Document::new("big", Size::new(w, hh), ColorMode::Rgb, SampleType::U8);
    doc.layers.push(layer_from("background", Rect::new(0, 0, wi, hi), |x, y| {
        let n = hash(x, y) / 8;
        [(80 + (x % 120)) as u8 + n, (60 + (y % 140)) as u8 + n, 150, 255]
    }));
    // `full` canvas-sized layers, the rest 1024×1024 patches spread over the document (canvas-sized
    // ones for all of a 100-layer PSB would need over 20 GB).
    for i in 1..layers {
        let r = if i < full {
            Rect::new(0, 0, wi, hi)
        } else {
            let (x, y) = ((i as i32 * 977) % (wi - 1024).max(1), (i as i32 * 1531) % (hi - 1024).max(1));
            Rect::new(x, y, x + 1024, y + 1024)
        };
        doc.layers.push(layer_from(&format!("layer {i}"), r, move |x, y| [200, hash(x + i as i32, y) / 2, 120, 90 + hash(y, x) / 2]));
    }
    println!("built {w}×{hh}, {layers} layers in {:.0} ms", ms(t));

    let builder = egui_kittest::Harness::builder().with_size(egui::vec2(1600.0, 1000.0));
    let mut h: H = builder.wgpu_setup(photocraft_ui_egui::gpu_canvas::wgpu_setup()).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            app.set_wgpu(rs.clone());
        }
        app.session.open_document(doc, Some("big.psb".into()));
        app.sync_views();
        app
    });
    for _ in 0..6 {
        frame(&mut h);
    }
    // With a selection the drag happens above it (y = 250): a press inside would drag it instead.
    let with_selection = args.iter().any(|a| a == "--selection");
    let focus_y = if with_selection { 250.0 } else { hh as f64 / 2.0 };
    if with_selection {
        select_ellipse(&mut h, w, hh);
        for _ in 0..4 {
            frame(&mut h);
        }
    }
    h.state_mut().ui.tool = Tool::Lasso;
    {
        let app = h.state_mut();
        let idx = app.session.active_index().unwrap_or(0);
        if let Some(v) = app.ui.views.get_mut(idx) {
            v.zoom = zoom;
            v.center = [w as f32 / 2.0, focus_y as f32];
        }
    }
    for _ in 0..6 {
        frame(&mut h);
    }
    let idle: Vec<f64> = (0..30).map(|_| frame(&mut h)).collect();
    stats("idle frame", idle);
    let hover: Vec<f64> = (0..30)
        .map(|i| {
            let p = screen(&h, w as f64 / 2.0 + i as f64, focus_y);
            h.event(Event::PointerMoved(p));
            frame(&mut h)
        })
        .collect();
    stats("hover frame", hover);

    for rep in 0..2 {
        let (cx, cy, r) = (w as f64 / 2.0, focus_y, 300.0);
        let at = |k: usize| {
            let a = k as f64 / moves as f64 * std::f64::consts::TAU * 3.0;
            (cx + r * a.cos() * (1.0 + k as f64 / moves as f64), cy + r * a.sin())
        };
        let (x0, y0) = at(0);
        let p0 = screen(&h, x0, y0);
        h.event(Event::PointerMoved(p0));
        frame(&mut h);
        h.event(Event::PointerButton { pos: p0, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        let press = frame(&mut h);
        let mut times = Vec::with_capacity(moves);
        for k in 1..=moves {
            let (x, y) = at(k);
            h.event(Event::PointerMoved(screen(&h, x, y)));
            times.push(frame(&mut h));
        }
        let has_sel = h.state().session.active().is_some_and(|st| st.doc.selection.is_some());
        println!("lasso active at the end: {}, selection exists: {has_sel}", photocraft_ui_egui::lasso_ui::active(h.state()));
        let early: Vec<f64> = times.iter().take(moves / 5).copied().collect();
        let late: Vec<f64> = times.iter().skip(moves * 4 / 5).copied().collect();
        let (x1, y1) = at(moves);
        let p1 = screen(&h, x1, y1);
        h.event(Event::PointerButton { pos: p1, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        let release = frame(&mut h);
        let after = frame(&mut h);
        {
            let p = &h.state().perf;
            println!(
                "after release: refresh {} {} px, composite {:.1} ms, upload {:.1} ms, gpu uploads {}, fallback {:?}, last command {} {:.1} ms",
                p.last_refresh, p.last_refresh_px, p.composite_ms, p.upload_ms, p.gpu_uploads, p.gpu_fallback, p.last_command, p.command_ms
            );
        }
        println!("rep {rep}: press {press:.2} ms, release {release:.2} ms, next {after:.2} ms");
        stats("lasso drag frames (all)", times);
        stats("lasso drag frames (first 20%)", early);
        stats("lasso drag frames (last 20%)", late);
        println!("spans: {:?}", h.state().perf.spans);
        let r = h.state_mut().run("select.deselect", json!({}));
        println!("deselect: {r:?}");
        for _ in 0..4 {
            frame(&mut h);
        }
        if with_selection {
            select_ellipse(&mut h, w, hh);
            for _ in 0..4 {
                frame(&mut h);
            }
        }
    }
}
