//! A lost wgpu device must not take the app down (#243): the GPU canvas stops issuing GPU work,
//! the app falls back to the CPU canvas for the rest of the session with a notice, and the
//! documents stay. Both an injected loss and a real `Device::destroy()` are covered. Skips when
//! there is no GPU adapter.

use eframe::egui_wgpu::RenderState;
use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, Size};
use photocraft_gpu::Fault;
use photocraft_ui_egui::gpu_canvas::{self, GpuCanvas};
use photocraft_ui_egui::{PhotocraftApp, gpu_status};
use serde_json::json;

use photocraft_testkit::gpu_lock;

fn render_state() -> Option<RenderState> {
    let rs =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(gpu_canvas::wgpu_setup(), Default::default()))).ok();
    if rs.is_none() {
        eprintln!("skipping: no GPU adapter");
    }
    rs
}

fn doc() -> Document {
    Document::with_background("t", Size::new(300, 200), ColorMode::Rgb, SampleType::U8, Color::rgba(0.2, 0.5, 0.8, 1.0))
}

#[test]
fn composite_reuse_reports_held_bytes_and_honors_the_canvas_override() {
    let _gpu = gpu_lock();
    let Some(rs) = render_state() else { return };
    let g = GpuCanvas::new(&rs);
    let mut d = doc();
    d.layers.push(photocraft_doc::Layer::new("top", photocraft_doc::LayerContent::Fill(photocraft_doc::Fill::Solid(Color::rgba(0.8, 0.1, 0.2, 0.5)))));
    g.set_memory_budget(1 << 30);
    g.set_composite_cache_budget(0);
    assert_eq!(g.composite(&d, d.bounds(), false).unwrap().prefix_hits, 0);
    assert_eq!(g.compositor_bytes().unwrap().1, 0, "no effect maps or composite chunks");
    g.set_composite_cache_budget(512 << 20);
    assert_eq!(g.composite(&d, d.bounds(), false).unwrap().prefix_hits, 0);
    let bpp = if photocraft_gpu::Compositor::preferred_acc_format(&rs.adapter) == eframe::wgpu::TextureFormat::Rgba32Float { 16 } else { 8 };
    // The cache holds whole chunk-aligned cells, not the document's exact rectangle.
    let cell = u64::from(photocraft_gpu::CHUNK);
    let cells = u64::from(d.size.width).div_ceil(cell) * cell * u64::from(d.size.height).div_ceil(cell) * cell;
    assert_eq!(g.compositor_bytes().unwrap().1 as u64, cells * bpp);
    d.layers[1].opacity = 0.4;
    assert!(g.composite(&d, d.bounds(), false).unwrap().prefix_hits > 0);
    g.set_composite_cache_budget(0);
    g.set_memory_budget(512 << 20);
    assert_eq!(g.composite(&d, d.bounds(), false).unwrap().prefix_hits, 0);
    assert_eq!(g.compositor_bytes().unwrap().1, 0, "override survives the overall budget update");
}

#[test]
fn injected_loss_stops_gpu_work_without_panicking() {
    let _gpu = gpu_lock();
    let Some(rs) = render_state() else { return };
    let g = GpuCanvas::new(&rs);
    let d = doc();
    let key = d.id.0;
    let first = g.refresh(key, &d, None, None);
    assert_ne!(first.kind, "lost");
    assert!(g.fault().is_none());
    g.health().mark(Fault::Lost("test".into()));
    // Every entry point is a no-op or an error now.
    let r = g.refresh(key, &d, None, None);
    assert_eq!(r.kind, "lost");
    assert!(r.fallback.as_deref().unwrap_or("").contains("GPU device lost"));
    assert!(g.composite(&d, d.bounds(), false).is_err());
    assert!(!g.upload_rect(key, [0, 0], [1, 1], &[0; 4]));
    g.upload_full(key, [1, 1], &[0; 4]);
    g.set_display_lut(key, 0, 2, Some(&[0; 32]));
    assert!(g.read_texels(key).is_none());
    g.release();
    assert!(!g.has(key, [300, 200]));
}

#[test]
fn destroyed_device_falls_back_without_panicking() {
    let _gpu = gpu_lock();
    let Some(rs) = render_state() else { return };
    let g = GpuCanvas::new(&rs);
    let d = doc();
    let key = d.id.0;
    g.refresh(key, &d, None, None);
    rs.device.destroy();
    // The next refreshes run into the destroyed device: errors are recorded, nothing panics.
    let mut d2 = d.clone();
    d2.layers[0].opacity = 0.5;
    for _ in 0..3 {
        let r = g.refresh(key, &d2, Some(photocraft_geom::Rect::new(0, 0, 50, 50)), None);
        if r.kind == "lost" {
            break;
        }
    }
    assert!(g.read_texels(key).is_none() || g.fault().is_some());
    assert!(g.fault().is_some(), "device loss not detected");
    assert_eq!(g.refresh(key, &d2, None, None).kind, "lost");
}

#[test]
fn app_switches_to_the_cpu_canvas_and_keeps_the_documents() {
    let _gpu = gpu_lock();
    let Some(rs) = render_state() else { return };
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), photocraft_ui_egui::Services::default());
    app.set_wgpu(rs);
    app.run("file.new", json!({"width": 320, "height": 240})).unwrap();
    app.sync_views();
    assert!(app.gpu_active());
    assert_eq!(app.perf.gpu_info.canvas, "gpu");
    let started = std::rc::Rc::new(std::cell::Cell::new(false));
    let flag = started.clone();
    app.on_started(move |_| flag.set(true));
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            app.check_gpu(&ctx);
            egui::CentralPanel::default().show(ui, |ui| photocraft_ui_egui::canvas::document_area(app, ui));
            app.check_gpu(&ctx);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, photocraft_ui_egui::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    assert!(h.state().gpu_active(), "GPU canvas in use before the loss");
    // The device is lost (e.g. the OS killed a huge submission).
    h.state().gpu_health().unwrap().mark(Fault::Lost("killed by the OS".into()));
    h.run_steps(3);
    let app = h.state();
    assert!(!app.gpu_active(), "still on the GPU canvas");
    assert_eq!(app.session.documents().len(), 1, "document kept");
    assert_eq!(app.ui.status, gpu_status::LOST_MESSAGE);
    assert!(app.ui.status_error);
    assert!(app.ui.gpu_fallback_notice.as_deref().unwrap_or("").contains("killed by the OS"));
    assert!(app.ui.notices.is_empty(), "one recovery warning, no duplicate notice");
    assert!(app.perf.gpu_info.lost.as_deref().unwrap_or("").contains("killed by the OS"));
    assert_eq!(app.perf.gpu_info.canvas, "cpu");
    // The CPU path draws the document: a refresh after the loss is a CPU one.
    assert!(matches!(app.perf.last_refresh, "full" | "rect"), "{}", app.perf.last_refresh);
    // Editing keeps working.
    h.state_mut().run("layer.new.layer", json!({})).unwrap();
    h.run_steps(2);
    assert_eq!(h.state().session.active().map(|d| d.doc.layers.len()), Some(2));
    // The perf output carries the state for agents.
    let perf = serde_json::to_value(&h.state().perf).unwrap();
    assert_eq!(perf["gpuInfo"]["canvas"], "cpu");
    let _ = started;
}

#[test]
fn started_hook_runs_once_after_the_first_frames() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), photocraft_ui_egui::Services::default());
    let runs = std::rc::Rc::new(std::cell::Cell::new(0));
    let r = runs.clone();
    app.on_started(move |_| r.set(r.get() + 1));
    let ctx = egui::Context::default();
    for frame in 1..=2 {
        app.frame = frame;
        app.check_gpu(&ctx);
        assert_eq!(runs.get(), 0, "too early (frame {frame})");
    }
    for frame in 3..=6 {
        app.frame = frame;
        app.check_gpu(&ctx);
        assert_eq!(runs.get(), 1, "runs once (frame {frame})");
    }
}

#[test]
fn system_info_lists_the_graphics_state() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), photocraft_ui_egui::Services::default());
    app.perf.gpu_info.adapter = "Test GPU".into();
    app.perf.gpu_info.backend = "dx12".into();
    app.perf.gpu_info.preference = "auto".into();
    app.perf.gpu_info.selected = "dx12".into();
    app.perf.gpu_info.fallback = Some("the previous start didn't finish on vulkan; using dx12".into());
    let lines = gpu_status::system_info(&app);
    let text = lines.join("\n");
    for want in ["Graphics adapter: Test GPU", "Backend: dx12", "Selected at launch: dx12", "Fallback: the previous start", "Image compositor: CPU"] {
        assert!(text.contains(want), "{want} missing from\n{text}");
    }
    let v = gpu_status::system_info_json(&app);
    assert_eq!(v["gpu"]["adapter"], "Test GPU");
}

/// A canvas whose display textures exceed the canvas budget gets none: no texture set is built,
/// the refresh says so and the app draws it on the CPU canvas instead (#2934).
#[test]
fn canvas_over_the_texture_budget_builds_no_textures() {
    let _gpu = gpu_lock();
    let Some(rs) = render_state() else { return };
    let g = GpuCanvas::new(&rs);
    let d = doc();
    let (key, size) = (d.id.0, [d.size.width, d.size.height]);
    // Less than the full-resolution level alone.
    g.set_canvas_budget(u64::from(size[0] * size[1]) * 4 - 1);
    assert!(!g.fits_canvas(&d));
    let r = g.refresh(key, &d, None, None);
    assert_eq!(r.kind, gpu_canvas::REFRESH_OVER_BUDGET);
    assert!(!g.has(key, size) && g.texture_info(key).is_none());
    assert!(g.composite(&d, d.bounds(), false).is_err());
    g.upload_full(key, size, &vec![0; size[0] as usize * size[1] as usize * 4]);
    assert!(g.texture_info(key).is_none(), "no entry point builds the set");
    // Within the budget the same document is drawn on the GPU, at the bytes it was priced at.
    g.set_canvas_budget(1 << 30);
    assert!(g.fits_canvas(&d));
    assert_ne!(g.refresh(key, &d, None, None).kind, gpu_canvas::REFRESH_OVER_BUDGET);
    assert!(g.has(key, size));
    let (format, bytes) = g.texture_info(key).unwrap();
    let bpp = if format == eframe::wgpu::TextureFormat::Rgba16Float { 8 } else { 4 };
    assert_eq!(bytes, gpu_canvas::canvas_texture_bytes(size, gpu_canvas::DEFAULT_TILE, bpp));
    // Shrinking the budget below an existing set drops it on the next refresh.
    g.set_canvas_budget(1);
    assert_eq!(g.refresh(key, &d, None, None).kind, gpu_canvas::REFRESH_OVER_BUDGET);
    assert!(g.texture_info(key).is_none());
    g.release();
}
