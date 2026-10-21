//! #2121: Interface › UI Scale is egui's `zoom_factor`, folded into `pixels_per_point` with the
//! display scale. It must change only the interface, never the image: at 100% zoom one document
//! pixel stays one *physical* display pixel at every UI scale.
//!
//! These tests drive the real UI Scale preference, render the app offscreen on the GPU and the
//! CPU canvas, and measure the document's on-screen size in the pixels. A regression where the
//! canvas stopped dividing by `pixels_per_point` (drawing the image at `UI scale`×) or the GPU
//! shader stopped multiplying back would change the measured size. They skip when no wgpu
//! adapter is available (like `checker_canvas.rs`).

use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::control::{ControlRequest, handle};
use serde_json::{Value, json};

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

/// The app with the GPU canvas (`gpu`), or without it (the CPU fallback path).
fn harness(gpu: bool) -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(1200.0, 800.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            if let Some(rs) = cc.wgpu_render_state.as_ref().filter(|_| gpu) {
                app.set_wgpu(rs.clone());
            }
            app
        })
    });
    match built {
        Ok(h) => Some(h),
        Err(_) => {
            eprintln!("skipping: no GPU adapter");
            None
        }
    }
}

fn control(h: &mut Harness, method: &str, params: Value) {
    let ctx = h.ctx.clone();
    let (req, _rx) = ControlRequest::new(method, params);
    handle(h.state_mut(), &ctx, &req);
    h.run_steps(4);
}

/// Longest run of `true` in `flags`, as `(start, len)`.
fn longest_run(flags: impl Iterator<Item = bool>) -> (usize, usize) {
    let (mut best, mut start, mut len) = ((0, 0), 0, 0);
    for (i, f) in flags.enumerate() {
        if f {
            if len == 0 {
                start = i;
            }
            len += 1;
            if len > best.1 {
                best = (start, len);
            }
        } else {
            len = 0;
        }
    }
    best
}

/// The black document's width and height in physical screen pixels, measured through the canvas
/// centre (the longest dark run on the centre row and column; the pasteboard is lighter).
fn document_physical_size(h: &mut Harness) -> (i64, i64) {
    let ppp = h.ctx.pixels_per_point();
    let rendered = h.render().expect("render");
    let (w, height) = (rendered.width(), rendered.height());
    let px = rendered.into_raw();
    let at = |x: u32, y: u32| -> u8 {
        let i = ((y * w + x) * 4) as usize;
        px[i]
    };
    let r = h.state().last_canvas_rect;
    let cx = ((r.center().x * ppp) as u32).min(w - 1);
    let cy = ((r.center().y * ppp) as u32).min(height - 1);
    let on_doc = |x: u32, y: u32| at(x, y) < 12;
    let (_, dw) = longest_run((0..w).map(|x| on_doc(x, cy)));
    let (_, dh) = longest_run((0..height).map(|y| on_doc(cx, y)));
    (dw as i64, dh as i64)
}

/// One black `DOC`-square document at 100%/200% zoom at each UI scale, on the GPU or CPU canvas.
fn assert_document_size_is_physical(gpu: bool) {
    let Some(mut h) = harness(gpu) else { return };
    h.run_steps(4);
    const DOC: u32 = 60;
    for ui_scale in ["100", "125", "150", "200"] {
        {
            let app = h.state_mut();
            while app.session.active().is_some() {
                app.run("file.close", json!({"discard": true})).expect("close");
            }
            app.run("prefs.set", json!({"values": {"interface.uiScale": ui_scale}})).expect("ui scale");
            app.run("file.new", json!({"width": DOC, "height": DOC, "background": "white"})).expect("new");
            app.run("edit.fill", json!({"color": "#000000"})).expect("fill");
            app.sync_views();
        }
        h.run_steps(6);
        assert_eq!(h.state().perf.gpu, gpu, "UI Scale {ui_scale}: expected the {} canvas", if gpu { "GPU" } else { "CPU" });
        for zoom in [1.0f32, 2.0] {
            control(&mut h, "ui.set", json!({"zoom": zoom, "center": [DOC as f32 / 2.0, DOC as f32 / 2.0]}));
            h.run_steps(4);
            let (w, hgt) = document_physical_size(&mut h);
            let want = (DOC as f32 * zoom).round() as i64;
            assert!(
                (w - want).abs() <= 2 && (hgt - want).abs() <= 2,
                "{} canvas, UI Scale {ui_scale}, zoom {zoom}: the document is {w}x{hgt} physical px, want {want}",
                if gpu { "GPU" } else { "CPU" }
            );
        }
    }
}

#[test]
fn ui_scale_keeps_the_document_pixel_size_on_the_gpu_canvas() {
    let _gpu = photocraft_testkit::gpu_lock();
    assert_document_size_is_physical(true);
}

#[test]
fn ui_scale_keeps_the_document_pixel_size_on_the_cpu_canvas() {
    let _gpu = photocraft_testkit::gpu_lock();
    assert_document_size_is_physical(false);
}
