//! The canvas is colour-managed (#46): a Display P3 document shows its colours converted to
//! the (sRGB) monitor on the GPU canvas (display LUT) and on the CPU canvas (flipped views draw
//! through the CPU path); sRGB documents are shown unchanged. Skips when no GPU adapter with
//! 32-bit float render targets exists (like `crates/gpu/tests/parity.rs`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use photocraft_ui_egui::PhotocraftApp;
use serde_json::json;

/// Renders the UI with one 512² document of `rgb` tagged with `profile`; returns the median
/// colour of a patch at the centre of the canvas, for the GPU and the CPU canvas.
fn render(profile: &str, rgb: [f32; 3], reopen: bool) -> Option<([u8; 3], [u8; 3])> {
    let float_targets = Arc::new(AtomicBool::new(false));
    let ft = float_targets.clone();
    let observer = Arc::new(std::sync::Mutex::new(None));
    let observer_out = observer.clone();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            if let Some(rs) = cc.wgpu_render_state.as_ref() {
                ft.store(photocraft_gpu::Compositor::preferred_acc_format(&rs.adapter) == eframe::wgpu::TextureFormat::Rgba32Float, Ordering::SeqCst);
                *observer_out.lock().expect("GPU observer lock") = Some(photocraft_ui_egui::gpu_canvas::GpuCanvas::new(rs));
                app.set_wgpu(rs.clone());
            }
            app
        })
    }));
    let Ok(mut harness) = built else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    if !float_targets.load(Ordering::SeqCst) {
        eprintln!("skipping: adapter can't render Rgba32Float");
        return None;
    }
    // Outside the catch_unwind: a failing setup command fails the test instead of reading as
    // "no GPU adapter" (#760).
    let hex = format!("#{:02x}{:02x}{:02x}", (rgb[0] * 255.0).round() as u8, (rgb[1] * 255.0).round() as u8, (rgb[2] * 255.0).round() as u8);
    let app = harness.state_mut();
    app.run("file.new", json!({"width": 512, "height": 512, "mode": "rgb", "depth": 8})).expect("new");
    app.run("edit.assignProfile", json!({"profile": profile})).expect("assign");
    app.run("edit.fill", json!({"color": hex})).expect("fill");
    // Native loading starts at revision one; preserve that collision in the reopen case.
    if reopen {
        app.session.active_mut().expect("active document").revision = 1;
    }
    let sample = |harness: &mut egui_kittest::Harness<'_, PhotocraftApp>| {
        harness.run_steps(6);
        let img = harness.render().expect("render");
        // The document is fitted into the canvas area: its centre lies on the document.
        let (cx, cy) = (img.width() as i32 * 2 / 5, img.height() as i32 / 2);
        let mut px: Vec<[u8; 3]> = Vec::new();
        for dy in -4..=4 {
            for dx in -4..=4 {
                let p = img.get_pixel((cx + dx) as u32, (cy + dy) as u32);
                px.push([p[0], p[1], p[2]]);
            }
        }
        px.sort();
        px[px.len() / 2]
    };
    let gpu = sample(&mut harness);
    if reopen {
        let doc = harness.state().session.active().expect("active document").doc.clone();
        let id = doc.id;
        let bytes = photocraft_format::save_to_bytes(&doc, &photocraft_format::SaveOptions::default()).expect("save native document");
        drop(doc);
        let observer = observer.lock().expect("GPU observer lock").clone().expect("GPU observer");
        let (request, _) = photocraft_ui_egui::control::ControlRequest::new("ui.menu.invoke", json!({"id":"filter.blur.gaussianBlur"}));
        let ctx = harness.ctx.clone();
        let response = photocraft_ui_egui::control::handle(harness.state_mut(), &ctx, &request);
        assert!(matches!(response, photocraft_ui_egui::control::Outcome::Done(_)));
        harness.run_steps(6);
        let preview = id.0 ^ (1u64 << 61);
        assert!(observer.has(preview, [512, 512]), "filter preview uploaded");
        harness.state_mut().run("file.close", json!({})).expect("close");
        assert!(!observer.has(preview, [512, 512]), "close releases preview resources");
        // No repaint between close and reopen: sync_views must release the old resource/signature.
        let reopened = photocraft_format::load_from_bytes(&bytes).expect("load native document");
        assert_eq!(reopened.id, id, "native loading preserves document ID");
        harness.state_mut().session.add_document(reopened, None);
        harness.state_mut().sync_views();
        harness.run_steps(6);
        assert!(observer.has(preview, [512, 512]), "same-ID reopen uploads the still-open filter preview again");
        harness.state_mut().ui.dialogs.clear();
        let after_reopen = sample(&mut harness);
        assert!((0..3).all(|i| gpu[i].abs_diff(after_reopen[i]) <= 1), "GPU native reopen changed display color: {gpu:?} -> {after_reopen:?}");
        let (request, _) = photocraft_ui_egui::control::ControlRequest::new("ui.menu.invoke", json!({"id":"image.adjustments.brightnessContrast"}));
        photocraft_ui_egui::control::handle(harness.state_mut(), &ctx, &request);
        photocraft_ui_egui::adjust_preview::display_doc(harness.state_mut(), 0).expect("adjustment preview");
        harness.state_mut().run("file.close", json!({})).expect("close adjustment owner");
        assert!(photocraft_ui_egui::adjust_preview::shown_key(harness.state()).is_none(), "close drops adjustment preview state before same-ID reopen");
        harness.state_mut().session.add_document(photocraft_format::load_from_bytes(&bytes).expect("reopen adjustment owner"), None);
        harness.state_mut().sync_views();
        photocraft_ui_egui::adjust_preview::display_doc(harness.state_mut(), 0).expect("rebuilt adjustment preview");
        harness.state_mut().ui.dialogs.clear();
    }
    harness.state_mut().ui.view.flip_horizontal = true;
    let cpu = sample(&mut harness);
    eprintln!("gpu {gpu:?} cpu {cpu:?}");
    Some((gpu, cpu))
}

fn close(a: [u8; 3], b: [f32; 3], tol: f32) -> bool {
    (0..3).all(|i| (a[i] as f32 - b[i]).abs() <= tol)
}

#[test]
fn display_p3_is_converted_on_gpu_and_cpu_canvases() {
    let _gpu = photocraft_testkit::gpu_lock();
    // P3 (0.8, 0.5, 0.3) → sRGB (216.7, 122.9, 64.3) (see engine display_color_tests); raw would be (204, 128, 77).
    let Some((gpu, cpu)) = render("display-p3", [0.8, 0.5, 0.3], true) else { return };
    assert!(close(cpu, [216.7, 122.9, 64.3], 2.0), "CPU canvas {cpu:?}");
    assert!(close(gpu, [216.7, 122.9, 64.3], 3.0), "GPU canvas {gpu:?}");
}

#[test]
fn srgb_documents_are_unchanged() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some((gpu, cpu)) = render("srgb", [0.8, 0.5, 0.3], false) else { return };
    assert!(close(cpu, [204.0, 128.0, 77.0], 1.0), "CPU canvas {cpu:?}");
    assert!(close(gpu, [204.0, 128.0, 77.0], 1.0), "GPU canvas {gpu:?}");
}
