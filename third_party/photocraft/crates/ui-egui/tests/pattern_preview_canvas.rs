//! View › Pattern Preview (#1067): the canvas shows the document repeated around itself, on the
//! GPU canvas and on the CPU canvas (flipped views draw through the CPU path), and nothing past
//! the document while it is off. Skips when no GPU adapter exists (like `color_managed_canvas.rs`).

use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::canvas::ViewXform;
use serde_json::json;

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

fn harness() -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            if let Some(rs) = cc.wgpu_render_state.as_ref() {
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

/// The colour on screen at document point `at` (which may lie past the document).
fn sample(h: &mut Harness, at: [f32; 2]) -> [u8; 3] {
    h.run_steps(6);
    let xf = ViewXform::active(h.state()).expect("canvas view");
    let p = xf.to_screen(at[0], at[1]);
    let img = h.render().expect("render");
    let px = img.get_pixel(p.x.round() as u32, p.y.round() as u32);
    [px[0], px[1], px[2]]
}

/// View › Pattern Preview, as the menu runs it.
fn toggle(h: &mut Harness) {
    let (request, _) = photocraft_ui_egui::control::ControlRequest::new("ui.menu.invoke", json!({"id": "view.patternPreview"}));
    let ctx = h.ctx.clone();
    let response = photocraft_ui_egui::control::handle(h.state_mut(), &ctx, &request);
    assert!(matches!(response, photocraft_ui_egui::control::Outcome::Done(_)));
}

fn near(a: [u8; 3], b: [u8; 3]) -> bool {
    (0..3).all(|i| a[i].abs_diff(b[i]) <= 3)
}

#[test]
fn pattern_preview_repeats_the_document_on_gpu_and_cpu_canvases() {
    let Some(mut h) = harness() else { return };
    let app = h.state_mut();
    app.run("file.new", json!({"width": 200, "height": 100, "mode": "rgb", "depth": 8})).expect("new");
    app.run("edit.fill", json!({"color": "#2050d0"})).expect("fill");
    // A red band along the left edge only, so the copy to the right shows it beside the right edge.
    app.run("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 100})).expect("select");
    app.run("edit.fill", json!({"color": "#e01010"})).expect("fill band");
    app.run("select.deselect", json!({})).expect("deselect");
    let v = app.ui.views.get_mut(0).expect("view");
    v.zoom = 1.0;
    v.center = [100.0, 50.0];
    v.fit_pending = false;
    v.fill_pending = false;
    let blue = [0x20, 0x50, 0xd0];
    let red = [0xe0, 0x10, 0x10];
    // Points past the document: right of its right edge (the next copy's red band), below it
    // (the copy underneath), and up-left (the diagonal copy's blue interior).
    let right = [210.0, 50.0];
    let below = [100.0, 150.0];
    let diagonal = [-60.0, -40.0];
    for flip in [false, true] {
        let path = if flip { "CPU" } else { "GPU" };
        h.state_mut().ui.view.flip_horizontal = flip;
        h.state_mut().ui.view.pattern_preview = false;
        let off = sample(&mut h, right);
        assert!(!near(off, red) && !near(off, blue), "{path}: pasteboard without Pattern Preview, got {off:?}");
        assert!(near(sample(&mut h, [10.0, 50.0]), red), "{path}: the document's own band");
        toggle(&mut h);
        assert!(h.state().ui.view.pattern_preview);
        let got = sample(&mut h, right);
        assert!(near(got, red), "{path}: copy to the right shows the left edge, got {got:?}");
        let got = sample(&mut h, below);
        assert!(near(got, blue), "{path}: copy below, got {got:?}");
        let got = sample(&mut h, diagonal);
        assert!(near(got, blue), "{path}: diagonal copy, got {got:?}");
        toggle(&mut h);
        assert!(!h.state().ui.view.pattern_preview);
    }
}
