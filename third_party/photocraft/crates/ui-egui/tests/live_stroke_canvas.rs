//! #189: while a Brush stroke is drawn, the canvas must show the stroke as it will be committed.
//! v0.1 drew a stand-in while dragging, a foreground-coloured egui polyline as wide as the brush,
//! which egui tessellates into hard pie wedges fanning out from the start once the brush is much
//! wider than the pointer's steps (zoomed in). The real stroke only appeared on release.
//!
//! These drive the real app offscreen and compare the screen at the last pointer move with the
//! screen after release, on the GPU canvas and on the CPU canvas (flipped views draw through it),
//! at several zooms, with smoothing, pressure, and 8- and 16-bit documents. Skips when no GPU
//! adapter exists (like `color_managed_canvas.rs`).

use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::control::{ControlRequest, handle};
use serde_json::{Value, json};

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;
type Pixels = Vec<[u8; 4]>;

fn harness() -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(900.0, 640.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(|cc| {
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

/// Runs a control call and the frame that shows it; returns that (call + frame) time in ms.
fn control(h: &mut Harness, method: &str, params: Value) -> f64 {
    let ctx = h.ctx.clone();
    let (req, _rx) = ControlRequest::new(method, params);
    let t0 = std::time::Instant::now();
    handle(h.state_mut(), &ctx, &req);
    h.step();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    h.run_steps(2);
    ms
}

#[derive(Clone, Copy, Debug)]
struct Case {
    zoom: f32,
    smoothing: f32,
    hardness: f32,
    depth: u8,
    pressure: bool,
    /// Flipped view: drawn through the CPU canvas texture instead of the GPU canvas.
    flip: bool,
}

/// A dark blue document with a white line, then a soft black stroke across it. Returns the
/// canvas pixels before the stroke, at its last pointer move, and after release, and whether
/// the GPU canvas drew them.
fn frames(h: &mut Harness, c: Case) -> (Pixels, Pixels, Pixels, bool) {
    let mut times = Vec::new();
    {
        let app = h.state_mut();
        // Close what an earlier case left open.
        while app.session.active().is_some() {
            app.run("file.close", json!({"discard": true})).expect("close");
        }
        app.run("file.new", json!({"width": 300, "height": 200, "depth": c.depth, "background": "white"})).expect("new");
        app.run("edit.fill", json!({"color": "#1a2550"})).expect("fill");
        app.run("paint.stroke", json!({"points": [[60, 60, 1], [240, 140, 1]], "size": 12, "hardness": 1.0, "color": "#ffffff"})).expect("line");
        app.run("tools.setColors", json!({"foreground": [0.0, 0.0, 0.0, 1.0]})).expect("colours");
        let brush = json!({"size": 30, "hardness": c.hardness, "pressureSize": c.pressure, "smoothing": {"amount": c.smoothing}});
        app.run("tools.setBrush", json!({ "brush": brush })).expect("brush");
        app.sync_views();
        app.ui.view.flip_horizontal = c.flip;
    }
    control(h, "ui.set", json!({"tool": "brush", "zoom": c.zoom, "center": [150, 100]}));
    // The smoothing tool option syncs on the first frame with the Brush: set it after.
    h.state_mut().session.tools.brush.smoothing.amount = c.smoothing;
    let canvas = |h: &mut Harness| {
        h.run_steps(3);
        let r = h.state().last_canvas_rect.shrink(4.0);
        let img = h.render().expect("render");
        let mut px = Vec::new();
        for y in r.top() as u32..(r.bottom() as u32).min(img.height()) {
            for x in r.left() as u32..(r.right() as u32).min(img.width()) {
                px.push(img.get_pixel(x, y).0);
            }
        }
        px
    };
    let before = canvas(h);
    let pressure = |t: f64| if c.pressure { 0.3 + 0.7 * t } else { 1.0 };
    control(h, "ui.pointer", json!({"events": [{"kind": "down", "x": 100, "y": 100, "pressure": pressure(0.0)}]}));
    let n = 14;
    let mut last = (100.0, 100.0);
    for i in 1..=n {
        let t = f64::from(i) / f64::from(n);
        last = (100.0 + 100.0 * t, 100.0 + 40.0 * (t * 6.0).sin());
        times.push(control(h, "ui.pointer", json!({"events": [{"kind": "move", "x": last.0, "y": last.1, "pressure": pressure(t)}]})));
    }
    times.sort_by(f64::total_cmp);
    // Pointer move → stroke rendered → frame built (canvas refresh and upload included).
    eprintln!("{c:?}: move-to-frame median {:.2} ms, max {:.2} ms", times[times.len() / 2], times[times.len() - 1]);
    let drawing = canvas(h);
    let gpu = h.state().perf.gpu && !c.flip;
    control(h, "ui.pointer", json!({"events": [{"kind": "up", "x": last.0, "y": last.1}]}));
    let done = canvas(h);
    (before, drawing, done, gpu)
}

fn max_diff(a: &[[u8; 4]], b: &[[u8; 4]]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |i| p[i].abs_diff(q[i]))).max().unwrap_or(0)
}

#[test]
fn the_canvas_shows_the_committed_stroke_while_it_is_drawn() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    let mut cases = Vec::new();
    for zoom in [0.5, 1.0, 4.0, 12.0] {
        for smoothing in [0.0, 0.1, 0.5] {
            cases.push(Case { zoom, smoothing, hardness: 0.0, depth: 8, pressure: false, flip: false });
        }
    }
    cases.push(Case { zoom: 4.0, smoothing: 0.1, hardness: 1.0, depth: 8, pressure: false, flip: false });
    cases.push(Case { zoom: 4.0, smoothing: 0.1, hardness: 0.0, depth: 16, pressure: false, flip: false });
    cases.push(Case { zoom: 4.0, smoothing: 0.1, hardness: 0.0, depth: 8, pressure: true, flip: false });
    cases.push(Case { zoom: 4.0, smoothing: 0.1, hardness: 0.0, depth: 8, pressure: false, flip: true });
    cases.push(Case { zoom: 12.0, smoothing: 0.5, hardness: 0.0, depth: 16, pressure: true, flip: true });
    let mut gpu_seen = false;
    for c in cases {
        let (before, drawing, done, gpu) = frames(&mut h, c);
        gpu_seen |= gpu;
        // The stroke is on screen while drawing (not a blank canvas compared with a blank one)…
        assert!(max_diff(&before, &drawing) > 100, "{c:?}: the stroke shows while drawing");
        // …and is exactly what release leaves there: nothing extra (no stand-in shape), nothing missing.
        let d = max_diff(&drawing, &done);
        assert!(d <= 1, "{c:?}: the canvas changed by {d}/255 on release");
    }
    if !gpu_seen {
        eprintln!("note: the GPU compositor fell back to the CPU for every case");
    }
}
