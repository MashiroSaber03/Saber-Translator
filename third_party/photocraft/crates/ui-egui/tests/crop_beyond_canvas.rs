//! Crop tool: while a frame is edited, the canvas shows the layers' pixels past its edges, as
//! Photoshop's crop preview does.
//!
//! A crop with Delete Cropped Pixels off keeps what it cut away, but PhotoCraft drew only the
//! canvas, so a frame couldn't be pulled back out over those pixels by eye. Photoshop (measured on
//! 25.4) grows the view to every layer's bounds and the frame once the frame is pressed: the
//! hidden pixels under the shield, transparency as the checkerboard, until ↵ or Esc.
//!
//! The real app runs offscreen on wgpu with real egui pointer events; the test reads the rendered
//! pixels. Skips when no GPU adapter exists (like `drag_preview_canvas.rs`).

use egui::{Key, Modifiers, PointerButton, Pos2};
use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::control::{ControlRequest, handle};
use serde_json::{Value, json};

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

/// A rendered frame: RGBA8, physical pixels.
struct Image {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Image {
    fn rgb(&self, p: Pos2) -> [u8; 3] {
        let (x, y) = (p.x.round() as i64, p.y.round() as i64);
        assert!(x >= 0 && y >= 0 && x < i64::from(self.w) && y < i64::from(self.h), "{p:?} is off the frame");
        let i = (y as usize * self.w as usize + x as usize) * 4;
        [self.px[i], self.px[i + 1], self.px[i + 2]]
    }
}

fn harness() -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(900.0, 640.0)).with_max_steps(64).wgpu().build_eframe(|cc| {
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

fn control(h: &mut Harness, method: &str, params: Value) {
    let ctx = h.ctx.clone();
    let (req, _rx) = ControlRequest::new(method, params);
    handle(h.state_mut(), &ctx, &req);
    h.run_steps(3);
}

fn screen(h: &Harness, x: f32, y: f32) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    let xf = ViewXform {
        rect: photocraft_ui_egui::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    };
    xf.to_screen(x, y)
}

fn button(h: &mut Harness, x: f32, y: f32, pressed: bool) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerMoved(p));
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    h.step();
}

fn drag(h: &mut Harness, from: [f32; 2], to: [f32; 2]) {
    button(h, from[0], from[1], true);
    for i in 1..=6 {
        let t = i as f32 / 6.0;
        h.event(egui::Event::PointerMoved(screen(h, from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t)));
        h.step();
    }
    button(h, to[0], to[1], false);
}

fn render(h: &mut Harness) -> Image {
    let img = h.render().expect("render");
    Image { w: img.width(), h: img.height(), px: img.into_raw() }
}

/// Mostly `c` (one channel well above the other two), dimmed or not.
fn tinted(p: [u8; 3], c: usize) -> bool {
    (0..3).filter(|&i| i != c).all(|i| p[c] > 60 && p[i] < p[c] / 2)
}

#[test]
fn editing_a_crop_frame_shows_the_pixels_past_the_canvas() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    {
        let app = h.state_mut();
        // 300 × 120: red down the left 60 px and blue down the right 60 px of a new layer, then
        // cropped to (50, 0) 200 × 100 keeping the pixels: 50 px of red past the left edge, 50 of
        // blue past the right, and 20 rows of the white Background below.
        app.run("file.new", json!({"width": 300, "height": 120, "background": "white"})).expect("new");
        app.run("layer.new.layer", json!({})).expect("layer");
        for (x, color) in [(0, "#ff0000"), (240, "#0000ff")] {
            app.run("select.rect", json!({"x": x, "y": 0, "width": 60, "height": 120})).expect("select");
            app.run("edit.fill", json!({"color": color})).expect("fill");
        }
        app.run("select.deselect", json!({})).expect("deselect");
        app.run("image.crop", json!({"x": 50, "y": 0, "width": 200, "height": 100, "deleteCroppedPixels": false})).expect("crop");
        app.sync_views();
        app.ui.extras.rulers = false;
        app.ui.extras.snap = false;
    }
    // Classic Mode: the edge drag below moves only that edge (Auto Center Preview in the default
    // mode would keep the box centred, growing it on both sides).
    control(&mut h, "ui.set", json!({"tool": "crop", "zoom": 1.0, "center": [100, 50], "cropShield": {"classic_mode": true}}));
    h.run_steps(3);
    assert_eq!(h.state().ui.crop_rect, Some([0.0, 0.0, 200.0, 100.0]), "the default frame");
    let (red, blue, white, beyond) = ([-25.0, 50.0], [225.0, 50.0], [100.0, 110.0], [-80.0, 50.0]);
    let at = |h: &Harness, img: &Image, p: [f32; 2]| img.rgb(screen(h, p[0], p[1]));

    // Picking the tool shows only the canvas (and the pasteboard around it, dimmed by the shield).
    let before = render(&mut h);
    let shown = |h: &Harness, img: &Image| [red, blue, white, beyond].map(|p| at(h, img, p));
    let was = shown(&h, &before);
    assert!(!tinted(was[0], 0) && !tinted(was[1], 2), "nothing past the canvas yet: {was:?}");

    // A press on the frame starts editing, before any move: the pixels past the canvas show,
    // under the shield (the press asks for the frame that draws them).
    button(&mut h, 100.0, 50.0, true);
    h.step();
    let img = render(&mut h);
    let [r, b, w, past] = shown(&h, &img);
    assert!(tinted(r, 0), "hidden red: {r:?}");
    assert!(tinted(b, 2), "hidden blue: {b:?}");
    assert!(w.iter().all(|&c| c > 90 && c < 200) && w[0] == w[1] && w[1] == w[2] && w != was[2], "hidden white, dimmed: {w:?}");
    assert_eq!(past, was[3], "past every layer and the frame: the pasteboard");
    // Inside the frame nothing changed.
    assert_eq!(at(&h, &img, [150.0, 30.0]), at(&h, &before, [150.0, 30.0]));
    // Releasing (a click on the untouched frame) keeps the crop being edited.
    button(&mut h, 100.0, 50.0, false);
    let img = render(&mut h);
    assert!(tinted(at(&h, &img, red), 0), "still shown after the click");

    // The left edge pulled out past the red: the red inside the frame is undimmed, and where
    // there are no pixels the frame shows transparency.
    drag(&mut h, [0.0, 50.0], [-80.0, 50.0]);
    assert_eq!(h.state().ui.crop_rect, Some([-80.0, 0.0, 200.0, 100.0]));
    let img = render(&mut h);
    let r = at(&h, &img, red);
    assert!(r[0] > 230 && r[1] < 40 && r[2] < 40, "red inside the frame at full strength: {r:?}");
    // (egui_kittest samples textures through its own clamped bilinear filter, so the checkerboard
    // shows as a light, neutral grey here rather than as squares.)
    for p in [[-70.0, 50.0], [-62.0, 80.0]] {
        let c = at(&h, &img, p);
        assert!(c.iter().all(|&v| v >= 190) && c[0] == c[1] && c[1] == c[2], "transparency past the layers at {p:?}: {c:?}");
    }
    // The red runs on across the canvas edge: no canvas border line between.
    for x in [-2.0, -1.0, 0.0, 1.0] {
        let c = at(&h, &img, [x, 50.0]);
        assert!(c[0] > 230 && c[1] < 40, "at x = {x}: {c:?}");
    }

    // Esc cancels: back to the canvas alone.
    h.event(egui::Event::PointerGone);
    h.event(egui::Event::Key { key: Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    let img = render(&mut h);
    assert_eq!(h.state().ui.crop_rect, Some([0.0, 0.0, 200.0, 100.0]));
    assert_eq!(shown(&h, &img), was, "hidden again after Esc");
}
