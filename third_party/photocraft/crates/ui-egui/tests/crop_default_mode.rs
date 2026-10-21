//! Crop tool, Photoshop's default mode (#1919), through the real canvas: egui pointer and key
//! events on the full app UI, frames in between, as a user's drag arrives.
//!
//! In the default mode the view moves and turns with the image while the box stays put on
//! screen. The canvas works on a copy of the view for the frame and writes it back at the end, so
//! a view change made by the Crop tool during the frame used to be lost: a drag inside the box
//! made the box wander instead of moving the image, and a resize with Auto Center Preview didn't
//! keep the box's centre in place. Tests driving `tool_event` directly never saw that.
//!
//! Skips when no GPU
//! adapter exists (like `crop_beyond_canvas.rs`).

use egui::{Key, Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::kittest::Queryable;
use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::control::{ControlRequest, handle};
use serde_json::{Value, json};

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

fn harness() -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(900.0, 640.0)).with_max_steps(64).wgpu().build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            // The queries below use English labels; don't follow the system locale.
            app.session.prefs.edit(|p| p.interface.language = "en".into());
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

fn xf(h: &Harness) -> ViewXform {
    ViewXform::active(h.state()).expect("a view")
}

/// The crop box's corners on screen.
fn screen_box(h: &Harness) -> [Pos2; 4] {
    let app = h.state();
    let r = app.ui.crop_rect.expect("a frame");
    let x = xf(h);
    photocraft_ui_egui::crop_ui::corners(r, photocraft_ui_egui::crop_ui::angle(app)).map(|p| x.to_screen(p[0] as f32, p[1] as f32))
}

fn mid(q: [Pos2; 4]) -> Pos2 {
    pos2((q[0].x + q[2].x) / 2.0, (q[0].y + q[2].y) / 2.0)
}

fn press(h: &mut Harness, p: Pos2, pressed: bool) {
    h.event(egui::Event::PointerMoved(p));
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    h.step();
}

/// A drag in screen points, `steps` frames of moves; `each` runs after every move frame.
fn drag(h: &mut Harness, from: Pos2, to: Pos2, steps: usize, mut each: impl FnMut(&mut Harness, usize)) {
    press(h, from, true);
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        h.event(egui::Event::PointerMoved(from + (to - from) * t));
        h.step();
        each(h, i);
    }
    press(h, to, false);
    h.run_steps(2);
}

fn setup() -> Option<Harness> {
    let mut h = harness()?;
    h.run_steps(4);
    {
        let app = h.state_mut();
        app.run("file.new", json!({"width": 300, "height": 200, "background": "white"})).expect("new");
        app.run("select.rect", json!({"x": 100, "y": 50, "width": 60, "height": 60})).expect("select");
        app.run("edit.fill", json!({"color": "#ff0000"})).expect("fill");
        app.run("select.deselect", json!({})).expect("deselect");
        app.sync_views();
        app.ui.extras.rulers = false;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
    }
    control(&mut h, "ui.set", json!({"tool": "crop", "zoom": 1.0, "center": [150, 100]}));
    h.run_steps(3);
    let s = &h.state().ui.tool_options.crop_shield;
    assert!(!s.classic_mode && s.auto_center_preview, "Photoshop's defaults");
    Some(h)
}

#[test]
fn a_ratio_dropdown_updates_the_preview_and_enter_without_touching_a_handle() {
    // Windows WARP can access-violate when GPU harnesses initialize or tear down concurrently:
    // hold the GPU lock for the whole test, harness included.
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = setup() else { return };
    h.get_by_value("Ratio").click();
    h.run_steps(3);
    h.get_by_label("1 : 1 (Square)").click();
    h.run_steps(3);
    assert_eq!(h.state().ui.crop_rect, Some([50.0, 0.0, 250.0, 200.0]));
    if let Some(dir) = std::env::var_os("CRAFT_UI_VISUAL_FIXTURES") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        h.render().unwrap().save(dir.join("photocraft-crop-square.png")).unwrap();
    }
    h.key_press(Key::Enter);
    h.run_steps(3);
    assert_eq!(h.state().session.active().unwrap().doc.size, photocraft_doc::Size::new(200, 200));
}

#[test]
fn the_swap_button_and_x_apply_the_same_inscribed_ratio() {
    // Windows WARP can access-violate when GPU harnesses initialize or tear down concurrently:
    // hold the GPU lock for the whole test, harness included.
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = setup() else { return };
    let original = [30.0, 20.0, 270.0, 180.0];
    h.state_mut().ui.crop_rect = Some(original);
    h.state_mut().ui.tool_options.crop_ratio = "3:2".into();
    h.run_steps(3);
    let before = mid(screen_box(&h));
    h.get_by_label("Swaps height and width").click();
    h.run_steps(3);
    let button = h.state().ui.crop_rect.unwrap();
    assert_eq!(h.state().ui.tool_options.crop_ratio, "2:3");
    assert!(((button[2] - button[0]) / (button[3] - button[1]) - 2.0 / 3.0).abs() < 1e-10);
    assert!(button[0] >= original[0] && button[2] <= original[2] && button[1] >= original[1] && button[3] <= original[3]);
    assert!(mid(screen_box(&h)).distance(before) < 0.01);
    h.state_mut().ui.crop_rect = Some(original);
    h.state_mut().ui.tool_options.crop_ratio = "3:2".into();
    h.run_steps(3);
    h.key_press(Key::X);
    h.run_steps(3);
    assert_eq!(h.state().ui.crop_rect, Some(button));
    assert!(mid(screen_box(&h)).distance(before) < 0.01);
}

#[test]
fn a_resize_renders_the_frame_and_image_with_the_same_camera() {
    // Windows WARP can access-violate when GPU harnesses initialize or tear down concurrently:
    // hold the GPU lock for the whole test, harness included.
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = setup() else { return };
    let box0 = screen_box(&h);
    let c = mid(box0);
    let right = pos2(box0[1].x, c.y);
    drag(&mut h, right, right - vec2(48.0, 0.0), 6, |h, i| {
        let img = h.render().expect("render");
        let b = screen_box(h);
        // The left edge also moves inward with Auto Center Preview. Sample either side
        // away from the handles and the red patch: white inside, dimmed white outside.
        let y = c.y + 70.0;
        for x in [b[0].x - 5.0, b[1].x + 5.0] {
            let px = img.get_pixel(x.round() as u32, y.round() as u32);
            assert!(px[0] < 230 && px[1] < 230 && px[2] < 230, "frame {i}, outside at {x}: {px:?}");
        }
        for x in [b[0].x + 5.0, b[1].x - 5.0] {
            let px = img.get_pixel(x.round() as u32, y.round() as u32);
            assert!(px[0] > 245 && px[1] > 245 && px[2] > 245, "frame {i}, inside at {x}: {px:?}");
        }
    });
}

#[test]
fn dragging_inside_moves_the_image_and_the_box_stays_centred() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = setup() else { return };
    assert!(!photocraft_ui_egui::crop_ui::pending(h.state()), "the untouched frame");
    let c = h.state().last_canvas_rect.center();
    // Inside the untouched frame a drag draws a new box; Auto Center Preview centres it.
    drag(&mut h, c + vec2(-100.0, -60.0), c + vec2(60.0, 40.0), 6, |_, _| {});
    assert!(photocraft_ui_egui::crop_ui::pending(h.state()), "a real frame now");
    let box0 = screen_box(&h);
    assert!(mid(box0).distance(c) < 1.0, "the drawn box is centred: {:?} vs {c:?}", mid(box0));
    let r0 = h.state().ui.crop_rect.expect("frame");
    let red0 = xf(&h).to_screen(130.0, 80.0);
    // Inside the box: the image moves with the pointer, the box stays put.
    let (from, to) = (mid(box0) + vec2(-20.0, -10.0), mid(box0) + vec2(40.0, 30.0));
    drag(&mut h, from, to, 8, |h, i| {
        let b = screen_box(h);
        assert!(mid(b).distance(mid(box0)) < 0.5, "frame {i}: the box stays put: {:?} vs {:?}", mid(b), mid(box0));
        // The image follows the pointer.
        let t = i as f32 / 8.0;
        let want = red0 + (to - from) * t;
        let red = xf(h).to_screen(130.0, 80.0);
        assert!(red.distance(want) < 0.5, "frame {i}: the image under the pointer: {red:?} vs {want:?}");
    });
    let b = screen_box(&h);
    assert!(mid(b).distance(mid(box0)) < 0.5, "after the release: {:?} vs {:?}", mid(b), mid(box0));
    let r = h.state().ui.crop_rect.expect("frame");
    assert!((r[0] - (r0[0] - 60.0)).abs() < 0.5 && (r[1] - (r0[1] - 40.0)).abs() < 0.5, "the frame lies 60, 40 up-left on the image: {r:?} (was {r0:?})");
    assert_eq!((r[2] - r[0], r[3] - r[1]), (r0[2] - r0[0], r0[3] - r0[1]), "same size");
}

#[test]
fn a_resize_keeps_the_box_centre_after_a_space_pan() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = setup() else { return };
    // Space held: the temporary Hand pans the view 80, 40.
    let c = h.state().last_canvas_rect.center();
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.step();
    drag(&mut h, c + vec2(0.0, 200.0), c + vec2(80.0, 240.0), 4, |_, _| {});
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    let box0 = screen_box(&h);
    assert!(mid(box0).distance(c + vec2(80.0, 40.0)) < 1.0, "panned: {:?}", mid(box0));
    // The right edge, dragged 48 out over several frames (8 a frame: past egui's drag threshold).
    let right = pos2(box0[1].x, (box0[1].y + box0[2].y) / 2.0);
    drag(&mut h, right, right + vec2(48.0, 0.0), 6, |h, i| {
        let b = screen_box(h);
        assert!(mid(b).distance(mid(box0)) < 0.5, "frame {i}: the centre stays put: {:?} vs {:?}", mid(b), mid(box0));
        let want = right.x + 48.0 * i as f32 / 6.0;
        assert!((b[1].x - want).abs() < 0.5, "frame {i}: the handle follows the pointer: {} vs {want}", b[1].x);
    });
    // On the image the left edge stayed put: the frame grew 96 to the right.
    let r = h.state().ui.crop_rect.expect("frame");
    assert!((r[0] - 0.0).abs() < 0.5 && (r[2] - 396.0).abs() < 0.5, "{r:?}");
    // On release the box stays at the panned position (Photoshop doesn't snap it back to the
    // middle).
    let b = screen_box(&h);
    assert!(mid(b).distance(mid(box0)) < 0.5, "still where it was: {:?} vs {:?}", mid(b), mid(box0));
}
