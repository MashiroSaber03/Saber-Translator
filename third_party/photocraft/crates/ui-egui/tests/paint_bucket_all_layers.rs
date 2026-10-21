//! #1118: Paint Bucket's All Layers checkbox must sample the visible image while painting
//! only the active layer. Drive the checkbox and canvas through real egui pointer events.

use egui::{Modifiers, PointerButton, Pos2};
use egui_kittest::kittest::Queryable;
use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::control::{ControlRequest, Outcome, handle};
use serde_json::json;

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

fn harness() -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(1100.0, 720.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
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

fn setup(h: &mut Harness) -> LayerId {
    let background = {
        let app = h.state_mut();
        while app.session.active().is_some() {
            app.run("file.close", json!({"discard": true})).expect("close");
        }
        app.run("file.new", json!({"width": 400, "height": 300, "background": "white"})).expect("new");
        let background = app.session.active().expect("document").active_layer.expect("background");
        app.run("select.rect", json!({"x": 0, "y": 0, "width": 200, "height": 300})).expect("select left half");
        app.run("edit.fill", json!({"contents": "black"})).expect("black left half");
        app.run("select.deselect", json!({})).expect("deselect");
        app.run("layer.new.layer", json!({"name": "Bucket fill"})).expect("empty active layer");
        app.run("tools.setColors", json!({"foreground": [1.0, 0.0, 0.0, 1.0]})).expect("red foreground");
        app.sync_views();
        app.ui.extras.rulers = false;
        let o = &mut app.ui.tool_options;
        o.tolerance = 0.0;
        o.anti_alias = false;
        o.contiguous = true;
        o.sample_all_layers = false;
        o.bucket_fill_pattern = false;
        o.fill_opacity = 100.0;
        background
    };
    let ctx = h.ctx.clone();
    let (req, _rx) = ControlRequest::new("ui.set", json!({"tool": "paintBucket", "zoom": 1.0, "center": [200, 150]}));
    assert!(matches!(handle(h.state_mut(), &ctx, &req), Outcome::Done(_)));
    h.run_steps(6);
    background
}

fn screen(h: &Harness, x: f32, y: f32) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    ViewXform {
        rect: photocraft_ui_egui::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    }
    .to_screen(x, y)
}

fn pixels(h: &Harness, layer: LayerId) -> Vec<[f32; 4]> {
    let st = h.state().session.active().expect("document");
    let surface = st.doc.layer(layer).expect("layer").surface().expect("pixel layer");
    let mut pixels = vec![[0.0; 4]; 400 * 300];
    surface.read_rgba_into(Rect::new(0, 0, 400, 300), &mut pixels);
    pixels
}

#[test]
fn all_layers_checkbox_limits_bucket_fill_to_the_visible_color() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    for all_layers in [false, true] {
        let background = setup(&mut h);
        let before = pixels(&h, background);
        if all_layers {
            h.get_by_label("All Layers").click();
            h.run_steps(3);
        }
        assert_eq!(h.state().ui.tool_options.sample_all_layers, all_layers, "checkbox changed the option");

        let pos = screen(&h, 300.0, 150.0);
        h.event(egui::Event::PointerMoved(pos));
        h.step();
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
            h.step();
        }
        h.run_steps(6);

        // Optional real UI evidence, saved before assertions so a failing baseline has a frame.
        if let Some(dir) = std::env::var_os("PHOTOCRAFT_BUCKET_EVIDENCE_DIR") {
            std::fs::create_dir_all(&dir).expect("evidence directory");
            let name = if all_layers { "all-layers-on.png" } else { "all-layers-off.png" };
            h.render().expect("render UI").save(std::path::Path::new(&dir).join(name)).expect("save UI evidence");
        }

        assert_eq!(pixels(&h, background), before, "bucket must leave the sampled background unchanged");
        let active = h.state().session.active().expect("document").active_layer.expect("active layer");
        for (i, pixel) in pixels(&h, active).into_iter().enumerate() {
            let x = i % 400;
            let expected = if all_layers && x < 200 { [0.0; 4] } else { [1.0, 0.0, 0.0, 1.0] };
            assert_eq!(pixel, expected, "All Layers={all_layers}, active-layer pixel ({x}, {})", i / 400);
        }
    }
}
