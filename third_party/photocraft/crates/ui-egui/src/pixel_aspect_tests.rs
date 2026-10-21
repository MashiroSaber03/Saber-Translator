//! View › Pixel Aspect Ratio (#2594): with correction on, the canvas shows each document pixel
//! that many times wider, and the pointer still lands on the pixel drawn under it. The document's
//! pixels never change.

use egui::{Modifiers, PointerButton, Pos2, Rect, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 100, "background": "transparent"})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#000000"})).unwrap();
    app.ui.tool = Tool::Pencil;
    crate::paint_mouse::sync_tool_brush(&mut app);
    app.run("tools.setBrush", json!({"brush": {"size": 1, "smoothing": {"amount": 0.0}}})).unwrap();
    app.sync_views();
    let mut h = Harness::builder().with_size(vec2(1200.0, 800.0)).with_step_dt(1.0 / 60.0).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

fn menu(h: &mut Harness<'static, PhotocraftApp>, id: &str, params: serde_json::Value) {
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, id, params).unwrap();
    h.run_steps(3);
}

/// The image's screen rectangle, as the canvas draws it.
fn image_rect(h: &Harness<'static, PhotocraftApp>) -> Rect {
    let app = h.state();
    let doc = &app.session.active().unwrap().doc;
    ViewXform::active(app).unwrap().doc_rect(doc.bounds())
}

fn click(h: &mut Harness<'static, PhotocraftApp>, pos: Pos2) {
    h.event(egui::Event::PointerMoved(pos));
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(2);
    }
}

fn alpha(h: &Harness<'static, PhotocraftApp>, x: i32, y: i32) -> f32 {
    let st = h.state().session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)[3]
}

#[test]
fn the_mapping_stretches_document_x_and_inverts_exactly() {
    let rect = Rect::from_min_size(Pos2::new(10.0, 20.0), vec2(800.0, 600.0));
    let square = ViewXform { rect, zoom: 1.5, center: [100.0, 50.0], flip: false, rotation: 0.0, aspect: 1.0 };
    let wide = ViewXform { aspect: 2.0, ..square };
    let r = wide.doc_rect(photocraft_geom::Rect::new(0, 0, 200, 100));
    assert!((r.width() - 600.0).abs() < 1e-3 && (r.height() - 150.0).abs() < 1e-3, "{r:?}");
    assert_eq!(r.center(), square.doc_rect(photocraft_geom::Rect::new(0, 0, 200, 100)).center());
    assert!((wide.zoom_x() - 3.0).abs() < 1e-6);
    for flip in [false, true] {
        for rotation in [0.0, 30.0, -90.0] {
            for aspect in [0.91, 1.0, 2.0] {
                let xf = ViewXform { flip, rotation, aspect, ..square };
                let p = xf.to_screen(37.0, 81.0);
                let d = xf.to_doc(p);
                assert!((d[0] - 37.0).abs() < 1e-3 && (d[1] - 81.0).abs() < 1e-3, "{flip} {rotation} {aspect}: {d:?}");
                // Pan and zoom-about deltas agree with the point mapping.
                let q = xf.to_screen(40.0, 85.0);
                let u = xf.unmap_vec(q - p) / xf.zoom;
                assert!((u.x - 3.0).abs() < 1e-3 && (u.y - 4.0).abs() < 1e-3, "{flip} {rotation} {aspect}: {u:?}");
            }
        }
    }
    // A hostile ratio draws square instead of collapsing the image.
    for aspect in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let xf = ViewXform { aspect, ..square };
        assert_eq!(xf.to_screen(5.0, 5.0), square.to_screen(5.0, 5.0));
    }
}

#[test]
fn anamorphic_ratio_draws_the_canvas_twice_as_wide_and_correction_off_restores_it() {
    let mut h = harness();
    let square = image_rect(&h);
    assert!((square.width() / square.height() - 2.0).abs() < 0.02, "{square:?}");
    menu(&mut h, "view.pixelAspectRatio.anamorphic2To1", json!({}));
    assert!(h.state().ui.view.pixel_aspect_correction, "correction is on by default, as in Photoshop");
    let wide = image_rect(&h);
    assert!((wide.width() / wide.height() - 4.0).abs() < 0.02, "{wide:?}");
    assert!((wide.height() - square.height()).abs() < 0.5, "only the width changes: {square:?} {wide:?}");
    // The document itself keeps its pixels.
    let doc = &h.state().session.active().unwrap().doc;
    assert_eq!([doc.size.width, doc.size.height], [200, 100]);
    // View › Pixel Aspect Ratio Correction off: square pixels again.
    menu(&mut h, "view.pixelAspectRatioCorrection", json!({}));
    assert!(!h.state().ui.view.pixel_aspect_correction);
    let off = image_rect(&h);
    assert!((off.width() / off.height() - 2.0).abs() < 0.02, "{off:?}");
}

#[test]
fn a_click_paints_the_pixel_drawn_under_the_pointer() {
    let mut h = harness();
    menu(&mut h, "view.pixelAspectRatio.anamorphic2To1", json!({}));
    let xf = ViewXform::active(h.state()).unwrap();
    let at = xf.to_screen(120.5, 30.5);
    click(&mut h, at);
    assert_eq!(alpha(&h, 120, 30), 1.0, "painted under the pointer");
    // A square mapping of the same screen point is a different pixel, left blank.
    let square = ViewXform { aspect: 1.0, ..xf }.to_doc(at);
    let (sx, sy) = (square[0].floor() as i32, square[1].floor() as i32);
    assert_ne!((sx, sy), (120, 30));
    assert_eq!(alpha(&h, sx, sy), 0.0);
}

#[test]
fn fit_and_fill_on_screen_use_the_stretched_width() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 1000, "height": 500})).unwrap();
    let doc = app.session.active().unwrap().doc.clone();
    // Fit keeps 20 points of margin per side: 1000 points across for 1000 px wide, 2000 shown.
    let area = vec2(1040.0, 800.0);
    let mut view = crate::state::View::default();
    crate::canvas::fit_view(&mut view, &doc, area, 1.0, 1.0);
    assert_eq!(view.zoom, 1.0);
    crate::canvas::fit_view(&mut view, &doc, area, 1.0, 2.0);
    assert_eq!(view.zoom, 0.5);
    assert_eq!(view.center, [500.0, 250.0]);
    // Fill: the stretched image is wider than tall, so the height fills.
    crate::canvas::fill_view(&mut view, &doc, vec2(1000.0, 1000.0), 1.0, 2.0);
    assert_eq!(view.zoom, 2.0);
}
