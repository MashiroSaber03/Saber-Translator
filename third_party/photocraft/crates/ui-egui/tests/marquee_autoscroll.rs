//! #4183: dragging a Rectangular Marquee past the edge of a zoomed-in canvas scrolls the view, so
//! the selection can be larger than what is on screen (as in Photoshop).
//!
//! Before the fix the view stayed put and the selection stopped at the visible edge. The real app
//! runs offscreen on wgpu with real egui pointer events. Skips when no GPU adapter exists (like
//! `drag_preview_canvas.rs`).

use egui::{Modifiers, PointerButton, Pos2};
use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::control::{ControlRequest, handle};
use serde_json::{Value, json};

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

fn harness() -> Option<Harness> {
    harness_with_size(egui::vec2(900.0, 640.0))
}

fn harness_with_size(size: egui::Vec2) -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(size).with_max_steps(64).wgpu().build_eframe(|cc| {
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

/// A 3000 × 2000 document at 100 %, centred, with `tool` active: far larger than the canvas.
fn setup(h: &mut Harness, tool: &str) {
    {
        let app = h.state_mut();
        while app.session.active().is_some() {
            app.run("file.close", json!({"discard": true})).expect("close");
        }
        app.run("file.new", json!({"width": 3000, "height": 2000, "background": "white"})).expect("new");
        app.sync_views();
        app.ui.extras.rulers = false;
    }
    control(h, "ui.set", json!({"tool": tool, "zoom": 1.0, "center": [1500, 1000]}));
    h.run_steps(3);
}

fn xf(h: &Harness) -> ViewXform {
    ViewXform::active(h.state()).expect("an active document view")
}

fn pointer_at(h: &mut Harness, p: Pos2) {
    h.event(egui::Event::PointerMoved(p));
    h.step();
}

fn button_at(h: &mut Harness, p: Pos2, pressed: bool) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    h.step();
}

/// Selection bounds `[x0, y0, x1, y1]` in document pixels.
fn selection(h: &Harness) -> Option<[i32; 4]> {
    let st = h.state().session.active()?;
    let b = st.doc.selection.as_ref()?.content_bounds();
    (!b.is_empty()).then_some([b.x0, b.y0, b.x1, b.y1])
}

#[test]
fn marquee_dragged_past_the_right_edge_scrolls_and_selects_beyond_the_view() {
    let Some(mut h) = harness() else { return };
    setup(&mut h, "rectMarquee");
    let start = xf(&h);
    let rect = start.rect;
    // The document column at the canvas's right edge before the drag: the most a selection
    // could reach without scrolling.
    let visible_right = start.to_doc(egui::pos2(rect.right(), rect.center().y))[0];

    let from = rect.center();
    pointer_at(&mut h, from);
    button_at(&mut h, from, true);
    for i in 1..=6 {
        let t = i as f32 / 6.0;
        pointer_at(&mut h, from + egui::vec2((rect.right() - from.x - 10.0) * t, 60.0 * t));
    }
    // Past the edge, then held still for a while: the view keeps scrolling.
    let outside = egui::pos2(rect.right() + 40.0, from.y + 60.0);
    pointer_at(&mut h, outside);
    for _ in 0..30 {
        h.step();
    }
    let moved = xf(&h).center[0] - start.center[0];
    assert!(moved > 50.0, "the view scrolls right while the drag is held past the edge (moved {moved})");
    assert!((xf(&h).center[1] - start.center[1]).abs() < 1e-3, "and only horizontally");

    button_at(&mut h, outside, false);
    h.run_steps(3);
    let sel = selection(&h).expect("a selection");
    assert!(sel[2] as f64 > visible_right + 50.0, "the selection reaches past the column that was visible at the press: {sel:?} vs {visible_right}");

    // Once released, the view no longer scrolls even with the pointer outside.
    let after = xf(&h).center;
    pointer_at(&mut h, outside);
    h.run_steps(10);
    assert_eq!(xf(&h).center, after);
}

#[test]
fn a_drag_inside_the_canvas_does_not_scroll() {
    let Some(mut h) = harness() else { return };
    setup(&mut h, "rectMarquee");
    let start = xf(&h);
    let from = start.rect.center();
    pointer_at(&mut h, from);
    button_at(&mut h, from, true);
    for i in 1..=6 {
        pointer_at(&mut h, from + egui::vec2(15.0 * i as f32, 10.0 * i as f32));
    }
    for _ in 0..20 {
        h.step();
    }
    assert_eq!(xf(&h).center, start.center);
    button_at(&mut h, from + egui::vec2(90.0, 60.0), false);
}

#[test]
fn rotated_marquee_scrolls_on_the_axis_that_is_visibly_clipped() {
    let Some(mut h) = harness_with_size(egui::vec2(1200.0, 640.0)) else { return };
    setup(&mut h, "rectMarquee");
    let rect = xf(&h).rect;
    assert!(rect.width() > rect.height());
    // Fits horizontally before rotation, but at 90 degrees it overflows vertically.
    let width = ((rect.width() + rect.height()) / 2.0) as u32;
    let height = (rect.width() * 2.0) as u32;
    for overscroll in [true, false] {
        {
            let app = h.state_mut();
            app.run("file.close", json!({"discard": true})).expect("close");
            app.run("file.new", json!({"width": width, "height": height, "background": "white"})).expect("new");
            app.run("prefs.set", json!({"path": "tools.overscroll", "value": overscroll})).expect("overscroll");
            app.sync_views();
        }
        control(&mut h, "ui.set", json!({"tool": "rectMarquee", "zoom": 1.0, "center": [width / 2, height / 2]}));
        h.state_mut().ui.views[0].rotation = 90.0;
        h.run_steps(3);
        let start = xf(&h);
        let from = start.rect.center();
        let outside = egui::pos2(from.x + 30.0, start.rect.bottom() + 40.0);
        pointer_at(&mut h, from);
        button_at(&mut h, from, true);
        pointer_at(&mut h, outside);
        h.run_steps(30);
        let moved = xf(&h).center[0] - start.center[0];
        if let Some(dir) = std::env::var_os("CRAFT_UI_VISUAL_FIXTURES") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("fixture directory");
            h.render().expect("render").save(dir.join(format!("rotated-marquee-overscroll-{overscroll}.png"))).expect("save fixture");
        }
        assert!(moved > 10.0, "rotated vertical overflow must scroll document x, overscroll={overscroll}, moved={moved}");
        let limit = width as f32 - start.rect.height() / (2.0 * start.zoom);
        assert!((xf(&h).center[0] - limit).abs() < 0.01, "stops at the rotated image edge, overscroll={overscroll}");
        assert!((xf(&h).center[1] - start.center[1]).abs() < 0.01);
        button_at(&mut h, outside, false);
        h.run_steps(3);
        let after = xf(&h).center;
        h.run_steps(10);
        assert_eq!(xf(&h).center, after);
    }
}
