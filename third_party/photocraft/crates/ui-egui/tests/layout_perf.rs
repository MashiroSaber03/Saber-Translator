//! Layout documents (#125, #128) through the real app: the Move tool shows the layers at the
//! pointer while dragging and commits one `layer.translate` (one undo step) that lands where the
//! pointer went; Auto-Select takes the layer under the pointer; selecting a layer recomposites
//! nothing and hiding one recomposites only its area; and the GPU compositor (which composites
//! small layers over their bounds only) matches the CPU reference on the whole document.

#[path = "support/layout_doc.rs"]
mod layout_doc;

use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_engine::layer_multi_cmds::{layer_bounds, move_targets, moved};
use photocraft_geom::Rect;
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::state::Tool;
use photocraft_ui_egui::{PhotocraftApp, Services};
use serde_json::json;

type H = Harness<'static, PhotocraftApp>;

/// The app (CPU canvas) with the small layout document open.
fn app() -> (H, layout_doc::Handles) {
    let (doc, handles) = layout_doc::build(layout_doc::Spec::small());
    let h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).with_pixels_per_point(1.0).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
        app.session.open_document(doc, None);
        app.sync_views();
        app
    });
    let mut h = h;
    h.run_steps(6);
    (h, handles)
}

fn doc(h: &H) -> std::sync::Arc<Document> {
    h.state().session.active().expect("document").doc.clone()
}

fn bounds(h: &H, id: LayerId) -> Rect {
    layer_bounds(doc(h).layer(id).expect("layer")).expect("bounds")
}

fn screen(h: &H, p: [f64; 2]) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    let rect = photocraft_ui_egui::rulers::content_rect(app, app.last_canvas_rect);
    ViewXform { rect, zoom: v.zoom, center: v.center, flip: false, rotation: v.rotation, aspect: 1.0 }.to_screen(p[0] as f32, p[1] as f32)
}

/// A point where `id` (or, for a group, one of its layers) is the topmost layer with pixels.
fn point_on(h: &H, id: LayerId) -> [f64; 2] {
    fn holds(l: &photocraft_doc::Layer, id: LayerId) -> bool {
        l.id == id || l.children().is_some_and(|c| c.iter().any(|c| holds(c, id)))
    }
    let d = doc(h);
    let b = bounds(h, id);
    let layer = d.layer(id).expect("layer");
    for y in b.y0..b.y1 {
        for x in b.x0..b.x1 {
            if photocraft_engine::pick_cmds::layers_at(&d, x, y).first().is_some_and(|&top| holds(layer, top)) {
                return [x as f64 + 0.5, y as f64 + 0.5];
            }
        }
    }
    panic!("layer {id:?} has no visible pixel");
}

/// Press at document point `from`, move by `d` in `steps` frames; returns before releasing.
fn press_and_drag(h: &mut H, from: [f64; 2], d: [f64; 2], steps: usize) {
    let p0 = screen(h, from);
    h.event(Event::PointerMoved(p0));
    h.step();
    h.event(Event::PointerButton { pos: p0, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.step();
    for i in 1..=steps {
        let f = i as f64 / steps as f64;
        h.event(Event::PointerMoved(screen(h, [from[0] + d[0] * f, from[1] + d[1] * f])));
        h.step();
    }
}

fn release(h: &mut H, at: [f64; 2]) {
    let p = screen(h, at);
    h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

/// Zoom so one document pixel is one screen point (pointer deltas are then exact).
fn actual_pixels(h: &mut H) {
    let app = h.state_mut();
    let size = app.session.active().map(|s| s.doc.size).expect("doc");
    app.ui.views[0].zoom = 1.0;
    app.ui.views[0].center = [size.width as f32 / 2.0, size.height as f32 / 2.0];
    app.ui.views[0].fit_pending = false;
    // View › Snap and smart guides would pull the layers onto their neighbours' edges.
    app.ui.extras.snap = false;
    app.ui.view.show.smart_guides = false;
    h.run_steps(2);
}

#[test]
fn move_drag_follows_the_pointer_and_is_one_undo_step() {
    let (mut h, k) = app();
    actual_pixels(&mut h);
    h.state_mut().ui.tool = Tool::Move;
    h.state_mut().ui.tool_options.move_auto_select = false;
    for id in [k.text, k.shape, k.group, k.smart, k.pixel] {
        h.state_mut().run("layer.select", json!({"layer": id.0})).expect("select");
        h.run_steps(2);
        let before = bounds(&h, id);
        let undo = h.state().session.active().expect("doc").history.entries().len();
        let from = point_on(&h, id);
        press_and_drag(&mut h, from, [40.0, 25.0], 6);
        // While dragging the document is untouched (no history step per pointer move) and the
        // canvas shows the moved layers.
        assert_eq!(bounds(&h, id), before, "the drag commits nothing until release");
        assert_eq!(h.state().session.active().expect("doc").history.entries().len(), undo);
        assert!(h.state().perf.spans.contains_key("move preview"), "the canvas shows the layers moving");
        release(&mut h, [from[0] + 40.0, from[1] + 25.0]);
        assert_eq!(bounds(&h, id), before.translate(40, 25), "the layer lands where the pointer went");
        assert_eq!(h.state().session.active().expect("doc").history.entries().len(), undo + 1, "one undo step");
        h.state_mut().run("edit.undo", json!({})).expect("undo");
        assert_eq!(bounds(&h, id), before);
    }
}

#[test]
fn auto_select_moves_the_layer_under_the_pointer() {
    let (mut h, k) = app();
    actual_pixels(&mut h);
    h.state_mut().ui.tool = Tool::Move;
    h.state_mut().ui.tool_options.move_auto_select = true;
    // The default target is Group (#4056); these layers sit in groups, and the test is about the
    // layer under the pointer.
    h.state_mut().ui.tool_options.move_target = "layer".into();
    h.state_mut().run("layer.select", json!({"layer": k.pixel.0})).expect("select");
    h.run_steps(2);
    let (pixel_before, text_before) = (bounds(&h, k.pixel), bounds(&h, k.text));
    let from = point_on(&h, k.text);
    press_and_drag(&mut h, from, [-15.0, 10.0], 4);
    release(&mut h, [from[0] - 15.0, from[1] + 10.0]);
    assert_eq!(h.state().session.active().and_then(|s| s.active_layer), Some(k.text));
    assert_eq!(bounds(&h, k.text), text_before.translate(-15, 10));
    assert_eq!(bounds(&h, k.pixel), pixel_before, "the previously selected layer stays");
}

#[test]
fn a_move_tool_click_selects_without_moving() {
    // Snapping stays on (the default): it used to pull the release point onto a nearby edge, so
    // a plain click nudged the layer (and recomposited the whole document).
    let (mut h, k) = app();
    h.state_mut().ui.tool = Tool::Move;
    h.state_mut().ui.tool_options.move_auto_select = true;
    h.state_mut().ui.tool_options.move_target = "layer".into();
    for id in [k.text, k.shape, k.pixel, k.text] {
        let before = (bounds(&h, id), h.state().session.active().expect("doc").history.entries().len());
        let p = screen(&h, point_on(&h, id));
        h.event(Event::PointerMoved(p));
        h.step();
        h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.step();
        h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(2);
        assert_eq!(h.state().session.active().and_then(|s| s.active_layer), Some(id), "the click selects the layer under it");
        assert_eq!((bounds(&h, id), h.state().session.active().expect("doc").history.entries().len()), before, "and moves nothing");
    }
}

#[test]
fn selecting_recomposites_nothing_and_hiding_only_the_layer() {
    let (mut h, k) = app();
    for id in [k.text, k.group, k.smart, k.shape] {
        h.state_mut().perf.last_refresh = "untouched";
        h.state_mut().run("layer.select", json!({"layer": id.0})).expect("select");
        h.run_steps(3);
        assert_eq!(h.state().perf.last_refresh, "untouched", "selecting a layer must not recomposite");
    }
    // The Properties panel's Levels histogram isn't recomputed for a selection change either.
    let levels = doc(&h).layers.iter().rev().find(|l| matches!(l.content, LayerContent::Adjustment(_))).map(|l| l.id).expect("levels layer");
    h.state_mut().run("layer.select", json!({"layer": levels.0})).expect("select");
    h.run_steps(3);
    assert!(h.state().perf.spans.contains_key("histogram"), "the Levels editor shows a histogram");
    h.state_mut().perf.spans.remove("histogram");
    h.state_mut().run("layer.select", json!({"layer": levels.0})).expect("select");
    h.run_steps(3);
    assert!(!h.state().perf.spans.contains_key("histogram"), "selecting changed no pixels");
    // Hiding a text layer refreshes its area, not the whole canvas.
    h.state_mut().run("layer.setProps", json!({"layer": k.text.0, "visible": false})).expect("hide");
    h.run_steps(2);
    assert_eq!(h.state().perf.last_refresh, "rect");
    let d = doc(&h);
    let reduced = h.state().perf.last_refresh_px;
    assert!(reduced < d.size.area() / 4, "{reduced} px refreshed");
}

#[test]
fn the_move_preview_is_what_the_move_commits() {
    let (d, k) = layout_doc::build(layout_doc::Spec::small());
    let mut s = photocraft_engine::Session::new();
    s.open_document(d, None);
    for id in [k.text, k.shape, k.group, k.smart, k.pixel, k.closed_group] {
        s.execute("layer.select", json!({"layer": id.0})).expect("select");
        let before = s.active().expect("doc").doc.clone();
        let ids = move_targets(&before, &[id]);
        let preview = moved(&before, &ids, 13, -7).expect("preview");
        s.execute("layer.translate", json!({"dx": 13, "dy": -7})).expect("move");
        let after = s.active().expect("doc");
        // The damage the canvas refreshes covers where the layer was and is.
        let dmg = after.last_damage.unwrap_or_else(|| panic!("{id:?} {}: a move has a damage rect", before.layer(id).map_or("?", |l| l.content.kind_name())));
        let b0 = photocraft_compose::composite_bounds(before.layer(id).expect("layer"), before.bounds()).expect("bounds");
        assert!(dmg.contains_rect(&b0) && dmg.contains_rect(&b0.translate(13, -7).intersect(&before.bounds())), "{dmg:?} vs {b0:?}");
        let (a, b) = (photocraft_compose::flatten(&preview), photocraft_compose::flatten(&after.doc));
        let worst = a.px.iter().zip(&b.px).flat_map(|(p, q)| (0..4).map(move |c| (p[c] * p[3] - q[c] * q[3]).abs())).fold(0.0f32, f32::max);
        assert!(worst <= 2.0 / 255.0, "{id:?}: preview differs from the commit by {}/255", worst * 255.0);
        s.execute("edit.undo", json!({})).expect("undo");
    }
}

#[test]
fn layout_document_composites_on_the_gpu_like_the_cpu() {
    use eframe::wgpu;
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let mut f = std::pin::pin!(f);
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::yield_now();
        }
    }
    let Ok(adapter) = block_on(wgpu::Instance::default().request_adapter(&wgpu::RequestAdapterOptions::default())) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if photocraft_gpu::Compositor::preferred_acc_format(&adapter) != wgpu::TextureFormat::Rgba32Float {
        eprintln!("skipping: adapter can't render Rgba32Float");
        return;
    }
    let Ok((device, queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) else { return };
    let Ok(mut comp) = photocraft_gpu::Compositor::try_new_with_format(&device, wgpu::TextureFormat::Rgba32Float) else { return };
    let (d, _) = layout_doc::build(layout_doc::Spec::small());
    let (_, texts) = layout_doc::count(&d);
    assert!(texts >= 10);
    assert!(d.walk().iter().any(|(_, _, l)| matches!(l.content, LayerContent::Smart(_))));
    let cpu = photocraft_compose::flatten(&d);
    let gpu = photocraft_gpu::render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).expect("the layout composites on the GPU");
    let mut worst = 0.0f32;
    for (c, g) in cpu.px.iter().zip(&gpu) {
        for k in 0..4 {
            let (a, b) = if k == 3 { (c[3], g[3]) } else { (c[k] * c[3], g[k] * g[3]) };
            worst = worst.max((a - b).abs());
        }
    }
    assert!(worst <= 2.0 / 255.0, "GPU differs from the CPU by {}/255", worst * 255.0);
}
