//! Background full CPU redraws (#4024). The GPU tests skip when there is no adapter.

use super::*;
use serde_json::json;

#[test]
fn blocks_cover_the_document_once() {
    let bounds = Rect::new(0, 0, 2500, 1100);
    let b = blocks(bounds);
    assert_eq!(b.len(), 3 * 2);
    assert_eq!(b.iter().map(|r| r.width() as u64 * r.height() as u64).sum::<u64>(), 2500 * 1100);
    for (i, r) in b.iter().enumerate() {
        assert!(r.x0 % BLOCK == 0 && r.y0 % BLOCK == 0, "block {i} starts on the grid: {r:?}");
        assert!(b[i + 1..].iter().all(|o| o.intersect(r).is_empty()), "block {i} overlaps another");
    }
}

#[test]
fn visible_blocks_come_first_nearest_the_view_centre() {
    let b = blocks(Rect::new(0, 0, 4096, 4096));
    // The view shows the right half of the second block row.
    let visible = Rect::new(2100, 1100, 4000, 1900);
    let o = order(&b, 0..b.len(), visible);
    assert_eq!(o.len(), 16);
    let shown: Vec<usize> = (0..b.len()).filter(|&i| !b[i].intersect(&visible).is_empty()).collect();
    assert_eq!(shown.len(), 2);
    assert!(o[..2].iter().all(|i| shown.contains(i)), "{o:?}");
    // Then the block row's neighbours, the far corners last.
    assert!(o[14..].contains(&12), "the bottom-left corner is among the last: {o:?}");
    // Only the blocks asked for.
    assert_eq!(order(&b, [3, 9].into_iter(), visible).len(), 2);
}

struct Gpu {
    harness: egui_kittest::Harness<'static, PhotocraftApp>,
    observer: GpuCanvas,
}

/// The app on a real device with background jobs, or `None` without a GPU adapter.
fn gpu_app() -> Option<Gpu> {
    let observer = Arc::new(std::sync::Mutex::new(None));
    let observer_out = observer.clone();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            if let Some(rs) = cc.wgpu_render_state.as_ref() {
                *observer_out.lock().expect("observer") = Some(GpuCanvas::new(rs));
                app.set_wgpu(rs.clone());
            }
            app.background_jobs = true;
            app
        })
    }));
    let Ok(harness) = built else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    let observer = observer.lock().expect("observer").take()?;
    Some(Gpu { harness, observer })
}

/// A document the wgpu compositor leaves to the CPU (knockout on its top layer, which also has
/// Blend If).
fn cpu_only_document(app: &mut PhotocraftApp, w: u32, h: u32) -> DocId {
    app.run("file.new", json!({"width": w, "height": h, "background": "#336699"})).expect("new");
    app.run("layer.new.layer", json!({})).expect("layer");
    app.run("edit.fill", json!({"color": "#cc8844"})).expect("fill");
    app.run("layer.layerStyle.blendingOptions", json!({"opacity": 60, "knockout": "shallow", "blendIf": {"channel": "gray", "underlying": [0, 10, 245, 255]}}))
        .expect("blending options");
    app.session.active().expect("document").doc.id
}

/// Frames until no redraw is running.
fn settle(g: &mut Gpu) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        g.harness.step();
        if g.harness.state().redraws.is_empty() {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "the redraw didn't finish");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// The canvas texture of `doc` against the synchronous CPU refresh of the current document.
fn assert_matches_synchronous(g: &mut Gpu, doc: DocId) {
    let app = g.harness.state_mut();
    let d = app.session.active().expect("document").doc.clone();
    let (display, _) = crate::canvas::canvas_display(app, &d, None);
    let reference = doc.0 ^ (1u64 << 60);
    g.observer.upload_composite(reference, &d, display.as_deref());
    let (_, size, shown) = g.observer.read_texels(doc.0).expect("canvas texture");
    let (_, _, want) = g.observer.read_texels(reference).expect("reference texture");
    assert_eq!(size, [d.size.width, d.size.height]);
    let differ = shown.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "{differ} texels differ from the synchronous refresh");
}

#[test]
fn a_full_cpu_refresh_draws_in_the_background_with_the_synchronous_pixels() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 2600, 1700);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc), "the frame started a redraw instead of compositing inline");
    settle(&mut g);
    let app = g.harness.state();
    assert_eq!(app.perf.last_refresh, "background");
    assert!(app.perf.gpu_fallback.as_deref().is_some_and(|f| f.contains("Advanced Blending")), "{:?}", app.perf.gpu_fallback);
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn an_edit_during_a_redraw_is_not_overwritten_by_blocks_drawn_before_it() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    // 4×3 blocks: the worker can't hand them all over before the next frame.
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc));
    // A corner, drawn last (farthest from the view's centre), changes on a layer of its own while
    // the redraw runs: a small edit, drawn at once.
    g.harness.state_mut().run("layer.new.layer", json!({})).expect("layer");
    g.harness.step();
    g.harness.state_mut().run("select.rect", json!({"x": 3900, "y": 2900, "width": 150, "height": 150})).expect("select");
    g.harness.step();
    g.harness.state_mut().run("edit.fill", json!({"color": "#22ee22"})).expect("fill");
    g.harness.step();
    assert_eq!(g.harness.state().perf.last_refresh, "rect", "the fill's damage was drawn synchronously");
    assert!(g.harness.state().redraws.contains_key(&doc), "the redraw goes on");
    settle(&mut g);
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn an_edit_damaging_most_of_a_large_document_is_drawn_in_the_background_too() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 2600, 1700);
    settle(&mut g);
    // Filling part of the Blend If layer can change the composite anywhere: the whole canvas.
    g.harness.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 300, "height": 300})).expect("select");
    g.harness.step();
    g.harness.state_mut().run("edit.fill", json!({"color": "#22ee22"})).expect("fill");
    assert_eq!(g.harness.state().session.active().expect("document").last_damage, Some(Rect::new(0, 0, 2600, 1700)));
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc), "not composited inline: {}", g.harness.state().perf.last_refresh);
    settle(&mut g);
    assert_eq!(g.harness.state().perf.last_refresh, "background");
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn closing_the_document_stops_its_redraw() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc));
    g.harness.state_mut().run("file.close", json!({"save": false})).expect("close");
    g.harness.step();
    assert!(g.harness.state().redraws.is_empty());
}
