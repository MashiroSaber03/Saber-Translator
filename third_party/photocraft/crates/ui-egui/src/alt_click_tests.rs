//! ⌥-click through the real canvas: the Brush's temporary Eyedropper (#1659) and the Clone
//! Stamp's source point and painting (#1658), the Zoom tool's Zoom Out (#3137).

use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [100.0, 50.0];
    v.fit_pending = false;
    h.run_steps(2);
    h
}

fn screen(h: &Harness<'static, PhotocraftApp>, x: f32, y: f32) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    let xf = ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    };
    xf.to_screen(x, y)
}

fn mods(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers) {
    h.event(egui::Event::ModifiersChanged(m));
    h.run_steps(1);
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool, m: Modifiers) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: m });
    h.run_steps(1);
}

/// A click at document point (x, y) with `m` held from before the press until after the release.
fn click(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    let p = screen(h, x, y);
    mods(h, m);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, m);
    button(h, p, false, m);
    h.run_steps(1);
    mods(h, Modifiers::NONE);
}

/// A drag between document points with `m` held throughout.
fn drag(h: &mut Harness<'static, PhotocraftApp>, from: [f32; 2], to: [f32; 2], m: Modifiers) {
    let p = screen(h, from[0], from[1]);
    mods(h, m);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, m);
    for i in 1..=8 {
        let t = i as f32 / 8.0;
        let q = screen(h, from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t);
        h.event(egui::Event::PointerMoved(q));
        h.run_steps(1);
    }
    button(h, screen(h, to[0], to[1]), false, m);
    h.run_steps(2);
    mods(h, Modifiers::NONE);
}

/// Two halves: red on the left, green on the right, on one background layer.
fn two_colours() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 100, "background": "white"})).unwrap();
    app.run("select.rect", json!({"x": 0, "y": 0, "width": 100, "height": 100})).unwrap();
    app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
    app.run("select.rect", json!({"x": 100, "y": 0, "width": 100, "height": 100})).unwrap();
    app.run("edit.fill", json!({"color": "#00ff00"})).unwrap();
    app.run("select.deselect", json!({})).unwrap();
    app.sync_views();
    app
}

#[test]
fn alt_click_with_the_brush_samples_the_foreground() {
    let mut h = harness(two_colours());
    h.state_mut().ui.tool = Tool::Brush;
    h.state_mut().run("tools.setColors", json!({"foreground": [0.0, 0.0, 1.0, 1.0]})).unwrap();
    let rev = h.state().session.active().unwrap().revision;
    click(&mut h, 50.0, 50.0, Modifiers::ALT);
    assert_eq!(h.state().session.tools.foreground, [1.0, 0.0, 0.0, 1.0], "⌥-click sampled red");
    assert_eq!(h.state().session.active().unwrap().revision, rev, "nothing was painted");
    click(&mut h, 150.0, 50.0, Modifiers::ALT);
    assert_eq!(h.state().session.tools.foreground, [0.0, 1.0, 0.0, 1.0], "⌥-click sampled green");
}

#[test]
fn alt_click_sets_the_clone_source_and_a_drag_paints_it() {
    let mut h = harness(two_colours());
    h.state_mut().ui.tool = Tool::CloneStamp;
    h.state_mut().ui.tool_options.clone_sample = "all".into();
    h.state_mut().run("tools.setBrush", json!({"brush": {"size": 10, "hardness": 1.0}})).unwrap();
    click(&mut h, 150.0, 50.0, Modifiers::ALT);
    assert!(h.state().session.presets.clone.active().source.is_some(), "⌥-click set the source");
    let rev = h.state().session.active().unwrap().revision;
    drag(&mut h, [30.0, 50.0], [70.0, 50.0], Modifiers::NONE);
    let st = h.state().session.active().unwrap();
    assert!(st.revision > rev, "the clone stroke was committed");
    let px = st.doc.layers[0].surface().unwrap().rgba(50, 50);
    assert!(px[1] > 0.9 && px[0] < 0.1, "green cloned over red: {px:?}");
}

#[test]
fn clone_stamp_paints_an_empty_layer_from_the_layers_below() {
    // The reported setup: a new empty layer on top, Sample: All Layers (or Current & Below).
    for sample in ["all", "currentAndBelow"] {
        for aligned in [true, false] {
            let mut app = two_colours();
            app.run("layer.new.layer", json!({})).unwrap();
            app.sync_views();
            let mut h = harness(app);
            h.state_mut().ui.tool = Tool::CloneStamp;
            h.state_mut().ui.tool_options.clone_sample = sample.into();
            h.state_mut().ui.tool_options.clone_aligned = aligned;
            h.state_mut().run("tools.setBrush", json!({"brush": {"size": 10, "hardness": 1.0}})).unwrap();
            click(&mut h, 150.0, 50.0, Modifiers::ALT);
            drag(&mut h, [30.0, 50.0], [70.0, 50.0], Modifiers::NONE);
            drag(&mut h, [30.0, 20.0], [70.0, 20.0], Modifiers::NONE);
            let st = h.state().session.active().unwrap();
            let top = st.doc.layers.last().unwrap().surface().unwrap();
            for y in [50, 20] {
                let px = top.rgba(50, y);
                assert!(px[1] > 0.9 && px[0] < 0.1 && px[3] > 0.9, "{sample} aligned={aligned} y={y}: {px:?}");
            }
        }
    }
}

/// What X11 and Wayland deliver when the window manager grabs an Alt+left press (Cinnamon's
/// window move): the pointer leaves and the keyboard focus goes (winit clears the modifiers),
/// and both come back on release. The click itself never arrives.
fn grabbed_alt_click(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, refocus: bool) {
    let p = screen(h, x, y);
    mods(h, Modifiers::ALT);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(2);
    h.event(egui::Event::PointerGone);
    h.event(egui::Event::WindowFocused(false));
    h.input_mut().focused = false;
    mods(h, Modifiers::NONE);
    h.run_steps(1);
    if refocus {
        h.event(egui::Event::WindowFocused(true));
        h.input_mut().focused = true;
        mods(h, Modifiers::ALT);
    }
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(2);
    mods(h, Modifiers::NONE);
    h.input_mut().focused = true;
}

#[test]
fn an_alt_click_the_window_manager_took_says_so() {
    for tool in [Tool::Brush, Tool::Pencil, Tool::CloneStamp, Tool::Healing, Tool::Zoom] {
        let mut h = harness(two_colours());
        h.state_mut().ui.tool = tool;
        grabbed_alt_click(&mut h, 50.0, 50.0, true);
        assert_eq!(h.state().ui.status, crate::alt_grab::message(), "{tool:?}");
        assert!(h.state().ui.status_error);
    }
}

#[test]
fn leaving_the_window_without_a_grab_says_nothing() {
    // Alt+Tab to another window: the pointer comes back to an unfocused window.
    let mut h = harness(two_colours());
    h.state_mut().ui.tool = Tool::Brush;
    grabbed_alt_click(&mut h, 50.0, 50.0, false);
    assert_ne!(h.state().ui.status, crate::alt_grab::message());
    // No ⌥, or a tool whose ⌥-click does nothing special.
    for (tool, m) in [(Tool::Brush, Modifiers::NONE), (Tool::Eraser, Modifiers::ALT)] {
        let mut h = harness(two_colours());
        h.state_mut().ui.tool = tool;
        let p = screen(&h, 50.0, 50.0);
        mods(&mut h, m);
        h.event(egui::Event::PointerMoved(p));
        h.run_steps(2);
        h.event(egui::Event::PointerGone);
        h.run_steps(1);
        h.event(egui::Event::PointerMoved(p));
        h.run_steps(2);
        assert_ne!(h.state().ui.status, crate::alt_grab::message(), "{tool:?} {m:?}");
    }
    // A real ⌥-click arrives: it samples, and no message follows.
    let mut h = harness(two_colours());
    h.state_mut().ui.tool = Tool::Brush;
    click(&mut h, 50.0, 50.0, Modifiers::ALT);
    assert_eq!(h.state().session.tools.foreground, [1.0, 0.0, 0.0, 1.0]);
    assert_ne!(h.state().ui.status, crate::alt_grab::message());
}

#[test]
fn a_latched_alt_samples_and_sets_the_clone_source_without_the_key() {
    // The way around a desktop that takes Alt+click: latch ⌥ in Window › Modifier Keys.
    let mut h = harness(two_colours());
    h.state_mut().ui.shell.sticky_alt = true;
    h.state_mut().ui.tool = Tool::Brush;
    click(&mut h, 150.0, 50.0, Modifiers::NONE);
    assert_eq!(h.state().session.tools.foreground, [0.0, 1.0, 0.0, 1.0]);
    h.state_mut().ui.tool = Tool::CloneStamp;
    click(&mut h, 150.0, 50.0, Modifiers::NONE);
    assert!(h.state().session.presets.clone.active().source.is_some());
}

#[test]
fn alt_click_with_the_zoom_tool_zooms_out_held_or_latched() {
    for (sticky, scrubby, m) in [(false, true, Modifiers::ALT), (true, true, Modifiers::NONE), (true, false, Modifiers::NONE)] {
        let mut h = harness(two_colours());
        h.state_mut().ui.tool = Tool::Zoom;
        h.state_mut().ui.tool_options.zoom_scrubby = scrubby;
        h.state_mut().ui.shell.sticky_alt = sticky;
        click(&mut h, 50.0, 50.0, m);
        assert_eq!(h.state().ui.views[0].zoom, 2.0 / 3.0, "sticky={sticky} scrubby={scrubby}");
        // Without ⌥ the click zooms in again.
        h.state_mut().ui.shell.sticky_alt = false;
        click(&mut h, 50.0, 50.0, Modifiers::NONE);
        assert_eq!(h.state().ui.views[0].zoom, 1.0, "sticky={sticky} scrubby={scrubby}");
    }
}
