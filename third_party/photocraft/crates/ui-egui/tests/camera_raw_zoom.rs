//! Camera Raw navigation uses the real modal shell and never changes document history.
use egui::{Event, Key, Modifiers, PointerButton, Pos2, Rect, pos2, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_ui_egui::{PhotocraftApp, camera_raw_ui, control, theme::ThemeKind};
use serde_json::{Value, json};

fn fixture(ppp: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_step_dt(1.0 / 60.0).with_size(vec2(1200.0, 800.0)).with_pixels_per_point(ppp).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, ThemeKind::Pro);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width":1200,"height":900})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color":"#808080"})).unwrap();
        camera_raw_ui::open(&mut app, &cc.egui_ctx).unwrap();
        app
    });
    h.run_steps(4);
    h
}
fn inspect(h: &Harness<'_, PhotocraftApp>) -> Value {
    control::inspect(h.state(), &h.ctx)["cameraRaw"].clone()
}
fn rectangle(h: &Harness<'_, PhotocraftApp>, key: &str) -> Rect {
    let v = inspect(h);
    let a = v[key].as_array().unwrap();
    Rect::from_min_max(pos2(a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32), pos2(a[2].as_f64().unwrap() as f32, a[3].as_f64().unwrap() as f32))
}
fn menu(h: &mut Harness<'_, PhotocraftApp>, ui: Value) -> Result<Value, String> {
    let ctx = h.ctx.clone();
    camera_raw_ui::menu(h.state_mut(), &ctx, "filter.cameraRaw", &json!({"ui":ui})).unwrap()
}
fn press(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, down: bool, modifiers: Modifiers) {
    h.event(Event::ModifiersChanged(modifiers));
    h.event(Event::PointerMoved(at));
    h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: down, modifiers });
    h.run_steps(2);
}
fn drag(h: &mut Harness<'_, PhotocraftApp>, start: Pos2, end: Pos2, modifiers: Modifiers) {
    press(h, start, true, modifiers);
    for n in 1..=4 {
        h.event(Event::PointerMoved(start + (end - start) * n as f32 / 4.0));
        h.run_steps(2);
    }
    press(h, end, false, modifiers);
}
fn key(h: &mut Harness<'_, PhotocraftApp>, key: Key, modifiers: Modifiers, down: bool) {
    h.event(Event::ModifiersChanged(modifiers));
    h.event(Event::Key { key, physical_key: None, pressed: down, repeat: false, modifiers });
    h.run_steps(2);
}
fn zoom(h: &Harness<'_, PhotocraftApp>) -> f64 {
    inspect(h)["zoom"].as_f64().unwrap()
}

#[test]
fn click_toggles_fit_and_actual_pixels_at_both_display_scales_without_editing() {
    for ppp in [1.0, 2.0] {
        let mut h = fixture(ppp);
        let initial = inspect(&h);
        let history = h.state().session.active().unwrap().history.past_len();
        let revision = initial["previewRevision"].clone();
        let at = rectangle(&h, "viewportRect").center();
        press(&mut h, at, true, Modifiers::NONE);
        press(&mut h, at, false, Modifiers::NONE);
        assert_eq!(zoom(&h), 1.0);
        assert!((rectangle(&h, "previewRect").width() * ppp - 1200.0).abs() < 0.01, "100% maps one source pixel to one display pixel");
        assert_eq!(inspect(&h)["previewApproximate"], false);
        // Separate clicks so egui does not treat them as a double click.
        h.run_steps(30);
        press(&mut h, at, true, Modifiers::NONE);
        press(&mut h, at, false, Modifiers::NONE);
        assert_eq!(inspect(&h)["view"]["zoom"], Value::Null);
        assert_eq!(inspect(&h)["previewRevision"], revision);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    }
}

#[test]
fn scrub_zoom_and_alt_wheel_preserve_the_detail_under_the_initial_pointer() {
    let mut h = fixture(1.0);
    menu(&mut h, json!({"view":{"zoom":2.0}})).unwrap();
    h.run_steps(2);
    let at = rectangle(&h, "viewportRect").center() + vec2(35.0, 20.0);
    let image = rectangle(&h, "previewRect");
    let point = (at - image.min) / image.size();
    drag(&mut h, at, at + vec2(60.0, 0.0), Modifiers::NONE);
    assert!(zoom(&h) > 2.5, "rightward scrub increases zoom");
    let image = rectangle(&h, "previewRect");
    assert!(((at - image.min) / image.size() - point).length() < 0.002);
    let previous = zoom(&h);
    drag(&mut h, at, at - vec2(60.0, 0.0), Modifiers::NONE);
    assert!(zoom(&h) < previous, "leftward scrub decreases zoom");
    let old = zoom(&h);
    h.event(Event::PointerMoved(at));
    h.event(Event::ModifiersChanged(Modifiers::ALT));
    h.event(Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: vec2(0.0, 100.0), phase: egui::TouchPhase::Move, modifiers: Modifiers::ALT });
    // Release Alt while the wheel notch is still being smoothed: no residual pan.
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(10);
    assert!(zoom(&h) > old);
    let image = rectangle(&h, "previewRect");
    assert!(((at - image.min) / image.size() - point).length() < 0.002);
}

#[test]
fn space_pans_while_sampling_without_moving_probes_or_editing_the_filter() {
    let mut h = fixture(1.0);
    menu(&mut h, json!({"view":{"zoom":2.0},"scope":{"samplerTool":true,"samplers":[[0.5,0.5]]}})).unwrap();
    h.run_steps(2);
    let initial = inspect(&h);
    let at = rectangle(&h, "viewportRect").center();
    key(&mut h, Key::Space, Modifiers::NONE, true);
    drag(&mut h, at, at + vec2(80.0, 40.0), Modifiers::NONE);
    key(&mut h, Key::Space, Modifiers::NONE, false);
    assert_ne!(inspect(&h)["view"]["center"], initial["view"]["center"]);
    assert_eq!(inspect(&h)["scope"]["samplers"], initial["scope"]["samplers"]);
    assert_eq!(inspect(&h)["params"], initial["params"]);
    assert_eq!(inspect(&h)["previewRevision"], initial["previewRevision"]);
    h.event(Event::PointerMoved(at));
    h.run_steps(2);
    let image = rectangle(&h, "previewRect");
    let point = (at - image.min) / image.size();
    let sample = inspect(&h)["pointerReadout"]["position"].clone();
    assert!((sample[0].as_f64().unwrap() - point.x as f64).abs() < 0.001);
    assert!((sample[1].as_f64().unwrap() - point.y as f64).abs() < 0.001);
}

#[test]
fn command_shortcuts_and_box_zoom_use_the_same_path_for_control_and_command_keys() {
    for modifiers in [Modifiers { ctrl: true, command: true, ..Modifiers::NONE }, Modifiers { mac_cmd: true, command: true, ..Modifiers::NONE }] {
        let mut h = fixture(1.0);
        let at = rectangle(&h, "viewportRect").center();
        drag(&mut h, at - vec2(90.0, 60.0), at + vec2(90.0, 60.0), modifiers);
        assert!(zoom(&h) > 1.5, "command-drag fits the selected region");
        let old = zoom(&h);
        key(&mut h, Key::Minus, modifiers, true);
        key(&mut h, Key::Minus, modifiers, false);
        assert!(zoom(&h) < old);
        key(&mut h, Key::Num0, modifiers, true);
        key(&mut h, Key::Num0, modifiers, false);
        assert_eq!(inspect(&h)["view"]["zoom"], Value::Null);
        key(&mut h, Key::Num0, modifiers | Modifiers::ALT, true);
        key(&mut h, Key::Num0, modifiers | Modifiers::ALT, false);
        assert_eq!(zoom(&h), 1.0);
        menu(&mut h, json!({"view":{"zoom":2.0},"scope":{"samplerTool":true}})).unwrap();
        key(&mut h, Key::H, Modifiers::NONE, true);
        key(&mut h, Key::H, Modifiers::NONE, false);
        assert_eq!(inspect(&h)["scope"]["samplerTool"], false);
        assert_eq!(inspect(&h)["view"]["hand"], true);
        drag(&mut h, at, at + vec2(60.0, 0.0), modifiers);
        assert!(zoom(&h) > 2.5, "command temporarily zooms while Hand is selected");
        key(&mut h, Key::Z, Modifiers::NONE, true);
        key(&mut h, Key::Z, Modifiers::NONE, false);
        press(&mut h, at, true, modifiers | Modifiers::SHIFT);
        press(&mut h, at, false, modifiers | Modifiers::SHIFT);
        assert_eq!(zoom(&h), 1.0, "command-shift-click returns to actual pixels");
    }
}

#[test]
fn invalid_view_requests_are_atomic_and_reopening_starts_in_fit_view() {
    let mut h = fixture(1.0);
    let before = inspect(&h);
    for view in [
        json!(null),
        json!({"zoom":0}),
        json!({"zoom":100}),
        json!({"zoom":"100%"}),
        json!({"center":[-1,0]}),
        json!({"center":[0]}),
        json!({"hand":1}),
        json!({"unknown":true}),
    ] {
        assert!(menu(&mut h, json!({"view":view,"set":{"exposure":1.0},"before":true})).is_err());
        assert_eq!(inspect(&h), before);
    }
    menu(&mut h, json!({"view":{"zoom":2.0,"hand":true}})).unwrap();
    menu(&mut h, json!({"cancel":true})).unwrap();
    menu(&mut h, json!({})).unwrap();
    h.run_steps(2);
    assert_eq!(inspect(&h)["view"], json!({"zoom":null,"center":[0.5,0.5],"hand":false}));
    h.get_by_label(photocraft_ui_egui::i18n::tr(photocraft_ui_egui::i18n::current(), "100%")).click();
    h.run_steps(2);
    assert_eq!(zoom(&h), 1.0);
}

#[test]
fn asynchronous_refinement_completes_for_the_latest_edit_and_before_keeps_navigation() {
    let mut h = fixture(1.0);
    menu(&mut h, json!({"view":{"zoom":1.0},"set":{"exposure":0.5},"scope":{"samplers":[[0.5,0.5]]}})).unwrap();
    h.run_steps(2);
    // Changing parameters while refinement may still be running must never display an old edit.
    menu(&mut h, json!({"set":{"exposure":1.0}})).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while inspect(&h)["previewApproximate"] != false && std::time::Instant::now() < deadline {
        h.run_steps(1);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(inspect(&h)["detailError"], Value::Null);
    assert_eq!(inspect(&h)["detailPending"], false);
    assert_eq!(inspect(&h)["previewApproximate"], false);
    let developed = inspect(&h)["samplerReadouts"][0]["values"][0].as_f64().unwrap();
    let view = inspect(&h)["view"].clone();
    menu(&mut h, json!({"before":true})).unwrap();
    h.run_steps(2);
    assert_eq!(inspect(&h)["view"], view);
    assert_eq!(inspect(&h)["previewApproximate"], false);
    let original = inspect(&h)["samplerReadouts"][0]["values"][0].as_f64().unwrap();
    assert!(developed > original + 20.0);
    menu(&mut h, json!({"before":false})).unwrap();
    h.run_steps(2);
    assert_eq!(inspect(&h)["samplerReadouts"][0]["values"][0].as_f64().unwrap(), developed);
}
