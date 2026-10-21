//! Input routing through the real canvas, dialogs and shortcuts (#44, #45): keyboard zoom hits the
//! canvas (never egui's UI scale), and an open dialog keeps the canvas pannable and zoomable but
//! is not cancelled by a click outside it.

use egui::{Key, Modifiers, Pos2, pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.sync_views();
    let mut h = Harness::builder().with_size(vec2(1200.0, 800.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            // Fonts set up after the first frame only apply from the next one.
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            crate::dialogs::show(app, &ctx);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

fn zoom(h: &Harness<'static, PhotocraftApp>) -> f32 {
    h.state().current_zoom()
}

fn open_dialog(h: &mut Harness<'static, PhotocraftApp>) {
    crate::dialogs::open_command_dialog(h.state_mut(), "image.adjustments.brightnessContrast", "Brightness/Contrast…");
    h.run_steps(3);
    assert_eq!(h.state().ui.dialogs.len(), 1);
}

/// A point on the canvas well away from the centred dialog.
fn free_canvas(h: &Harness<'static, PhotocraftApp>) -> Pos2 {
    let r = h.state().last_canvas_rect;
    pos2(r.left() + 60.0, r.bottom() - 60.0)
}

#[test]
fn ctrl_plus_zooms_the_canvas_not_the_interface() {
    let mut h = harness();
    let z0 = zoom(&h);
    // ⌘+ / Ctrl+'+' as typed with ⇧ on a US layout, and as the numpad / Nordic `+`.
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, Key::Plus);
    h.run_steps(2);
    let z1 = zoom(&h);
    assert!(z1 > z0, "⌘+ should zoom the canvas in ({z0} -> {z1})");
    h.key_press_modifiers(Modifiers::COMMAND, Key::Plus);
    h.run_steps(2);
    assert!(zoom(&h) > z1);
    h.key_press_modifiers(Modifiers::COMMAND, Key::Minus);
    h.run_steps(2);
    assert!((zoom(&h) - z1).abs() < 1e-4);
    // ⌘1 is 100%, ⌘0 fits on screen again.
    h.key_press_modifiers(Modifiers::COMMAND, Key::Num1);
    h.run_steps(2);
    assert_eq!(zoom(&h), 1.0);
    h.key_press_modifiers(Modifiers::COMMAND, Key::Num0);
    h.run_steps(3);
    assert!((zoom(&h) - z0).abs() < 1e-3, "⌘0 fits on screen: {} vs {z0}", zoom(&h));
    assert_eq!(h.ctx.zoom_factor(), 1.0, "the interface must never scale");
}

#[test]
fn keyboard_zoom_works_with_a_dialog_open() {
    let mut h = harness();
    open_dialog(&mut h);
    let z0 = zoom(&h);
    h.key_press_modifiers(Modifiers::COMMAND, Key::Equals);
    h.run_steps(2);
    assert!(zoom(&h) > z0);
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, Key::Plus);
    h.run_steps(2);
    assert_eq!(h.ctx.zoom_factor(), 1.0);
    assert_eq!(h.state().ui.dialogs.len(), 1);
    // Other commands stay blocked while the dialog is open (⌘J would duplicate the layer).
    let layers = h.state().session.active().unwrap().doc.layers.len();
    h.key_press_modifiers(Modifiers::COMMAND, Key::J);
    h.run_steps(2);
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), layers);
}

/// Select › Color Range: the eyedropper picks on the image itself (Photoshop), not only in the
/// dialog's preview. Shift adds a sample, and the dialog stays open.
#[test]
fn color_range_eyedropper_picks_on_the_canvas() {
    let mut h = harness();
    crate::color_range_ui::open(h.state_mut());
    h.run_steps(4);
    let points = |h: &Harness<'static, PhotocraftApp>| {
        let f = &h.state().ui.dialogs.last().unwrap().fields;
        (f["points"].as_array().unwrap().len(), f["points"].clone())
    };
    let click = |h: &mut Harness<'static, PhotocraftApp>, p: Pos2, modifiers: Modifiers| {
        h.hover_at(p);
        h.run_steps(1);
        h.event_modifiers(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers }, modifiers);
        h.run_steps(1);
        h.event_modifiers(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: false, modifiers }, modifiers);
        h.run_steps(2);
    };
    // Zoomed in, the document reaches out from under the centred dialog.
    {
        let v = &mut h.state_mut().ui.views[0];
        v.zoom = 3.0;
        v.center = [200.0, 150.0];
        v.fit_pending = false;
    }
    h.run_steps(2);
    let v = h.state().ui.views[0].clone();
    let c = h.state().last_canvas_rect.center();
    let screen = |d: [f32; 2]| pos2(c.x + (d[0] - v.center[0]) * v.zoom, c.y + (d[1] - v.center[1]) * v.zoom);
    let p = screen([30.5, 40.5]);
    assert!(h.state().last_canvas_rect.contains(p));
    click(&mut h, p, Modifiers::NONE);
    let (n, pts) = points(&h);
    assert_eq!(n, 1, "{pts}");
    let at = doc_at(&h, p);
    assert_eq!(pts[0][0].as_f64().unwrap().floor(), f64::from(at[0]).floor(), "{pts} vs {at:?}");
    assert_eq!(pts[0][1].as_f64().unwrap().floor(), f64::from(at[1]).floor(), "{pts} vs {at:?}");
    // ⇧-click adds another sample; the dialog never closes.
    click(&mut h, screen([60.5, 40.5]), Modifiers::SHIFT);
    assert_eq!(points(&h).0, 2);
    click(&mut h, screen([60.5, 40.5]), Modifiers::ALT);
    assert_eq!(h.state().ui.dialogs.last().unwrap().fields["subtractPoints"].as_array().unwrap().len(), 1);
    // A stationary held button must not subtract again on every redraw.
    let p = screen([60.5, 40.5]);
    h.event_modifiers(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers: Modifiers::ALT }, Modifiers::ALT);
    h.run_steps(4);
    assert_eq!(h.state().ui.dialogs.last().unwrap().fields["subtractPoints"].as_array().unwrap().len(), 2);
    h.event_modifiers(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: false, modifiers: Modifiers::ALT }, Modifiers::ALT);
    h.run_steps(2);
    // The modal eyedropper overrides a previously selected Hand tool.
    h.state_mut().ui.tool = crate::state::Tool::Hand;
    let center = h.state().ui.views[0].center;
    click(&mut h, screen([30.5, 40.5]), Modifiers::NONE);
    assert_eq!(points(&h).0, 1);
    assert_eq!(h.state().ui.views[0].center, center);
    assert_eq!(h.state().ui.dialogs.len(), 1);
    // The document is untouched until OK.
    assert!(h.state().session.active().unwrap().doc.selection.is_none());
}

#[test]
fn clicking_outside_a_dialog_keeps_it_open_and_the_canvas_pans_and_zooms() {
    let mut h = harness();
    open_dialog(&mut h);
    let p = free_canvas(&h);
    h.hover_at(p);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    assert_eq!(h.state().ui.dialogs.len(), 1, "a click outside must not cancel the dialog");

    // Scrolling over the canvas pans it.
    let c0 = h.state().ui.views[0].center;
    h.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: vec2(0.0, -40.0), phase: egui::TouchPhase::Move, modifiers: Modifiers::NONE });
    h.run_steps(8);
    let c1 = h.state().ui.views[0].center;
    assert!(c1[1] > c0[1], "scroll pans under a dialog: {c0:?} -> {c1:?}");

    // Space-drag pans too.
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    for i in 1..=5 {
        h.hover_at(p + vec2(10.0 * i as f32, 0.0));
        h.run_steps(1);
    }
    h.event(egui::Event::PointerButton { pos: p + vec2(50.0, 0.0), button: egui::PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    let c2 = h.state().ui.views[0].center;
    let z = zoom(&h);
    assert!((c2[0] - (c1[0] - 50.0 / z)).abs() < 0.5, "space-drag pans by 50 pt: {c1:?} -> {c2:?}");
    assert_eq!(h.state().ui.dialogs.len(), 1);

    // Esc still cancels.
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(h.state().ui.dialogs.is_empty());
}

fn press(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, pressed: bool) {
    h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    h.run_steps(1);
}

fn picker_color(h: &Harness<'static, PhotocraftApp>) -> String {
    let d = h.state().ui.dialogs.last().unwrap();
    d.fields.get("color").and_then(serde_json::Value::as_str).unwrap().to_string()
}

/// The Color Picker's eyedropper, as in Photoshop: over the image the pointer is a pipette, and a
/// click or drag there samples into the picker's new colour, whatever the tool. OK applies it.
#[test]
fn color_picker_samples_the_image_under_its_pipette() {
    let mut h = harness();
    h.state_mut().run("shape.create", json!({"kind": "rect", "rect": [0, 0, 200, 300], "fill": "#ff0000"})).unwrap();
    h.state_mut().run("shape.create", json!({"kind": "rect", "rect": [200, 0, 200, 300], "fill": "#00ff00"})).unwrap();
    // 400 %: the image covers the whole canvas, red on the left of the dialog, green on its right.
    let v = &mut h.state_mut().ui.views[0];
    (v.zoom, v.center, v.fit_pending) = (4.0, [200.0, 150.0], false);
    crate::color_picker_ui::open(h.state_mut(), "foreground");
    h.run_steps(3);
    let r = h.state().last_canvas_rect;
    let (red, green) = (pos2(r.left() + 60.0, r.center().y), pos2(r.right() - 60.0, r.center().y));
    assert_eq!(doc_at(&h, red)[0] < 200.0, doc_at(&h, green)[0] > 200.0);
    let foreground = h.state().session.tools.foreground;

    h.hover_at(red);
    h.run_steps(1);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "the pipette replaces the pointer");
    h.hover_at(r.center());
    h.run_steps(1);
    assert_ne!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "over the dialog it is the normal pointer");

    // A click samples; the press and release may land in one frame (`ui.click`).
    h.hover_at(red);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: red, button: egui::PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    press(&mut h, red, false);
    assert_eq!(picker_color(&h), "#ff0000");
    assert_eq!(h.state().session.tools.foreground, foreground, "only OK sets the foreground");

    // A drag keeps sampling, skipping the dialog on the way.
    press(&mut h, red, true);
    for i in 1..=6 {
        h.hover_at(red + (green - red) * (i as f32 / 6.0));
        h.run_steps(1);
    }
    press(&mut h, green, false);
    assert_eq!(picker_color(&h), "#00ff00");

    // Space-drag still pans instead of sampling.
    h.hover_at(red);
    h.run_steps(1);
    let c0 = h.state().ui.views[0].center;
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
    press(&mut h, red, true);
    h.hover_at(red + vec2(40.0, 0.0));
    h.run_steps(1);
    press(&mut h, red + vec2(40.0, 0.0), false);
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    let c1 = h.state().ui.views[0].center;
    assert!((c1[0] - (c0[0] - 10.0)).abs() < 0.5, "space-drag pans by 40 pt at 400 %: {c0:?} -> {c1:?}");
    assert_eq!(picker_color(&h), "#00ff00");

    // With the Hand tool a click samples too (the tool doesn't matter); OK applies the sample.
    h.state_mut().ui.tool = crate::state::Tool::Hand;
    h.hover_at(red);
    h.run_steps(1);
    press(&mut h, red, true);
    press(&mut h, red, false);
    assert_eq!(picker_color(&h), "#ff0000");
    h.key_press(Key::Enter);
    h.run_steps(2);
    assert!(h.state().ui.dialogs.is_empty());
    assert_eq!(h.state().session.tools.foreground, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn curves_picker_samples_document_coordinates_through_the_view_transform() {
    let mut h = harness();
    h.state_mut().run("select.rect", json!({"x": 0, "y": 0, "width": 200, "height": 300})).unwrap();
    h.state_mut().run("edit.fill", json!({"color": "#804020"})).unwrap();
    h.state_mut().run("select.rect", json!({"x": 200, "y": 0, "width": 200, "height": 300})).unwrap();
    h.state_mut().run("edit.fill", json!({"color": "#20a0e0"})).unwrap();
    h.state_mut().run("select.deselect", json!({})).unwrap();
    let committed = h.state().session.active().unwrap().doc.clone();
    let history = h.state().session.active().unwrap().history.past_len();
    let view = &mut h.state_mut().ui.views[0];
    (view.zoom, view.center, view.fit_pending) = (4.0, [200.0, 150.0], false);
    h.state_mut().ui.view.flip_horizontal = true;
    let dialog = crate::adjust_dialog::open(h.state_mut(), "image.adjustments.curves").unwrap();
    h.state_mut().ui.dialog_mut(dialog).unwrap().fields.insert("__curvePicker".into(), json!("black"));
    h.run_steps(3);
    let canvas = h.state().last_canvas_rect;
    let left = pos2(canvas.left() + 60.0, canvas.center().y);
    h.hover_at(left);
    h.run_steps(1);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::None);
    press(&mut h, left, true);
    press(&mut h, left, false);
    h.run_steps(3);
    assert!(std::sync::Arc::ptr_eq(&h.state().session.active().unwrap().doc, &committed));
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    assert!(crate::adjust_preview::display_doc(h.state_mut(), 0).is_some(), "the new curve is visible only through the preview");
    let red = h
        .state()
        .ui
        .dialogs
        .iter()
        .find(|candidate| candidate.id == dialog)
        .and_then(|candidate| candidate.fields.get("red"))
        .and_then(serde_json::Value::as_array)
        .and_then(|points| points.first())
        .and_then(serde_json::Value::as_array)
        .and_then(|point| point.first())
        .and_then(serde_json::Value::as_f64)
        .unwrap();
    assert!((red - 32.0).abs() < 0.6, "sampled document-right blue patch through the flipped 400% view: {red}");
}

/// Type tool (#206): Alt+←/→ at a collapsed caret kerns the pair before it by 20/1000 em (100
/// with ⌘/Ctrl), one history step per press; ⌘/Ctrl+←/→ moves by word; Alt+Shift+→ extends
/// the selection by a word.
fn wheel(h: &Harness<'static, PhotocraftApp>, dy: f32, modifiers: Modifiers) {
    h.event_modifiers(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: vec2(0.0, dy), phase: egui::TouchPhase::Move, modifiers }, modifiers);
}

/// The document point under screen point `p` (no flip).
fn doc_at(h: &Harness<'static, PhotocraftApp>, p: Pos2) -> [f32; 2] {
    let v = &h.state().ui.views[0];
    let c = h.state().last_canvas_rect.center();
    [v.center[0] + (p.x - c.x) / v.zoom, v.center[1] + (p.y - c.y) / v.zoom]
}

/// The whole window (status bar and Navigator included) on a 400 × 300 document.
fn app_window() -> Harness<'static, PhotocraftApp> {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(8);
    h
}

/// Photoshop 25.4: ⌥ + wheel goes on past 3200 % to exactly 12800 %, and further notches there
/// change nothing. The status bar's zoom field used to stop at 3200 %, clamping the zoom back
/// every frame: past it, each notch zoomed around the pointer and the field pulled the zoom
/// back, so the image slid towards the pointer.
#[test]
fn the_wheel_reaches_12800_percent_and_stops_dead_there() {
    let mut h = app_window();
    {
        let v = &mut h.state_mut().ui.views[0];
        (v.zoom, v.center, v.fit_pending) = (40.0, [200.0, 150.0], false);
    }
    h.run_steps(4);
    assert_eq!(zoom(&h), 40.0, "4000 % holds: no zoom control clamps it back");
    let r = h.state().last_canvas_rect;
    let p = pos2(r.center().x - 150.0, r.center().y - 90.0);
    // The document point under `p`, through the canvas inside the rulers.
    let doc_at = |h: &Harness<'static, PhotocraftApp>, p: Pos2| {
        let (app, v) = (h.state(), &h.state().ui.views[0]);
        let c = crate::rulers::content_rect(app, app.last_canvas_rect).center();
        [v.center[0] + (p.x - c.x) / v.zoom, v.center[1] + (p.y - c.y) / v.zoom]
    };
    h.hover_at(p);
    h.run_steps(2);
    let d0 = doc_at(&h, p);
    for _ in 0..14 {
        wheel(&h, 1.0, Modifiers::ALT);
        h.run_steps(30);
    }
    assert_eq!(zoom(&h), crate::zoom_levels::MAX, "a notch past the limit lands on it exactly");
    let d1 = doc_at(&h, p);
    assert!((d1[0] - d0[0]).abs() < 0.02 && (d1[1] - d0[1]).abs() < 0.02, "zoomed around the pointer: {d0:?} -> {d1:?}");
    let c = h.state().ui.views[0].center;
    for _ in 0..4 {
        wheel(&h, 1.0, Modifiers::ALT);
        h.run_steps(30);
    }
    assert_eq!(zoom(&h), crate::zoom_levels::MAX);
    assert_eq!(h.state().ui.views[0].center, c, "at the limit the image doesn't move");
    // One notch back: 12800 / 1.1, still around the pointer.
    wheel(&h, -1.0, Modifiers::ALT);
    h.run_steps(30);
    assert!((zoom(&h) - crate::zoom_levels::MAX / 1.1).abs() < 1e-2, "{}", zoom(&h));
    let d2 = doc_at(&h, p);
    assert!((d2[0] - d0[0]).abs() < 0.02 && (d2[1] - d0[1]).abs() < 0.02, "{d0:?} -> {d2:?}");
}

#[test]
fn alt_scroll_zooms_in_steps_around_the_pointer() {
    let mut h = harness();
    let r = h.state().last_canvas_rect;
    let p = pos2(r.center().x + 120.0, r.center().y - 70.0);
    h.hover_at(p);
    h.run_steps(2);
    let (z0, d0) = (zoom(&h), doc_at(&h, p));
    // One notch with ⌥ held: ×1.1 (Photoshop), the point under the pointer stays put. The modifiers are
    // released right after the event, while egui still smooths the notch over later frames, which
    // must neither ease the zoom nor turn into a pan (#1490).
    wheel(&h, 1.0, Modifiers::ALT);
    h.run_steps(2);
    let at_once = zoom(&h);
    h.run_steps(40);
    let (z1, d1) = (zoom(&h), doc_at(&h, p));
    assert!((z1 / z0 - 1.1).abs() < 1e-3, "one ⌥ notch is ×1.1: {z0} -> {z1}");
    assert_eq!(at_once, z1, "the notch is applied the frame it arrives, with no easing");
    assert!((d1[0] - d0[0]).abs() < 0.05 && (d1[1] - d0[1]).abs() < 0.05, "centred on the pointer: {d0:?} -> {d1:?}");
    // Three notches back out.
    wheel(&h, -3.0, Modifiers::ALT);
    h.run_steps(40);
    assert!((zoom(&h) / z1 - 1.1f32.powi(-3)).abs() < 1e-3, "{z1} -> {}", zoom(&h));

    // A plain notch still pans, and does not zoom.
    let (z2, c2) = (zoom(&h), h.state().ui.views[0].center);
    wheel(&h, -1.0, Modifiers::NONE);
    h.run_steps(40);
    assert_eq!(zoom(&h), z2, "a plain scroll never zooms");
    assert!(h.state().ui.views[0].center[1] > c2[1], "a plain scroll pans");

    // ⌘/Ctrl + scroll pans sideways (#635), on Windows (Ctrl) and macOS (⌘) alike.
    for m in [Modifiers { ctrl: true, command: true, ..Modifiers::NONE }, Modifiers { mac_cmd: true, command: true, ..Modifiers::NONE }] {
        let c = h.state().ui.views[0].center;
        wheel(&h, -1.0, m);
        h.run_steps(40);
        let c1 = h.state().ui.views[0].center;
        assert_eq!(zoom(&h), z2, "⌘/Ctrl-scroll never zooms");
        assert!(c1[0] > c[0] && c1[1] == c[1], "⌘/Ctrl-scroll pans sideways: {c:?} -> {c1:?}");
    }
    // A pinch zooms around the pointer.
    h.event(egui::Event::Zoom(1.25));
    h.run_steps(2);
    assert!((zoom(&h) / z2 - 1.25).abs() < 1e-3, "pinch zooms: {z2} -> {}", zoom(&h));
}

#[test]
fn alt_scroll_zooms_while_a_temporary_tool_is_held() {
    let mut h = harness();
    let p = h.state().last_canvas_rect.center();
    h.hover_at(p);
    // Space (the temporary Hand, #249) is down: ⌥ + scroll still zooms by notches, and the
    // held key never changes the current tool.
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    let z0 = zoom(&h);
    wheel(&h, 2.0, Modifiers::ALT);
    h.run_steps(40);
    assert!((zoom(&h) / z0 - 1.1f32.powi(2)).abs() < 1e-3, "{z0} -> {}", zoom(&h));
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, crate::state::Tool::Brush);
}

#[test]
fn type_tool_alt_arrows_kern_the_pair() {
    use photocraft_doc::text::Kerning;
    let mut h = harness();
    let id = h.state_mut().run("type.create", json!({"x": 20, "y": 80, "text": "AVA To", "size": 40, "font": "Inter"})).unwrap()["layer"].as_u64().unwrap();
    let text = |h: &Harness<'static, PhotocraftApp>| {
        let st = h.state().session.active().unwrap();
        match &st.doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
            photocraft_doc::LayerContent::Text(t) => t.clone(),
            _ => panic!("not text"),
        }
    };
    let kern = |h: &Harness<'static, PhotocraftApp>| {
        let t = text(h);
        let r = t.char_runs();
        (r[0].style.kerning, r[0].style.kern, r.len())
    };
    let steps = |h: &Harness<'static, PhotocraftApp>| h.state().session.active().unwrap().history.entries().len();
    let metric = photocraft_text::shared().lock().unwrap().pair_kerning(&text(&h), 72.0, 0).unwrap().round();
    h.state_mut().ui.text_edit = Some(crate::state::TextEdit {
        layer: id,
        caret: 1,
        anchor: 1,
        session: "kern-test".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    h.run_steps(2);
    let s0 = steps(&h);
    h.key_press_modifiers(Modifiers::ALT, Key::ArrowRight);
    h.run_steps(2);
    assert_eq!(kern(&h).0, Kerning::Off);
    assert_eq!(kern(&h).1, metric + 20.0);
    h.key_press_modifiers(Modifiers::ALT | Modifiers::COMMAND, Key::ArrowRight);
    h.run_steps(2);
    assert_eq!(kern(&h).1, metric + 120.0);
    h.key_press_modifiers(Modifiers::ALT, Key::ArrowLeft);
    h.run_steps(2);
    assert_eq!(kern(&h).1, metric + 100.0);
    assert_eq!(steps(&h), s0 + 3, "one history step per press");
    // The caret didn't move; ⌘/Ctrl+→ moves by word, Alt+Shift+→ selects by word.
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| (e.caret, e.anchor)), Some((1, 1)));
    h.key_press_modifiers(Modifiers::COMMAND, Key::ArrowRight);
    h.run_steps(2);
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| (e.caret, e.anchor)), Some((3, 3)));
    h.key_press_modifiers(Modifiers::ALT | Modifiers::SHIFT, Key::ArrowRight);
    h.run_steps(2);
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| (e.anchor, e.caret)), Some((3, 6)));
    // With a selection Alt+→ tracks it instead (see `type_tool_format_keys_*`); at the text end
    // there's no pair to kern.
    let before = kern(&h);
    h.key_press(Key::ArrowRight);
    h.run_steps(2);
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| (e.anchor, e.caret)), Some((6, 6)));
    h.key_press_modifiers(Modifiers::ALT, Key::ArrowRight);
    h.run_steps(2);
    assert_eq!(kern(&h), before);
    assert_eq!(steps(&h), s0 + 3);
    // Undo walks back one press at a time.
    assert!(h.state_mut().session.undo());
    h.run_steps(1);
    assert_eq!(kern(&h).1, metric + 120.0);
}

/// Starts a type editing session on `layer` with `anchor..caret` selected.
fn edit_type(h: &mut Harness<'static, PhotocraftApp>, layer: u64, anchor: usize, caret: usize) {
    h.state_mut().ui.text_edit = Some(crate::state::TextEdit {
        layer,
        caret,
        anchor,
        session: "format-keys".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    h.run_steps(2);
}

fn type_layer(h: &Harness<'static, PhotocraftApp>, id: u64) -> photocraft_doc::TextLayer {
    match &h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
        photocraft_doc::LayerContent::Text(t) => t.clone(),
        _ => panic!("not text"),
    }
}

/// The character style of character `c` of a type layer.
fn char_style(h: &Harness<'static, PhotocraftApp>, id: u64, c: usize) -> photocraft_doc::text::CharStyle {
    let t = type_layer(h, id);
    let byte = t.text.char_indices().nth(c).map_or(t.text.len(), |(i, _)| i);
    let mut end = 0;
    t.char_runs()
        .into_iter()
        .find(|r| {
            end += r.len;
            end > byte || t.text.is_empty()
        })
        .unwrap()
        .style
}

/// The paragraph style of the paragraph holding character `c`.
fn para_style(h: &Harness<'static, PhotocraftApp>, id: u64, c: usize) -> photocraft_doc::text::ParagraphStyle {
    let t = type_layer(h, id);
    let byte = t.text.char_indices().nth(c).map_or(t.text.len(), |(i, _)| i);
    let mut end = 0;
    t.paragraph_runs()
        .into_iter()
        .find(|r| {
            end += r.len;
            end > byte
        })
        .unwrap()
        .style
}

fn history(h: &Harness<'static, PhotocraftApp>) -> Vec<String> {
    h.state().session.active().unwrap().history.entries()
}

/// Photoshop's ⇧⌘K/U/B/I/: toggle All Caps, Underline, Faux Bold, Faux Italic and Strikethrough
/// on the selected characters while typing, one step per press, and never reach the global
/// ⇧⌘K/U/B/I (Color Settings, Desaturate, Auto Color, Inverse).
#[test]
fn type_tool_format_keys_toggle_character_styles() {
    use photocraft_doc::text::{Caps, CharStyle};
    let mut h = harness();
    // A selection, so Select ▸ Inverse would be enabled if the key reached it.
    h.state_mut().run("select.all", json!({})).unwrap();
    let id = h.state_mut().run("type.create", json!({"x": 20, "y": 80, "text": "Hello world", "size": 20})).unwrap()["layer"].as_u64().unwrap();
    edit_type(&mut h, id, 0, 5);
    let cmd_shift = Modifiers::COMMAND | Modifiers::SHIFT;
    type Has = fn(&CharStyle) -> bool;
    let keys: [(Key, Has); 6] = [
        (Key::K, |s| s.caps == Caps::AllCaps),
        (Key::U, |s| s.underline),
        (Key::B, |s| s.faux_bold),
        (Key::I, |s| s.faux_italic),
        (Key::Slash, |s| s.strikethrough),
        (Key::Questionmark, |s| s.strikethrough),
    ];
    for (key, has) in keys {
        let before = history(&h).len();
        let was = has(&char_style(&h, id, 0));
        h.key_press_modifiers(cmd_shift, key);
        h.run_steps(2);
        assert_eq!(has(&char_style(&h, id, 0)), !was, "{key:?} toggles the selection");
        assert_eq!(has(&char_style(&h, id, 4)), !was, "{key:?}");
        assert!(!has(&char_style(&h, id, 6)), "{key:?} leaves the unselected characters");
        let steps = history(&h);
        assert_eq!(steps.len(), before + 1, "{key:?}: one history step");
        assert_eq!(steps.last().map(String::as_str), Some("Set Type Style"), "{key:?}");
        assert!(h.state().ui.dialogs.is_empty(), "{key:?} must not open Color Settings");
        assert!(h.state().ui.text_edit.is_some());
    }
    // The second strikethrough key turned it off again; press the rest again: all off.
    for key in [Key::K, Key::U, Key::B, Key::I] {
        h.key_press_modifiers(cmd_shift, key);
        h.run_steps(2);
    }
    let s = char_style(&h, id, 2);
    assert!(s.caps == Caps::Normal && !s.underline && !s.faux_bold && !s.faux_italic && !s.strikethrough);
    // Partly underlined: the key underlines the whole selection rather than flipping each part.
    h.state_mut().run("type.setStyle", json!({"layer": id, "range": [0, 2], "underline": true})).unwrap();
    h.key_press_modifiers(cmd_shift, Key::U);
    h.run_steps(2);
    assert!((0..5).all(|c| char_style(&h, id, c).underline));
    // Undo takes back one press.
    assert!(h.state_mut().session.undo());
    h.run_steps(1);
    assert!(char_style(&h, id, 0).underline && !char_style(&h, id, 3).underline);
    // A collapsed caret in non-empty text: nothing changes, and nothing global runs.
    edit_type(&mut h, id, 3, 3);
    let before = history(&h);
    for key in [Key::K, Key::U, Key::B, Key::I] {
        h.key_press_modifiers(cmd_shift, key);
        h.run_steps(2);
    }
    assert_eq!(history(&h), before);
    assert!(h.state().ui.dialogs.is_empty());
    // Empty text: the key sets the style the next characters typed get.
    let empty = h.state_mut().run("type.create", json!({"x": 20, "y": 160, "text": "", "size": 20})).unwrap()["layer"].as_u64().unwrap();
    edit_type(&mut h, empty, 0, 0);
    h.key_press_modifiers(cmd_shift, Key::B);
    h.run_steps(2);
    h.event(egui::Event::Text("ab".into()));
    h.run_steps(2);
    assert_eq!(type_layer(&h, empty).text, "ab");
    assert!(char_style(&h, empty, 0).faux_bold && char_style(&h, empty, 1).faux_bold);
}

/// Outside text editing ⇧⌘K and ⇧⌘I keep their global meanings.
#[test]
fn global_cmd_shift_keys_still_work_outside_type_editing() {
    let mut h = harness();
    h.state_mut().run("select.all", json!({})).unwrap();
    let id = h.state_mut().run("type.create", json!({"x": 20, "y": 80, "text": "Hello", "size": 20})).unwrap()["layer"].as_u64().unwrap();
    h.run_steps(2);
    assert!(h.state().ui.text_edit.is_none());
    let cmd_shift = Modifiers::COMMAND | Modifiers::SHIFT;
    h.key_press_modifiers(cmd_shift, Key::I);
    h.run_steps(2);
    assert_eq!(history(&h).last().map(String::as_str), Some("Inverse"));
    h.key_press_modifiers(cmd_shift, Key::K);
    h.run_steps(3);
    assert_eq!(h.state().ui.dialogs.len(), 1, "Color Settings opens");
    assert!(!char_style(&h, id, 0).faux_italic && char_style(&h, id, 0).caps == photocraft_doc::text::Caps::Normal);
}

/// ⌥↑/↓ leading (2 pt, 10 with ⌘), ⇧⌥↑/↓ baseline shift and ⌥←/→ tracking over a selection
/// (20, 100 with ⌘): one step per press, from each character's own value, the selection kept.
#[test]
fn type_tool_format_keys_step_leading_baseline_and_tracking() {
    let mut h = harness();
    let id = h.state_mut().run("type.create", json!({"x": 20, "y": 80, "text": "Hello world", "size": 10})).unwrap()["layer"].as_u64().unwrap();
    edit_type(&mut h, id, 0, 5);
    let before = history(&h).len();
    let press = |h: &mut Harness<'static, PhotocraftApp>, m: Modifiers, k: Key| {
        h.key_press_modifiers(m, k);
        h.run_steps(2);
    };
    // Auto leading is 120% of 10 pt: ⌥↓ makes it 14.
    press(&mut h, Modifiers::ALT, Key::ArrowDown);
    assert_eq!(char_style(&h, id, 0).leading_pt, Some(14.0));
    press(&mut h, Modifiers::ALT, Key::ArrowUp);
    assert_eq!(char_style(&h, id, 4).leading_pt, Some(12.0));
    press(&mut h, Modifiers::ALT | Modifiers::COMMAND, Key::ArrowDown);
    assert_eq!(char_style(&h, id, 0).leading_pt, Some(22.0));
    press(&mut h, Modifiers::ALT | Modifiers::SHIFT, Key::ArrowUp);
    assert_eq!(char_style(&h, id, 0).baseline_shift_pt, 2.0);
    press(&mut h, Modifiers::ALT | Modifiers::SHIFT | Modifiers::COMMAND, Key::ArrowDown);
    assert_eq!(char_style(&h, id, 0).baseline_shift_pt, -8.0);
    press(&mut h, Modifiers::ALT, Key::ArrowRight);
    assert_eq!(char_style(&h, id, 0).tracking, 20.0);
    press(&mut h, Modifiers::ALT | Modifiers::COMMAND, Key::ArrowLeft);
    assert_eq!(char_style(&h, id, 4).tracking, -80.0);
    assert_eq!(history(&h).len(), before + 7, "one history step per press");
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| (e.anchor, e.caret)), Some((0, 5)), "the selection stays");
    let rest = char_style(&h, id, 7);
    assert_eq!((rest.leading_pt, rest.baseline_shift_pt, rest.tracking), (None, 0.0, 0.0));
    // A collapsed caret: ⌥↑/↓ change nothing and don't move the caret.
    edit_type(&mut h, id, 8, 8);
    let steps = history(&h).len();
    press(&mut h, Modifiers::ALT, Key::ArrowDown);
    press(&mut h, Modifiers::ALT | Modifiers::SHIFT, Key::ArrowUp);
    assert_eq!(history(&h).len(), steps);
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| (e.anchor, e.caret)), Some((8, 8)));
}

/// ⌥⇧⌘H toggles Hyphenate for the paragraphs the selection or caret touches.
#[test]
fn type_tool_format_keys_toggle_hyphenate() {
    let mut h = harness();
    let id = h.state_mut().run("type.create", json!({"box": [10, 10, 300, 200], "text": "one\ntwo\nthree", "size": 12})).unwrap()["layer"].as_u64().unwrap();
    let keys = Modifiers::COMMAND | Modifiers::ALT | Modifiers::SHIFT;
    // The caret in the second paragraph.
    edit_type(&mut h, id, 5, 5);
    let before = history(&h).len();
    h.key_press_modifiers(keys, Key::H);
    h.run_steps(2);
    assert_eq!([0, 5, 9].map(|c| para_style(&h, id, c).hyphenate), [false, true, false]);
    // A selection over the first two: not all hyphenate, so both do; again, neither.
    edit_type(&mut h, id, 1, 5);
    h.key_press_modifiers(keys, Key::H);
    h.run_steps(2);
    assert_eq!([0, 5, 9].map(|c| para_style(&h, id, c).hyphenate), [true, true, false]);
    h.key_press_modifiers(keys, Key::H);
    h.run_steps(2);
    assert_eq!([0, 5, 9].map(|c| para_style(&h, id, c).hyphenate), [false, false, false]);
    assert_eq!(history(&h).len(), before + 3);
}

/// The formatting keys never panic: an edit of a layer that has gone, or with no document.
#[test]
fn type_tool_format_keys_survive_hostile_state() {
    let mut h = harness();
    edit_type(&mut h, 9999, 0, 3);
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, Key::U);
    h.key_press_modifiers(Modifiers::ALT, Key::ArrowDown);
    h.run_steps(2);
    assert!(h.state().ui.text_edit.is_none(), "an edit of a missing layer ends");
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    for ed in [None, Some(1)] {
        app.ui.text_edit = ed.map(|layer| crate::state::TextEdit {
            layer,
            caret: 4,
            anchor: 0,
            session: "x".into(),
            created: false,
            mask: false,
            dragging: false,
            resize: None,
            preedit: None,
        });
        crate::type_tool::toggle_char_style(&mut app, crate::type_tool::CharToggle::AllCaps);
        crate::type_tool::step_char_metric(&mut app, "leadingBy", 2.0);
        crate::type_tool::step_char_metric(&mut app, "trackingBy", f32::NAN);
        crate::type_tool::toggle_hyphenate(&mut app);
    }
    assert!(app.session.active().is_none());
}

/// The Character panel's kerning field reads and parses Photoshop's values.
#[test]
fn kerning_field_values() {
    use photocraft_doc::text::Kerning;
    assert_eq!(crate::type_tool::kerning_label((Kerning::Metrics, 0.0)), "Metrics");
    assert_eq!(crate::type_tool::kerning_label((Kerning::Optical, 0.0)), "Optical");
    assert_eq!(crate::type_tool::kerning_label((Kerning::Off, 0.0)), "0");
    assert_eq!(crate::type_tool::kerning_label((Kerning::Off, -49.6)), "-50");
    for (s, v) in [("metrics", Some(json!("metrics"))), (" Optical ", Some(json!("optical"))), ("120", Some(json!(120.0))), ("-25.4", Some(json!(-25.0)))] {
        assert_eq!(crate::type_tool::parse_kerning(s), v, "{s}");
    }
    for s in ["", "tight", "1e9", "-5000", "NaN", "inf"] {
        assert_eq!(crate::type_tool::parse_kerning(s), None, "{s}");
    }
}

/// Sampling shows a pipette instead of the crosshair: the Eyedropper tool, and a painting tool
/// with ⌥ held. Preferences › Cursors › Other Cursors = Precise keeps the crosshair.
#[test]
fn eyedropper_and_alt_sampling_show_a_pipette() {
    let mut h = harness();
    let p = h.state().last_canvas_rect.center();
    let cursor = |h: &mut Harness<'static, PhotocraftApp>| {
        h.hover_at(p);
        h.run_steps(2);
        h.output().platform_output.cursor_icon
    };
    h.state_mut().ui.tool = crate::state::Tool::Eyedropper;
    assert_eq!(cursor(&mut h), egui::CursorIcon::None, "the pipette replaces the pointer");
    h.state_mut().ui.tool = crate::state::Tool::Brush;
    h.event(egui::Event::ModifiersChanged(Modifiers::ALT));
    assert_eq!(cursor(&mut h), egui::CursorIcon::None, "⌥ samples with a pipette");
    h.state_mut().run("prefs.set", json!({"values": {"cursors.other": "precise"}})).unwrap();
    // Every platform reports the crosshair. The desktop builds also hand the OS a black-and-white
    // bitmap for it (`tool_cursor`, #1160, #1343), with `Crosshair` underneath as the fallback.
    let precise = |h: &mut Harness<'static, PhotocraftApp>| {
        let icon = cursor(h);
        (icon, h.output().platform_output.cursor_image.is_some())
    };
    let desktop = cfg!(not(target_arch = "wasm32"));
    assert_eq!(precise(&mut h), (egui::CursorIcon::Crosshair, desktop), "Precise keeps the crosshair");
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.state_mut().ui.tool = crate::state::Tool::Eyedropper;
    assert_eq!(precise(&mut h), (egui::CursorIcon::Crosshair, desktop));
}

/// The Paint Bucket shows a bucket instead of the crosshair (#3968); Precise Other Cursors keeps
/// the crosshair.
#[test]
fn paint_bucket_shows_a_bucket_cursor() {
    let mut h = harness();
    let p = h.state().last_canvas_rect.center();
    h.state_mut().ui.tool = crate::state::Tool::PaintBucket;
    h.hover_at(p);
    h.run_steps(2);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "the bucket replaces the pointer");
    h.state_mut().run("prefs.set", json!({"values": {"cursors.other": "precise"}})).unwrap();
    h.hover_at(p);
    h.run_steps(2);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::Crosshair, "Precise keeps the crosshair");
}

/// The Hand tool, and Space held over another tool, show an open hand while hovering and a fist
/// while they pan (#2196). The OS draws those everywhere but Windows, where winit maps `Grab` to
/// the four-arrow move cursor, so the app hands it a bitmap there (`Crosshair` is the fallback).
#[test]
fn hand_tool_shows_an_open_hand_and_a_fist_while_panning() {
    let mut h = harness();
    let p = h.state().last_canvas_rect.center();
    let windows = cfg!(target_os = "windows");
    // The icon the OS gets, and the bitmap for it if any.
    let pointer = |h: &Harness<'static, PhotocraftApp>| {
        let out = &h.output().platform_output;
        (out.cursor_icon, out.cursor_image.as_ref().map(|i| i.rgba.clone()))
    };
    let expect = |got: (egui::CursorIcon, Option<std::sync::Arc<[u8]>>), native: egui::CursorIcon, what: &str| {
        if windows {
            assert_eq!(got.0, egui::CursorIcon::Crosshair, "{what}: fallback under the bitmap");
            assert!(got.1.is_some(), "{what}: a bitmap");
        } else {
            assert_eq!(got, (native, None), "{what}");
        }
        got.1
    };

    h.state_mut().ui.tool = crate::state::Tool::Hand;
    h.hover_at(p);
    h.run_steps(2);
    let open = expect(pointer(&h), egui::CursorIcon::Grab, "hovering with the Hand");
    press(&mut h, p, true);
    for i in 1..=3 {
        h.hover_at(p + vec2(10.0 * i as f32, 0.0));
        h.run_steps(1);
    }
    let fist = expect(pointer(&h), egui::CursorIcon::Grabbing, "dragging with the Hand");
    if windows {
        assert_ne!(open, fist, "a fist is not the open hand");
    }
    press(&mut h, p + vec2(30.0, 0.0), false);
    h.run_steps(2);
    expect(pointer(&h), egui::CursorIcon::Grab, "after the drag");

    // Space turns any tool into the Hand while held.
    h.state_mut().ui.tool = crate::state::Tool::Brush;
    h.hover_at(p);
    h.run_steps(2);
    let brush = pointer(&h);
    assert_ne!(brush.0, egui::CursorIcon::Grab, "the Brush is not a hand");
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    expect(pointer(&h), egui::CursorIcon::Grab, "Space over the Brush");
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert_eq!(pointer(&h), brush, "released: the Brush's own pointer is back");
}

/// With a dialog open the canvas still pans, and the Hand's pointer follows (#2196).
#[test]
fn hand_pointer_over_the_canvas_under_a_dialog() {
    let mut h = harness();
    open_dialog(&mut h);
    h.state_mut().ui.tool = crate::state::Tool::Hand;
    let p = free_canvas(&h);
    h.hover_at(p);
    h.run_steps(3);
    let out = &h.output().platform_output;
    if cfg!(target_os = "windows") {
        assert!(out.cursor_image.is_some());
    } else {
        assert_eq!(out.cursor_icon, egui::CursorIcon::Grab);
    }
}

/// Esc while drawing with the Pen ends the path where it is, left open, and keeps it as the work
/// path, as ↩ does; it used to throw the path away (#1769).
#[test]
fn esc_ends_a_pen_path_and_keeps_it() {
    let mut h = harness();
    h.state_mut().ui.tool = crate::state::Tool::Pen;
    h.state_mut().ui.pen = Some(crate::vector_ui::PenPath { knots: vec![[[50.0, 50.0]; 3], [[150.0, 50.0]; 3], [[150.0, 120.0]; 3]], ..Default::default() });
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(h.state().ui.pen.is_none(), "the Pen leaves drawing state");
    let doc = &h.state().session.active().unwrap().doc;
    let path = doc.work_path.as_ref().expect("the path is kept as the work path");
    assert_eq!(path.subpaths.len(), 1);
    assert!(!path.subpaths[0].closed, "left open");
    assert_eq!(path.subpaths[0].knots.len(), 3);
    assert_eq!(h.state().ui.selected_path.as_deref(), Some("work"));
}

/// Over the first anchor of a Pen path the canvas shows the close-path pointer (#4596), a bitmap
/// of its own; anywhere else the Pen keeps its crosshair.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn pen_shows_the_close_path_pointer_over_the_first_anchor() {
    let mut h = harness();
    h.state_mut().ui.tool = crate::state::Tool::Pen;
    h.state_mut().ui.pen = Some(crate::vector_ui::PenPath { knots: vec![[[50.0, 50.0]; 3], [[150.0, 50.0]; 3]], ..Default::default() });
    let screen = |h: &Harness<'static, PhotocraftApp>, x: f32, y: f32| {
        let v = &h.state().ui.views[0];
        let c = h.state().last_canvas_rect.center();
        pos2(c.x + (x - v.center[0]) * v.zoom, c.y + (y - v.center[1]) * v.zoom)
    };
    let pointer = |h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32| {
        let p = screen(h, x, y);
        h.hover_at(p);
        h.run_steps(2);
        let out = &h.output().platform_output;
        (out.cursor_icon, out.cursor_image.as_ref().map(|i| i.rgba.clone()))
    };
    let close = pointer(&mut h, 50.0, 50.0);
    assert_eq!(close.0, egui::CursorIcon::Crosshair, "the fallback under the bitmap");
    assert!(close.1.is_some(), "a bitmap");
    let elsewhere = pointer(&mut h, 200.0, 200.0);
    assert_ne!(elsewhere.1, close.1, "empty canvas: the plain Pen pointer");
    assert_ne!(pointer(&mut h, 150.0, 50.0).1, close.1, "the last anchor doesn't close");
    assert_eq!(pointer(&mut h, 50.0, 50.0), close, "back on the first anchor");
}

/// Esc with no active operation deselects (#2674), like ⌘D.
#[test]
fn escape_deselects_when_nothing_else_needs_cancelling() {
    let mut h = harness();
    h.state_mut().run("select.all", json!({})).unwrap();
    h.run_steps(1);
    assert!(h.state().session.active().unwrap().doc.selection.is_some());
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(h.state().session.active().unwrap().doc.selection.is_none(), "Esc deselects");
    assert!(!h.state().ui.status_error, "no error: {}", h.state().ui.status);
    // With nothing selected, Esc stays untouched (no history step, no error).
    let steps = h.state().session.active().unwrap().history.entries().len();
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps, "no Deselect step");
}

/// The dialog owns Esc: it cancels the dialog and the selection stays.
#[test]
fn escape_with_a_dialog_open_cancels_the_dialog_not_the_selection() {
    let mut h = harness();
    h.state_mut().run("select.all", json!({})).unwrap();
    open_dialog(&mut h);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(h.state().ui.dialogs.is_empty(), "Esc cancelled the dialog");
    assert!(h.state().session.active().unwrap().doc.selection.is_some(), "the selection stays");
}

/// The command palette owns Esc: it closes the palette and the selection stays.
#[test]
fn escape_with_the_palette_open_does_not_deselect() {
    let mut h = harness();
    h.state_mut().run("select.all", json!({})).unwrap();
    h.state_mut().ui.palette_open = true;
    h.run_steps(1);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(h.state().session.active().unwrap().doc.selection.is_some(), "Esc belongs to the palette");
}

/// Targa Options owns Esc (it cancels the save), so the selection stays.
#[test]
fn escape_with_targa_options_open_does_not_deselect() {
    let mut h = harness();
    h.state_mut().run("select.all", json!({})).unwrap();
    h.state_mut().save_as(Some("t.tga".into())).unwrap();
    assert!(h.state().targa_options.is_some());
    h.run_steps(1);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(h.state().session.active().unwrap().doc.selection.is_some(), "Esc belongs to Targa Options");
}

/// How much of the 400 × 300 document is on screen: the visible document span on each axis,
/// clipped to the image (0 when the image has left the view on that axis).
fn visible_image(h: &Harness<'static, PhotocraftApp>) -> [f32; 2] {
    let r = h.state().last_canvas_rect;
    let (a, b) = (doc_at(h, r.min), doc_at(h, r.max));
    [(b[0].min(400.0) - a[0].max(0.0)).max(0.0), (b[1].min(300.0) - a[1].max(0.0)).max(0.0)]
}

#[test]
fn wheel_zoom_over_the_pasteboard_keeps_the_image_in_view() {
    // #3720: zooming in with the pointer on the gray pasteboard anchored on the pointer, so a few
    // notches pushed the image out of the window. The anchor is now the nearest point of the image.
    for (zoom_with_wheel, m) in [(true, Modifiers::NONE), (false, Modifiers::ALT)] {
        let mut h = harness();
        h.state_mut().session.edit_prefs(|p| p.general.zoom_with_scroll_wheel = zoom_with_wheel);
        let r = h.state().last_canvas_rect;
        let p = pos2(r.left() + 20.0, r.top() + 20.0);
        let d = doc_at(&h, p);
        assert!(d[0] < 0.0 && d[1] < 0.0, "the pointer starts on the pasteboard: {d:?}");
        h.hover_at(p);
        h.run_steps(2);
        let z0 = zoom(&h);
        for _ in 0..20 {
            wheel(&h, 1.0, m);
            h.run_steps(3);
        }
        h.run_steps(20);
        assert!(zoom(&h) > z0 * 5.0, "the notches zoomed in: {z0} -> {}", zoom(&h));
        let v = visible_image(&h);
        assert!(v[0] > 0.0 && v[1] > 0.0, "zoom_with_wheel {zoom_with_wheel}: the image left the view {v:?}");
        // The image's corner nearest the pointer is the anchor: it stays where it was on screen.
        let c = doc_at(&h, pos2(r.left() + 20.0, r.top() + 20.0));
        assert!(c[0] < 0.0 && c[1] < 0.0, "still pasteboard under the pointer, the corner held: {c:?}");
        // A pinch from there behaves the same.
        h.event(egui::Event::Zoom(4.0));
        h.run_steps(3);
        let v = visible_image(&h);
        assert!(v[0] > 0.0 && v[1] > 0.0, "pinch: the image left the view {v:?}");
    }
}

#[test]
fn wheel_zoom_over_the_image_still_anchors_on_the_pointer() {
    let mut h = harness();
    let r = h.state().last_canvas_rect;
    let p = pos2(r.center().x - 150.0, r.center().y - 100.0);
    h.hover_at(p);
    h.run_steps(2);
    let d0 = doc_at(&h, p);
    assert!(d0[0] > 0.0 && d0[1] > 0.0);
    for _ in 0..5 {
        wheel(&h, 1.0, Modifiers::ALT);
        h.run_steps(3);
    }
    h.run_steps(20);
    let d1 = doc_at(&h, p);
    assert!((d1[0] - d0[0]).abs() < 0.1 && (d1[1] - d0[1]).abs() < 0.1, "{d0:?} -> {d1:?}");
}

/// #4658: with the Move tool's Show Transform Controls, the band just outside the box shows the
/// curved rotate arrow, not a plain pointer, and one drag from there rotates the layer.
#[test]
fn move_tool_controls_rotate_from_just_outside_the_box() {
    let mut h = harness();
    h.state_mut().run("shape.create", json!({"kind": "rect", "rect": [150, 100, 100, 100], "fill": "#ff0000"})).unwrap();
    h.state_mut().ui.tool = crate::state::Tool::Move;
    h.state_mut().ui.tool_options.move_show_transform = true;
    h.run_steps(3);
    let screen = |h: &Harness<'static, PhotocraftApp>, x: f32, y: f32| {
        let v = &h.state().ui.views[0];
        let c = h.state().last_canvas_rect.center();
        pos2(c.x + (x - v.center[0]) * v.zoom, c.y + (y - v.center[1]) * v.zoom)
    };
    let zoom = h.state().ui.views[0].zoom;
    // Just outside the top edge, a quarter of the way along: the rotation band.
    let at = screen(&h, 175.0, 100.0) - vec2(0.0, 12.0);
    h.hover_at(at);
    h.run_steps(2);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "the drawn rotate arrow replaces the pointer");
    press(&mut h, at, true);
    let centre = screen(&h, 200.0, 150.0);
    // A quarter turn around the box's centre.
    for k in 1..=10 {
        let a = std::f32::consts::FRAC_PI_2 * k as f32 / 10.0;
        let (dx, dy) = (at.x - centre.x, at.y - centre.y);
        let q = centre + vec2(dx * a.cos() - dy * a.sin(), dx * a.sin() + dy * a.cos());
        h.hover_at(q);
        h.run_steps(1);
    }
    let end = h.state().ui.transform.as_ref().map(|t| t.quad).expect("the drag opened Free Transform");
    let tilt = (end[1][1] - end[0][1]).atan2(end[1][0] - end[0][0]).to_degrees();
    assert!((tilt - 90.0).abs() < 8.0 || (tilt + 270.0).abs() < 8.0, "rotated about a quarter turn, got {tilt}° (zoom {zoom})");
}
