//! #3373: Field Blur's pins are placed, dragged and tuned on the image under the dialog, the
//! preview follows them, and OK applies them.

use egui::{Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::blur_pins::{COMMAND, Pin, pins, selected};
use crate::canvas::ViewXform;

/// A 400×300 document at 400 %, so it fills the window round the dialog, with Field Blur open.
fn harness(selection: bool) -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    if selection {
        app.run("select.rect", json!({"x": 100, "y": 50, "width": 200, "height": 100})).unwrap();
    }
    app.sync_views();
    app.ui.extras.rulers = false;
    let mut h = Harness::builder().with_size(vec2(1200.0, 800.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
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
    let v = &mut h.state_mut().ui.views[0];
    (v.zoom, v.center, v.fit_pending) = (4.0, [200.0, 150.0], false);
    crate::filter_dialog::open(h.state_mut(), COMMAND).unwrap();
    h.run_steps(4);
    h
}

fn fields(h: &Harness<'static, PhotocraftApp>) -> serde_json::Map<String, serde_json::Value> {
    h.state().ui.dialogs.last().unwrap().fields.clone()
}

fn xf(h: &Harness<'static, PhotocraftApp>) -> ViewXform {
    let app = h.state();
    let v = &app.ui.views[0];
    ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: false,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    }
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: Modifiers::NONE });
    h.run_steps(1);
}

fn move_to(h: &mut Harness<'static, PhotocraftApp>, p: Pos2) {
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
}

fn drag(h: &mut Harness<'static, PhotocraftApp>, path: &[Pos2]) {
    move_to(h, path[0]);
    button(h, path[0], true);
    for p in &path[1..] {
        move_to(h, *p);
    }
    button(h, *path.last().unwrap(), false);
    h.run_steps(2);
}

/// Where pin `p` is on screen.
fn on_screen(h: &Harness<'static, PhotocraftApp>, p: Pin, bounds: [f64; 4]) -> Pos2 {
    let (x, y) = (bounds[0] + p.x * (bounds[2] - bounds[0]), bounds[1] + p.y * (bounds[3] - bounds[1]));
    xf(h).to_screen(x as f32, y as f32)
}

#[test]
fn it_opens_with_one_selected_pin_in_the_middle_of_the_bounds() {
    let h = harness(false);
    let f = fields(&h);
    assert_eq!(pins(&f), vec![Pin { x: 0.5, y: 0.5, blur: 15.0 }]);
    assert_eq!(selected(&f), Some(0));
    assert_eq!(f["__bounds"], json!([0, 0, 400, 300]));
    assert!(!f.contains_key("centerX") && !f.contains_key("centerY"), "the pins replace the single centre");
    // With a selection the pins are fractions of its bounds, as the engine reads them.
    let h = harness(true);
    assert_eq!(fields(&h)["__bounds"], json!([100, 50, 300, 150]));
}

#[test]
fn pins_are_added_moved_tuned_and_removed_on_the_image() {
    let mut h = harness(false);
    let r = h.state().last_canvas_rect;
    // A click on the free image (left of the dialog) adds a pin there, selected.
    let a = pos2(r.left() + 80.0, r.bottom() - 80.0);
    move_to(&mut h, a);
    button(&mut h, a, true);
    button(&mut h, a, false);
    h.run_steps(2);
    let f = fields(&h);
    let list = pins(&f);
    assert_eq!(list.len(), 2, "{list:?}");
    assert_eq!(selected(&f), Some(1));
    let at = xf(&h).to_doc(a);
    assert!((list[1].x - at[0] / 400.0).abs() < 1e-3 && (list[1].y - at[1] / 300.0).abs() < 1e-3, "{:?} vs {at:?}", list[1]);
    assert_eq!(list[1].blur, 15.0, "a new pin takes the selected pin's blur");
    // Dragging the pin moves it by the pointer's movement (4 screen points per pixel at 400 %).
    let b = a + vec2(40.0, -20.0);
    drag(&mut h, &[a, a + vec2(20.0, -10.0), b]);
    let moved = pins(&fields(&h))[1];
    assert!((moved.x - (list[1].x + 10.0 / 400.0)).abs() < 1e-3 && (moved.y - (list[1].y - 5.0 / 300.0)).abs() < 1e-3, "{moved:?}");
    assert!(fields(&h).get("__drag").is_none(), "the drag ends on release");
    // A quarter turn clockwise on its ring adds a quarter of the 0–500 px range.
    let c = on_screen(&h, moved, [0.0, 0.0, 400.0, 300.0]);
    let ring: Vec<Pos2> = (0..=6).map(|k| c + vec2(24.0 * (k as f32 * 0.2618).cos(), 24.0 * (k as f32 * 0.2618).sin())).collect();
    drag(&mut h, &ring);
    let tuned = pins(&fields(&h))[1];
    assert!((tuned.blur - 140.0).abs() < 2.0, "15 + 125 px: {}", tuned.blur);
    assert!((tuned.x - moved.x).abs() < 1e-9, "the ring doesn't move the pin");
    assert_eq!(fields(&h)["blur"], json!(tuned.blur), "the dialog's Blur follows the selected pin");
    // The preview runs the filter with the pins (computed by the canvas's GPU path in the app;
    // asked for directly here, as the headless harness has no GPU).
    assert!(crate::canvas::ensure_filter_preview(h.state_mut(), 0, &egui::Context::default()).is_some());
    let preview = h.state().filter_preview.as_ref().expect("a live preview");
    assert_eq!(preview.key.params["pins"].as_array().map(Vec::len), Some(2));
    // Delete removes the selected pin; OK applies the rest in one step.
    h.key_press(egui::Key::Delete);
    h.run_steps(2);
    assert_eq!(pins(&fields(&h)).len(), 1);
    assert_eq!(selected(&fields(&h)), Some(0));
    let id = h.state().ui.dialogs.last().unwrap().id;
    let steps = h.state().session.active().unwrap().history.past_len();
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    let (cmd, p) = h.state().session.journal.last().cloned().unwrap();
    assert_eq!(cmd, COMMAND);
    assert_eq!(p["pins"], json!([{"x": 0.5, "y": 0.5, "blur": 15.0}]), "{p}");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
}

/// A click outside the bounds (beside a selection) adds no pin; pins are capped.
#[test]
fn pins_stay_inside_the_bounds_and_are_capped() {
    let mut h = harness(true);
    let r = h.state().last_canvas_rect;
    let outside = pos2(r.left() + 30.0, r.top() + 30.0);
    assert!(xf(&h).to_doc(outside)[0] < 100.0, "left of the selection");
    move_to(&mut h, outside);
    button(&mut h, outside, true);
    button(&mut h, outside, false);
    h.run_steps(2);
    assert_eq!(pins(&fields(&h)).len(), 1);
    let many: Vec<_> = (0..200).map(|i| json!({"x": f64::from(i) / 200.0, "y": 0.5, "blur": 1e9})).collect();
    h.state_mut().ui.dialogs.last_mut().unwrap().fields.insert("pins".into(), json!(many));
    let list = pins(&fields(&h));
    assert_eq!(list.len(), crate::blur_pins::MAX_PINS);
    assert!(list.iter().all(|p| p.blur == 500.0));
}

#[test]
fn proxy_previews_scale_the_pins_blur() {
    let mut p = json!({"pins": [{"x": 0.2, "y": 0.3, "blur": 40.0}, {"x": 0.9, "y": 0.9, "blur": 0.0}]});
    crate::blur_pins::scale_for_proxy(&mut p, 4);
    assert_eq!(p, json!({"pins": [{"x": 0.2, "y": 0.3, "blur": 10.0}, {"x": 0.9, "y": 0.9, "blur": 0.0}]}));
    crate::blur_pins::scale_for_proxy(&mut p, 1);
    assert_eq!(p["pins"][0]["blur"], json!(10.0));
}

/// Alt+Cancel → Reset (#3871) starts the workspace again from one pin in the middle, as when it
/// opened, instead of leaving it with no pins; the bounds captured on opening stay.
#[test]
fn alt_reset_starts_again_from_one_pin() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    let id = crate::filter_dialog::open(&mut app, COMMAND).unwrap();
    let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
    let bounds = fields["__bounds"].clone();
    fields.insert("pins".into(), json!([{"x": 0.1, "y": 0.2, "blur": 40.0}, {"x": 0.8, "y": 0.7, "blur": 90.0}]));
    fields.insert("__selected".into(), json!(1));
    assert!(crate::dialog_alt_reset::supported(&fields));
    crate::dialog_alt_reset::apply(&mut app, &egui::Context::default(), &mut fields);
    assert_eq!(pins(&fields), [Pin { x: 0.5, y: 0.5, blur: 15.0 }]);
    assert_eq!(selected(&fields), Some(0));
    assert_eq!(fields["__bounds"], bounds);
    assert!(!fields.contains_key("centerX") && !fields.contains_key("centerY"), "{fields:?}");
}
