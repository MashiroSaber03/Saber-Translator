//! Photoshop's tool keys that aren't plain shortcuts: Space+F cycles the pasteboard colour.

use egui::{Key, Modifiers};
use serde_json::json;

use crate::PhotocraftApp;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 20, "height": 10})).unwrap();
    app
}

/// Space+F steps the pasteboard colour forward and ⇧Space+F back, round and round; F alone is
/// still Cycle Screen Mode.
#[test]
fn space_f_cycles_the_canvas_colour() {
    use photocraft_engine::prefs::CanvasColor;
    let mut app = app();
    let ctx = egui::Context::default();
    let frame = |app: &mut PhotocraftApp, events: Vec<egui::Event>| {
        ctx.begin_pass(egui::RawInput { events, ..Default::default() });
        crate::shortcuts::handle(app, &ctx);
        ctx.end_pass().textures_delta.clear();
    };
    let key = |key, pressed, modifiers| egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers };
    let color = |app: &PhotocraftApp| app.session.prefs().interface.canvas_color;
    frame(&mut app, vec![key(Key::Space, true, Modifiers::NONE)]);
    frame(&mut app, vec![key(Key::F, true, Modifiers::NONE), key(Key::F, false, Modifiers::NONE)]);
    assert_eq!(color(&app), CanvasColor::Black);
    frame(&mut app, vec![egui::Event::ModifiersChanged(Modifiers::SHIFT), key(Key::F, true, Modifiers::SHIFT), key(Key::F, false, Modifiers::SHIFT)]);
    frame(&mut app, vec![key(Key::F, true, Modifiers::SHIFT), key(Key::F, false, Modifiers::SHIFT)]);
    assert_eq!(color(&app), CanvasColor::Custom, "backwards past Default wraps round");
    frame(&mut app, vec![egui::Event::ModifiersChanged(Modifiers::NONE), key(Key::Space, false, Modifiers::NONE)]);
    let mode = app.ui.view.screen_mode.clone();
    frame(&mut app, vec![key(Key::F, true, Modifiers::NONE), key(Key::F, false, Modifiers::NONE)]);
    assert_eq!(color(&app), CanvasColor::Custom, "F alone leaves the canvas colour");
    assert_ne!(app.ui.view.screen_mode, mode, "and cycles the screen mode");
}
