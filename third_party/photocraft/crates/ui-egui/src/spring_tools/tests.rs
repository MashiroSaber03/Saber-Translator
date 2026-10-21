//! #2672 through the real key handling and canvas: a tapped tool letter switches for good, a held
//! one (or one used on the canvas) springs back on release.

use egui::{Event, Key, Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use super::*;
use crate::canvas::ViewXform;

/// Frames are 1/60 s apart, so a hold is a count of frames.
const DT: f32 = 1.0 / 60.0;

fn harness(tool: Tool) -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.sync_views();
    app.ui.extras.rulers = false;
    app.ui.tool = tool;
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).with_step_dt(DT).build_ui_state(
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
    v.center = [200.0, 150.0];
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

/// One key event with its modifiers, run as a frame.
fn key_event(h: &mut Harness<'static, PhotocraftApp>, k: Key, pressed: bool, repeat: bool, m: Modifiers) {
    h.event(Event::ModifiersChanged(m));
    h.event(Event::Key { key: k, physical_key: None, pressed, repeat, modifiers: m });
    h.run_steps(1);
}

fn down(h: &mut Harness<'static, PhotocraftApp>, k: Key) {
    key_event(h, k, true, false, Modifiers::NONE);
}

fn up(h: &mut Harness<'static, PhotocraftApp>, k: Key) {
    key_event(h, k, false, false, Modifiers::NONE);
    h.run_steps(2);
}

/// Wait `secs` of input time.
fn wait(h: &mut Harness<'static, PhotocraftApp>, secs: f64) {
    h.run_steps((secs / f64::from(DT)).ceil() as usize);
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, pressed: bool, m: Modifiers) {
    h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: m });
    h.run_steps(1);
}

fn move_to(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32) {
    let p = screen(h, x, y);
    h.event(Event::PointerMoved(p));
    h.run_steps(1);
}

fn click(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    move_to(h, x, y);
    let p = screen(h, x, y);
    button(h, p, true, m);
    button(h, p, false, m);
}

#[test]
fn a_tap_switches_tools_for_good() {
    let mut h = harness(Tool::Brush);
    down(&mut h, Key::E);
    assert_eq!(h.state().ui.tool, Tool::Eraser);
    assert!(h.state().ui.spring.is_some(), "the press is pending: tap or hold");
    up(&mut h, Key::E);
    assert_eq!(h.state().ui.tool, Tool::Eraser);
    assert_eq!(h.state().ui.spring, None);
}

#[test]
fn a_held_letter_gives_the_previous_tool_back() {
    let mut h = harness(Tool::Brush);
    down(&mut h, Key::E);
    wait(&mut h, SPRING_HOLD_SECS + 0.1);
    assert_eq!(h.state().ui.tool, Tool::Eraser, "the temporary tool is the tool shown meanwhile");
    up(&mut h, Key::E);
    assert_eq!(h.state().ui.tool, Tool::Brush);
    assert_eq!(h.state().ui.spring, None);
}

#[test]
fn hold_z_click_zooms_then_a_quick_release_restores() {
    let mut h = harness(Tool::Brush);
    let z0 = h.state().ui.views[0].zoom;
    down(&mut h, Key::Z);
    click(&mut h, 200.0, 150.0, Modifiers::NONE);
    up(&mut h, Key::Z);
    let z1 = h.state().ui.views[0].zoom;
    assert!(z1 > z0, "{z0} -> {z1}");
    assert_eq!(h.state().ui.tool, Tool::Brush, "a canvas click makes even a short hold temporary");
    assert!(!h.state().session.active().unwrap().history.can_undo(), "nothing was painted");
    // ⌥ with a held Z zooms out, as ⌥ does with the Zoom tool.
    down(&mut h, Key::Z);
    h.event(Event::ModifiersChanged(Modifiers::ALT));
    click(&mut h, 200.0, 150.0, Modifiers::ALT);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    up(&mut h, Key::Z);
    assert!(h.state().ui.views[0].zoom < z1);
    assert_eq!(h.state().ui.tool, Tool::Brush);
}

#[test]
fn shift_letter_cycles_and_never_springs() {
    let mut h = harness(Tool::Brush);
    h.state_mut().session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = true);
    key_event(&mut h, Key::B, true, false, Modifiers::SHIFT);
    assert_eq!(h.state().ui.tool, Tool::Pencil);
    assert_eq!(h.state().ui.spring, None);
    wait(&mut h, SPRING_HOLD_SECS + 0.1);
    key_event(&mut h, Key::B, false, false, Modifiers::SHIFT);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, Tool::Pencil);

    // Without the preference ⇧E is still a permanent switch, however long it is held.
    let mut h = harness(Tool::Brush);
    key_event(&mut h, Key::E, true, false, Modifiers::SHIFT);
    wait(&mut h, SPRING_HOLD_SECS + 0.1);
    key_event(&mut h, Key::E, false, false, Modifiers::SHIFT);
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, Tool::Eraser);
    assert_eq!(h.state().ui.spring, None);
}

#[test]
fn key_repeats_are_not_presses() {
    let mut h = harness(Tool::Brush);
    down(&mut h, Key::L);
    assert_eq!(h.state().ui.tool, Tool::Lasso);
    for _ in 0..8 {
        key_event(&mut h, Key::L, true, true, Modifiers::NONE);
        wait(&mut h, 0.05);
    }
    assert_eq!(h.state().ui.tool, Tool::Lasso, "repeats don't cycle the group");
    up(&mut h, Key::L);
    assert_eq!(h.state().ui.tool, Tool::Brush);
}

#[test]
fn a_drag_keeps_the_temporary_tool_until_the_button_is_up() {
    let mut h = harness(Tool::Brush);
    down(&mut h, Key::M);
    assert_eq!(h.state().ui.tool, Tool::RectMarquee);
    move_to(&mut h, 50.0, 50.0);
    let p = screen(&h, 50.0, 50.0);
    button(&mut h, p, true, Modifiers::NONE);
    move_to(&mut h, 100.0, 90.0);
    up(&mut h, Key::M);
    assert_eq!(h.state().ui.tool, Tool::RectMarquee, "the drag finishes with the temporary tool");
    move_to(&mut h, 150.0, 110.0);
    let p = screen(&h, 150.0, 110.0);
    button(&mut h, p, false, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, Tool::Brush);
    assert_eq!(h.state().ui.spring, None);
    let sel = h.state().session.active().unwrap().doc.selection.as_ref().map(|s| s.content_bounds());
    assert!(sel.is_some_and(|r| r.width() > 50 && r.height() > 30), "{sel:?}");
}

#[test]
fn the_newest_letter_wins_and_release_restores_the_original() {
    let mut h = harness(Tool::Brush);
    down(&mut h, Key::E);
    down(&mut h, Key::L);
    assert_eq!(h.state().ui.tool, Tool::Lasso);
    up(&mut h, Key::E);
    assert_eq!(h.state().ui.tool, Tool::Lasso, "L is still held");
    up(&mut h, Key::L);
    assert_eq!(h.state().ui.tool, Tool::Brush);
}

#[test]
fn letters_type_while_editing_text() {
    let mut h = harness(Tool::Type);
    let id = h.state_mut().run("type.create", json!({"x": 20, "y": 80, "text": "a", "size": 40})).unwrap()["layer"].as_u64().unwrap();
    h.state_mut().ui.text_edit = Some(crate::state::TextEdit {
        layer: id,
        caret: 1,
        anchor: 1,
        session: "spring-test".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    h.run_steps(2);
    h.event(Event::Key { key: Key::B, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.event(Event::Text("b".into()));
    h.run_steps(1);
    wait(&mut h, SPRING_HOLD_SECS + 0.1);
    up(&mut h, Key::B);
    assert_eq!(h.state().ui.tool, Tool::Type);
    assert_eq!(h.state().ui.spring, None);
    let st = h.state().session.active().unwrap();
    match &st.doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
        photocraft_doc::LayerContent::Text(t) => assert_eq!(t.text, "ab"),
        _ => panic!("not text"),
    }
}

/// A tool picked another way while a letter is held (here, as the toolbar would) stands: the release
/// doesn't take it back.
#[test]
fn a_tool_picked_while_a_letter_is_held_stands() {
    let mut h = harness(Tool::Brush);
    down(&mut h, Key::E);
    wait(&mut h, SPRING_HOLD_SECS + 0.1);
    h.state_mut().ui.tool = Tool::Lasso;
    up(&mut h, Key::E);
    assert_eq!(h.state().ui.tool, Tool::Lasso);
    assert_eq!(h.state().ui.spring, None);
}

#[test]
fn plain_functions_follow_the_rules() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.ui.tool = Tool::Eraser;
    // The current group's own key cycles; no spring.
    pressed(&mut app, 'E', Tool::Eraser, true, 0.0);
    assert_eq!(app.ui.spring, None);
    // A ⇧ switch clears any pending spring.
    pressed(&mut app, 'B', Tool::Eraser, true, 0.0);
    assert!(app.ui.spring.is_some());
    pressed(&mut app, 'L', Tool::Brush, false, 0.1);
    assert_eq!(app.ui.spring, None);
    // A canvas press marks a pending spring used; nothing to mark without one.
    note_canvas_press(&mut app);
    pressed(&mut app, 'B', Tool::Eraser, true, 0.0);
    note_canvas_press(&mut app);
    assert!(app.ui.spring.as_ref().is_some_and(|s| s.used && s.previous == Tool::Eraser));
    assert_eq!(key_of('\0'), None);
    assert!(is_tool_key(Key::Z) && !is_tool_key(Key::F1));
}
