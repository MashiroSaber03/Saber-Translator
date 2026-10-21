//! #249 through the real canvas and key handling: Space repositions a marquee being drawn, Space
//! and ⌘Space / ⌥Space are temporary Hand / Zoom tools that give the tool back, D / X and the
//! fill keys are commands whose Keyboard Shortcuts overrides replace the defaults.

use egui::{Event, Key, Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use photocraft_geom::Rect;
use serde_json::json;

use super::*;
use crate::canvas::ViewXform;

fn harness(tool: Tool) -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.sync_views();
    app.ui.extras.rulers = false;
    app.ui.tool = tool;
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
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

/// The platform's modifiers for `Cmd` (⌘ on the Mac, Ctrl elsewhere), as egui-winit reports them.
fn platform(m: Modifiers) -> Modifiers {
    let mut out = Modifiers { alt: m.alt, shift: m.shift, ctrl: m.ctrl, ..Default::default() };
    if m.command {
        out.command = true;
        if cfg!(target_os = "macos") {
            out.mac_cmd = true;
        } else {
            out.ctrl = true;
        }
    }
    out
}

fn key(h: &mut Harness<'static, PhotocraftApp>, k: Key, pressed: bool, m: Modifiers) {
    let m = platform(m);
    h.event(Event::ModifiersChanged(m));
    h.event(Event::Key { key: k, physical_key: None, pressed, repeat: false, modifiers: m });
    h.run_steps(1);
}

/// A quick press and release in one frame (a tool letter held across frames springs, #2672).
fn tap(h: &mut Harness<'static, PhotocraftApp>, k: Key, m: Modifiers) {
    let m = platform(m);
    h.input_mut().events.extend([
        Event::ModifiersChanged(m),
        Event::Key { key: k, physical_key: None, pressed: true, repeat: false, modifiers: m },
        Event::Key { key: k, physical_key: None, pressed: false, repeat: false, modifiers: m },
    ]);
    h.run_steps(1);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(1);
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool, m: Modifiers) {
    h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: platform(m) });
    h.run_steps(1);
}

fn move_to(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32) {
    let p = screen(h, x, y);
    h.event(Event::PointerMoved(p));
    h.run_steps(2);
}

fn press_at(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32) {
    let p = screen(h, x, y);
    h.event(Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, Modifiers::NONE);
    move_to(h, x + 8.0, y + 6.0);
}

fn selection(h: &Harness<'static, PhotocraftApp>) -> Rect {
    h.state().session.active().unwrap().doc.selection.as_ref().map_or(Rect::EMPTY, |s| s.content_bounds())
}

fn click(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    let p = screen(h, x, y);
    h.event(Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, m);
    button(h, p, false, m);
    h.run_steps(1);
}

#[test]
fn space_repositions_a_marquee_being_drawn_then_sizing_resumes() {
    let mut h = harness(Tool::RectMarquee);
    press_at(&mut h, 50.0, 50.0);
    move_to(&mut h, 150.0, 110.0);
    key(&mut h, Key::Space, true, Modifiers::NONE);
    move_to(&mut h, 200.0, 150.0);
    move_to(&mut h, 230.0, 170.0);
    let center = h.state().ui.views[0].center;
    // Released with Space still down: the 100 × 60 rectangle, moved by (80, 60).
    release(&mut h, 230.0, 170.0, Modifiers::NONE);
    key(&mut h, Key::Space, false, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(selection(&h), Rect::new(130, 110, 230, 170), "offset, same size");
    assert_eq!(h.state().ui.views[0].center, center, "Space didn't pan");
    assert_eq!(h.state().ui.tool, Tool::RectMarquee);

    // Release Space mid-drag: sizing continues from the moved corner.
    let mut h = harness(Tool::EllipseMarquee);
    press_at(&mut h, 50.0, 50.0);
    move_to(&mut h, 150.0, 110.0);
    key(&mut h, Key::Space, true, Modifiers::NONE);
    move_to(&mut h, 170.0, 120.0);
    key(&mut h, Key::Space, false, Modifiers::NONE);
    move_to(&mut h, 200.0, 140.0);
    release(&mut h, 200.0, 140.0, Modifiers::NONE);
    h.run_steps(2);
    let r = selection(&h);
    assert!(r.x0.abs_diff(70) <= 1 && r.y0.abs_diff(60) <= 1 && r.x1.abs_diff(200) <= 1 && r.y1.abs_diff(140) <= 1, "{r:?}");
}

#[test]
fn space_moves_a_lasso_being_drawn() {
    let mut h = harness(Tool::Lasso);
    press_at(&mut h, 50.0, 50.0);
    move_to(&mut h, 120.0, 50.0);
    move_to(&mut h, 120.0, 100.0);
    key(&mut h, Key::Space, true, Modifiers::NONE);
    move_to(&mut h, 170.0, 130.0);
    release(&mut h, 170.0, 130.0, Modifiers::NONE);
    key(&mut h, Key::Space, false, Modifiers::NONE);
    h.run_steps(2);
    let r = selection(&h);
    assert!(r.x0.abs_diff(100) <= 1 && r.y0.abs_diff(80) <= 1 && r.x1.abs_diff(170) <= 1 && r.y1.abs_diff(130) <= 1, "{r:?}");
}

#[test]
fn space_is_a_temporary_hand_and_gives_the_tool_back() {
    let mut h = harness(Tool::Brush);
    key(&mut h, Key::Space, true, Modifiers::NONE);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::Hand));
    let before = h.state().ui.views[0].center;
    press_at(&mut h, 100.0, 100.0);
    move_to(&mut h, 160.0, 140.0);
    release(&mut h, 160.0, 140.0, Modifiers::NONE);
    key(&mut h, Key::Space, false, Modifiers::NONE);
    h.run_steps(2);
    let st = h.state();
    assert_ne!(st.ui.views[0].center, before, "the view panned");
    assert_eq!(st.ui.tool, Tool::Brush, "the brush is back");
    assert!(!st.session.active().unwrap().history.can_undo(), "nothing was painted");
}

#[test]
fn space_while_editing_type_is_not_the_hand_tool() {
    let mut h = harness(Tool::Type);
    let id = h.state_mut().run("type.create", json!({"text": "ئا", "size": 20, "x": 100, "y": 100})).unwrap()["layer"].as_u64().unwrap();
    h.state_mut().ui.text_edit = Some(crate::state::TextEdit {
        layer: id,
        caret: 2,
        anchor: 2,
        session: "t".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    key(&mut h, Key::Space, true, Modifiers::NONE);
    assert!(h.state().ui.text_edit.is_some(), "type session stays open");
    assert_eq!(held_tool(h.state(), &h.ctx), None, "Space types a space while editing");
    assert!(!reposition_held(h.state(), &h.ctx));
    key(&mut h, Key::Space, false, Modifiers::NONE);
}

#[test]
fn temporary_zoom_in_and_out_restore_the_tool() {
    let mut h = harness(Tool::RectMarquee);
    let z0 = h.state().ui.views[0].zoom;
    key(&mut h, Key::Space, true, Modifiers::COMMAND);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomIn));
    click(&mut h, 200.0, 150.0, Modifiers::COMMAND);
    key(&mut h, Key::Space, false, Modifiers::COMMAND);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
    let z1 = h.state().ui.views[0].zoom;
    assert!(z1 > z0, "{z0} -> {z1}");
    assert_eq!(h.state().ui.tool, Tool::RectMarquee);
    assert!(h.state().session.active().unwrap().doc.selection.is_none(), "the click didn't reach the marquee");

    key(&mut h, Key::Space, true, Modifiers::COMMAND | Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomOut));
    click(&mut h, 200.0, 150.0, Modifiers::COMMAND | Modifiers::ALT);
    key(&mut h, Key::Space, false, Modifiers::COMMAND | Modifiers::ALT);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
    assert!(h.state().ui.views[0].zoom < z1);
    assert_eq!(h.state().ui.tool, Tool::RectMarquee);
    assert_eq!(held_tool(h.state(), &h.ctx), None);
}

#[test]
fn temporary_zoom_drag_is_a_scrubby_zoom() {
    let mut h = harness(Tool::Brush);
    let z0 = h.state().ui.views[0].zoom;
    key(&mut h, Key::Space, true, Modifiers::COMMAND);
    press_at(&mut h, 200.0, 150.0);
    move_to(&mut h, 300.0, 150.0);
    // Releasing Space mid-drag keeps zooming until the button comes up.
    key(&mut h, Key::Space, false, Modifiers::COMMAND);
    move_to(&mut h, 320.0, 150.0);
    release(&mut h, 320.0, 150.0, Modifiers::COMMAND);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
    assert!(h.state().ui.views[0].zoom > z0 * 1.5);
    assert_eq!(h.state().ui.tool, Tool::Brush);
    assert!(!h.state().session.active().unwrap().history.can_undo(), "nothing was painted");
}

/// Changes only the held modifiers, as pressing or releasing ⌥ / ⌘ does with Space already down.
fn hold_mods(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers) {
    h.event(Event::ModifiersChanged(platform(m)));
    h.run_steps(1);
}

#[test]
fn space_then_alt_zooms_out_as_in_photoshop() {
    // Photoshop users hold Space first, then add ⌥ and click.
    let mut h = harness(Tool::Brush);
    h.state_mut().run("tools.setColors", json!({"foreground": "#ff0000", "background": "#00ff00"})).unwrap();
    let colors = fg_bg(&h);
    let z0 = h.state().ui.views[0].zoom;
    key(&mut h, Key::Space, true, Modifiers::NONE);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::Hand));
    hold_mods(&mut h, Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomOut));
    click(&mut h, 200.0, 150.0, Modifiers::ALT);
    let z1 = h.state().ui.views[0].zoom;
    assert!(z1 < z0, "{z0} -> {z1}");
    // Letting go of ⌥ with Space still down is the Hand again.
    hold_mods(&mut h, Modifiers::NONE);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::Hand));
    key(&mut h, Key::Space, false, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(held_tool(h.state(), &h.ctx), None);
    assert_eq!(h.state().ui.tool, Tool::Brush, "the brush is back");
    assert_eq!(fg_bg(&h), colors, "the ⌥-click didn't reach the brush's eyedropper");
    assert!(!h.state().session.active().unwrap().history.can_undo(), "nothing was painted");
}

#[test]
fn space_then_cmd_zooms_in_and_adding_alt_zooms_out() {
    let mut h = harness(Tool::RectMarquee);
    let z0 = h.state().ui.views[0].zoom;
    key(&mut h, Key::Space, true, Modifiers::NONE);
    hold_mods(&mut h, Modifiers::COMMAND);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomIn));
    click(&mut h, 200.0, 150.0, Modifiers::COMMAND);
    let z1 = h.state().ui.views[0].zoom;
    assert!(z1 > z0, "{z0} -> {z1}");
    hold_mods(&mut h, Modifiers::COMMAND | Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomOut));
    click(&mut h, 200.0, 150.0, Modifiers::COMMAND | Modifiers::ALT);
    assert!(h.state().ui.views[0].zoom < z1);
    hold_mods(&mut h, Modifiers::NONE);
    key(&mut h, Key::Space, false, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, Tool::RectMarquee);
    assert!(h.state().session.active().unwrap().doc.selection.is_none(), "no click reached the marquee");
}

#[test]
fn alt_turns_a_held_zoom_in_around_unless_the_key_includes_alt() {
    let mut h = harness(Tool::Brush);
    // With Zoom Out unbound, ⌘⌥Space still zooms out: ⌥ turns the held Zoom In around, as it
    // does the Zoom tool.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"tools.temporary.zoomOut": ""}})).unwrap();
    key(&mut h, Key::Space, true, Modifiers::COMMAND | Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomOut));
    key(&mut h, Key::Space, false, Modifiers::COMMAND | Modifiers::ALT);
    hold_mods(&mut h, Modifiers::NONE);
    // An ⌥ that is part of the Zoom In key doesn't.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"tools.temporary.zoomIn": "Alt+Z"}})).unwrap();
    let z0 = h.state().ui.views[0].zoom;
    key(&mut h, Key::Z, true, Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomIn));
    click(&mut h, 200.0, 150.0, Modifiers::ALT);
    key(&mut h, Key::Z, false, Modifiers::ALT);
    hold_mods(&mut h, Modifiers::NONE);
    assert!(h.state().ui.views[0].zoom > z0);
}

#[test]
fn rebound_temporary_zoom_uses_the_new_key_only() {
    let mut h = harness(Tool::RectMarquee);
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"tools.temporary.zoomIn": "Alt+Z"}})).unwrap();
    key(&mut h, Key::Space, true, Modifiers::COMMAND);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::Hand), "⌘Space is no longer zoom; Space still pans");
    key(&mut h, Key::Space, false, Modifiers::COMMAND);
    key(&mut h, Key::Z, true, Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::ZoomIn));
    key(&mut h, Key::Z, false, Modifiers::ALT);
    assert_eq!(held_tool(h.state(), &h.ctx), None);
    // Removed: no temporary hand at all.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"tools.temporary.hand": ""}})).unwrap();
    key(&mut h, Key::Space, true, Modifiers::NONE);
    assert_eq!(held_tool(h.state(), &h.ctx), None);
    assert!(!reposition_held(h.state(), &h.ctx));
}

fn fg_bg(h: &Harness<'static, PhotocraftApp>) -> ([f32; 4], [f32; 4]) {
    let t = &h.state().session.tools;
    (t.foreground, t.background)
}

#[test]
fn d_and_x_are_rebindable_commands() {
    let mut h = harness(Tool::Brush);
    h.state_mut().run("tools.setColors", json!({"foreground": "#ff0000", "background": "#00ff00"})).unwrap();
    tap(&mut h, Key::X, Modifiers::NONE);
    assert_eq!(fg_bg(&h), ([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]), "X swaps");
    tap(&mut h, Key::D, Modifiers::NONE);
    assert_eq!(fg_bg(&h), ([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]), "D resets");
    // Rebound: the old key no longer fires, the new one does.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"tools.swapColors": "Alt+Shift+X", "tools.defaultColors": ""}})).unwrap();
    h.state_mut().run("tools.setColors", json!({"foreground": "#ff0000", "background": "#00ff00"})).unwrap();
    tap(&mut h, Key::X, Modifiers::NONE);
    tap(&mut h, Key::D, Modifiers::NONE);
    assert_eq!(fg_bg(&h), ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]), "plain X and D do nothing now");
    tap(&mut h, Key::X, Modifiers::ALT | Modifiers::SHIFT);
    assert_eq!(fg_bg(&h), ([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]));
}

fn rgba(h: &Harness<'static, PhotocraftApp>, x: i32, y: i32) -> [f32; 4] {
    h.state().session.active().unwrap().doc.layers.last().unwrap().surface().unwrap().rgba(x, y)
}

#[test]
fn fill_keys_fill_and_follow_overrides() {
    let mut h = harness(Tool::Brush);
    h.state_mut().run("layer.new.layer", json!({})).unwrap();
    h.state_mut().run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    h.state_mut().run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    // ⌥⌫: foreground into the selection.
    tap(&mut h, Key::Backspace, Modifiers::ALT);
    assert_eq!(rgba(&h, 15, 15), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(rgba(&h, 50, 50)[3], 0.0);
    // ⇧⌘⌫ with no selection: background into the existing pixels only.
    h.state_mut().run("select.deselect", json!({})).unwrap();
    tap(&mut h, Key::Backspace, Modifiers::COMMAND | Modifiers::SHIFT);
    assert_eq!(rgba(&h, 15, 15), [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(rgba(&h, 50, 50)[3], 0.0, "transparency preserved");
    // ⇧⌥Delete (Windows' Delete key) does the same with the foreground.
    tap(&mut h, Key::Delete, Modifiers::ALT | Modifiers::SHIFT);
    assert_eq!(rgba(&h, 15, 15), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(rgba(&h, 50, 50)[3], 0.0);
    // ⌘⌫: background over the whole layer.
    tap(&mut h, Key::Backspace, Modifiers::COMMAND);
    assert_eq!(rgba(&h, 50, 50), [0.0, 0.0, 1.0, 1.0]);
    // Rebind Fill with Foreground to F9: ⌥⌫ no longer fills.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"edit.fillForeground": "F9"}})).unwrap();
    tap(&mut h, Key::Backspace, Modifiers::ALT);
    assert_eq!(rgba(&h, 50, 50), [0.0, 0.0, 1.0, 1.0], "the default no longer fires");
    tap(&mut h, Key::F9, Modifiers::NONE);
    assert_eq!(rgba(&h, 50, 50), [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn fill_key_on_a_locked_layer_reports_instead_of_crashing() {
    let mut h = harness(Tool::Brush);
    h.state_mut().run("layer.new.layer", json!({})).unwrap();
    h.state_mut().run("layer.lockLayers", json!({"pixels": true})).unwrap();
    tap(&mut h, Key::Backspace, Modifiers::ALT);
    let st = h.state();
    assert!(st.ui.status_error && st.ui.status.contains("locked"), "{}", st.ui.status);
    assert_eq!(rgba(&h, 5, 5)[3], 0.0);
}

#[test]
fn keyboard_shortcuts_dialog_lists_the_keys_under_tools() {
    let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let items = crate::prefs_ui::shortcut_items(&app);
    let find = |id: &str| items.iter().find(|i| i.0 == id).map(|i| (i.2.clone(), i.3.clone()));
    let tools = || vec!["Tools".to_string()];
    assert_eq!(find("tools.swapColors"), Some((tools(), Some("X".into()))));
    assert_eq!(find("tools.defaultColors"), Some((tools(), Some("D".into()))));
    assert_eq!(find("edit.fillForeground"), Some((tools(), Some("Alt+Backspace".into()))));
    assert_eq!(find("edit.fillBackgroundPreserve"), Some((tools(), Some("Cmd+Shift+Backspace".into()))));
    let temp = vec!["Tools".to_string(), "Temporary".to_string()];
    assert_eq!(find("tools.temporary.hand"), Some((temp.clone(), Some("Space".into()))));
    assert_eq!(find("tools.temporary.zoomOut"), Some((temp, Some("Alt+Space".into()))));
    assert_eq!(crate::prefs_ui::default_shortcut("tools.temporary.zoomIn").as_deref(), Some("Cmd+Space"));
    // One contiguous Tools section (the dialog prints a header where the top level changes).
    let first = items.iter().position(|i| i.2.first().map(String::as_str) == Some("Tools")).unwrap();
    assert!(items[first..].iter().all(|i| i.2.first().map(String::as_str) == Some("Tools")));
    // Temporary tools never dispatch as pressed commands, even when overridden.
    let mut app = app;
    app.run("edit.keyboardShortcuts", json!({"set": {"tools.temporary.hand": "F10"}})).unwrap();
    assert!(!crate::shortcut_dispatch::bindings(&app).iter().any(|(id, _)| is_temporary(id)));
}

#[test]
fn agents_reposition_a_marquee_with_space() {
    use crate::control::{ControlRequest, Outcome, handle};
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let ctx = egui::Context::default();
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    let mut call = |p: serde_json::Value| {
        let (req, _rx) = ControlRequest::new("ui.pointer", p);
        assert!(matches!(handle(&mut app, &ctx, &req), Outcome::Done(_)));
    };
    call(json!({"tool": "rectMarquee", "events": [{"kind": "down", "x": 20, "y": 20}, {"kind": "move", "x": 120, "y": 80}]}));
    call(json!({"space": true, "events": [{"kind": "move", "x": 170, "y": 110}]}));
    call(json!({"space": true, "events": [{"kind": "up", "x": 170, "y": 110}]}));
    let sel = app.session.active().unwrap().doc.selection.as_ref().unwrap().content_bounds();
    assert_eq!(sel, Rect::new(70, 50, 170, 110));
}

fn release(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    let p = screen(h, x, y);
    button(h, p, false, m);
}

/// A red square on a new layer for the ⌘ (Move) tests; Auto-Select, snapping and smart guides
/// off so the drag offsets are exact.
fn with_layer(h: &mut Harness<'static, PhotocraftApp>) -> photocraft_doc::LayerId {
    let app = h.state_mut();
    app.run("layer.new.layer", json!({})).unwrap();
    let id = app.session.active().unwrap().active_layer.unwrap();
    app.session
        .edit("paint", |doc, a| {
            doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(100, 100, 160, 160), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    app.ui.tool_options.move_auto_select = false;
    app.ui.extras.snap = false;
    app.ui.view.show.smart_guides = false;
    h.run_steps(2);
    id
}

fn bounds(h: &Harness<'static, PhotocraftApp>, id: photocraft_doc::LayerId) -> Rect {
    h.state().session.active().unwrap().doc.layer(id).unwrap().surface().unwrap().content_bounds()
}

/// A drag with `m` held throughout (the modifiers go down before the press, as on a keyboard).
fn drag_with(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers, x: f32, y: f32, x2: f32, y2: f32) {
    drag_raw(h, platform(m), x, y, x2, y2);
}

/// [`drag_with`] with the modifiers exactly as given, for one platform's layout on any host.
fn drag_raw(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers, x: f32, y: f32, x2: f32, y2: f32) {
    h.event(Event::ModifiersChanged(m));
    h.run_steps(1);
    let p = screen(h, x, y);
    h.event(Event::PointerMoved(p));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: m });
    h.run_steps(1);
    move_to(h, x + 8.0, y + 6.0);
    move_to(h, x2, y2);
    let q = screen(h, x2, y2);
    h.event(Event::PointerButton { pos: q, button: PointerButton::Primary, pressed: false, modifiers: m });
    h.run_steps(1);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
}

/// ⌘⌥ as a Mac sends it, and Ctrl+Alt as Windows and Linux send it (`command` is Ctrl there).
const MAC_CMD_ALT: Modifiers = Modifiers { alt: true, ctrl: false, shift: false, mac_cmd: true, command: true };
const CTRL_ALT: Modifiers = Modifiers { alt: true, ctrl: true, shift: false, mac_cmd: false, command: true };

/// ⌘ with the Brush is the Move tool: a ⌘-drag moves the layer and paints nothing; the Brush is
/// back when ⌘ comes up.
#[test]
fn cmd_drag_with_the_brush_is_the_move_tool() {
    let mut h = harness(Tool::Brush);
    let id = with_layer(&mut h);
    h.event(Event::ModifiersChanged(platform(Modifiers::COMMAND)));
    h.run_steps(1);
    assert_eq!(held_tool(h.state(), &h.ctx), Some(Temporary::Move));
    let steps = h.state().session.active().unwrap().history.past_len();
    drag_with(&mut h, Modifiers::COMMAND, 130.0, 130.0, 170.0, 150.0);
    assert_eq!(held_tool(h.state(), &h.ctx), None);
    let st = h.state().session.active().unwrap();
    assert_eq!(bounds(&h, id), Rect::new(140, 120, 200, 180), "the layer moved by (40, 20)");
    assert_eq!(st.history.undo_label(), Some("Move"));
    assert_eq!(st.history.past_len(), steps + 1, "one step: no brush stroke");
    assert_eq!(h.state().ui.tool, Tool::Brush, "the Brush is back");
}

/// ⌘ released mid-drag: the Move tool lasts until the button comes up (Photoshop), and the
/// Eraser's `Up` never fires.
#[test]
fn cmd_released_mid_drag_keeps_moving_until_the_button_comes_up() {
    let mut h = harness(Tool::Eraser);
    let id = with_layer(&mut h);
    h.event(Event::ModifiersChanged(platform(Modifiers::COMMAND)));
    h.run_steps(1);
    let p = screen(&h, 130.0, 130.0);
    h.event(Event::PointerMoved(p));
    h.run_steps(1);
    button(&mut h, p, true, Modifiers::COMMAND);
    move_to(&mut h, 150.0, 140.0);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    move_to(&mut h, 170.0, 150.0);
    release(&mut h, 170.0, 150.0, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(bounds(&h, id), Rect::new(140, 120, 200, 180));
    assert_eq!(h.state().session.active().unwrap().history.undo_label(), Some("Move"), "nothing erased");
}

/// ⌘⌥-drag duplicates the layer first and moves the copy, as the Move tool's ⌥-drag does. Off
/// the Mac the same keys are Ctrl+Alt, which the Brush keeps for its on-canvas resize
/// (`brush_resize`, Alt+right-drag there too), so Ctrl+Alt duplicates with the tools that have no
/// brush to resize, such as the Gradient.
#[test]
fn cmd_alt_drag_duplicates_the_layer_then_moves_the_copy() {
    for (tool, mods) in [(Tool::Brush, MAC_CMD_ALT), (Tool::Gradient, MAC_CMD_ALT), (Tool::Gradient, CTRL_ALT)] {
        let mut h = harness(tool);
        let orig = with_layer(&mut h);
        let n0 = h.state().session.active().unwrap().doc.layers.len();
        let h0 = h.state().session.active().unwrap().history.past_len();
        drag_raw(&mut h, mods, 130.0, 130.0, 170.0, 150.0);
        let st = h.state().session.active().unwrap();
        assert_eq!(st.doc.layers.len(), n0 + 1, "one copy ({tool:?}, {mods:?})");
        let copy = st.active_layer.unwrap();
        assert_ne!(copy, orig);
        assert_eq!(bounds(&h, orig), Rect::new(100, 100, 160, 160), "the original stays");
        assert_eq!(bounds(&h, copy), Rect::new(140, 120, 200, 180), "the copy moved");
        assert_eq!(st.history.past_len(), h0 + 1, "one undo step");
        assert_eq!(st.history.undo_label(), Some(crate::move_mods::DUPLICATE_MOVE_LABEL));
        assert_eq!(h.state().ui.tool, tool);
    }
}

/// Off the Mac, Ctrl+Alt-drag with the Brush resizes the brush, as before ⌘ became the Move
/// tool: no copy, no move, the layer untouched.
#[test]
fn ctrl_alt_drag_with_the_brush_resizes_it_off_the_mac() {
    let mut h = harness(Tool::Brush);
    let orig = with_layer(&mut h);
    let n0 = h.state().session.active().unwrap().doc.layers.len();
    let size0 = h.state().session.tools.brush.size;
    drag_raw(&mut h, CTRL_ALT, 130.0, 130.0, 170.0, 150.0);
    let st = h.state().session.active().unwrap();
    assert_eq!(st.doc.layers.len(), n0, "no copy");
    assert_eq!(st.active_layer, Some(orig));
    assert_eq!(bounds(&h, orig), Rect::new(100, 100, 160, 160), "the layer stays");
    assert_ne!(st.history.undo_label(), Some("Move"));
    assert_ne!(h.state().session.tools.brush.size, size0, "the brush was resized");
}

/// With a selection, ⌘-drag with the Brush moves the selected pixels (a floating piece), as the
/// Move tool does, not the whole layer.
#[test]
fn cmd_drag_with_the_brush_inside_the_selection_floats_the_selected_pixels() {
    let mut h = harness(Tool::Brush);
    let id = with_layer(&mut h);
    h.state_mut().run("select.rect", json!({"x": 110, "y": 110, "width": 40, "height": 40})).unwrap();
    h.run_steps(1);
    let n0 = h.state().session.active().unwrap().doc.layers.len();
    drag_with(&mut h, Modifiers::COMMAND, 130.0, 130.0, 170.0, 150.0);
    let st = h.state().session.active().unwrap();
    let f = photocraft_engine::float_cmds::floating(st).expect("a floating piece");
    assert_eq!((f.layer, f.offset), (id, (40, 20)));
    assert_eq!(st.doc.layers.len(), n0, "no layer was added");
    assert_eq!(bounds(&h, id), Rect::new(100, 100, 160, 160), "the layer itself stayed");
}

/// ⌘-click with the Brush picks the layer under the pointer, as a ⌘-click with the Move tool does
/// (Auto-Select off).
#[test]
fn cmd_click_with_the_brush_picks_the_layer_under_the_pointer() {
    let mut h = harness(Tool::Brush);
    let red = with_layer(&mut h);
    let bg = h.state().session.active().unwrap().doc.layers.first().unwrap().id;
    h.state_mut().run("layer.select", json!({"layer": bg.0})).unwrap();
    h.run_steps(1);
    h.event(Event::ModifiersChanged(platform(Modifiers::COMMAND)));
    h.run_steps(1);
    click(&mut h, 130.0, 130.0, Modifiers::COMMAND);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
    let st = h.state().session.active().unwrap();
    assert_eq!(st.active_layer, Some(red));
    assert_eq!(rgba(&h, 130, 130), [1.0, 0.0, 0.0, 1.0], "nothing painted");
}

/// Tools that keep ⌘ for themselves, and the selection tools (their ⌘ is handled per press in
/// `canvas`, #896), never become the Move tool this way.
#[test]
fn cmd_is_not_the_move_tool_for_selection_hand_zoom_pen_shape_or_type_tools() {
    for tool in [
        Tool::Hand,
        Tool::RotateView,
        Tool::Zoom,
        Tool::Pen,
        Tool::Rectangle,
        Tool::Type,
        Tool::VerticalType,
        Tool::Crop,
        Tool::PathSelection,
        Tool::DirectSelection,
        Tool::Move,
        Tool::RectMarquee,
        Tool::EllipseMarquee,
        Tool::Lasso,
        Tool::PolygonLasso,
        Tool::MagicWand,
    ] {
        assert!(!cmd_moves(tool), "{tool:?}");
        let mut h = harness(tool);
        h.event(Event::ModifiersChanged(platform(Modifiers::COMMAND)));
        h.run_steps(1);
        assert_eq!(held_tool(h.state(), &h.ctx), None, "{tool:?}");
    }
    for tool in [Tool::Brush, Tool::Eraser, Tool::Gradient, Tool::CloneStamp, Tool::Eyedropper, Tool::MagneticLasso, Tool::QuickSelection] {
        assert!(cmd_moves(tool), "{tool:?}");
    }
}

/// The params of the last brush stroke (`paint.stroke`, or the Pencil's `paint.pencil`).
fn last_stroke(app: &PhotocraftApp) -> serde_json::Value {
    let journal = &app.session.journal;
    journal.iter().rev().find(|(id, _)| id == "paint.stroke" || id == "paint.pencil").map(|(_, p)| p.clone()).unwrap_or_default()
}

/// The `erase` flag of the last brush stroke.
fn last_stroke_erases(h: &Harness<'static, PhotocraftApp>) -> Option<bool> {
    last_stroke(h.state())["erase"].as_bool()
}

/// `~` held with the Brush or the Pencil erases with that tool's own brush (#3439): Photoshop's
/// Clear mode for that stroke, so History names the tool. The tool stays, and the next stroke
/// without the key paints again.
#[test]
fn tilde_held_with_the_brush_or_pencil_erases_with_its_brush() {
    for (tool, label) in [(Tool::Brush, "Brush Tool"), (Tool::Pencil, "Pencil")] {
        let mut h = harness(tool);
        with_layer(&mut h);
        key(&mut h, Key::Backtick, true, Modifiers::NONE);
        drag_with(&mut h, Modifiers::NONE, 110.0, 130.0, 150.0, 130.0);
        key(&mut h, Key::Backtick, false, Modifiers::NONE);
        assert_eq!(last_stroke_erases(&h), Some(true), "{tool:?}");
        assert_eq!(rgba(&h, 130, 130)[3], 0.0, "{tool:?}: erased to transparency");
        assert_eq!(last_stroke(h.state())["mode"], json!("clear"), "{tool:?}");
        assert_eq!(h.state().session.active().unwrap().history.undo_label(), Some(label));
        assert_eq!(h.state().ui.tool, tool, "the tool stays");
        drag_with(&mut h, Modifiers::NONE, 110.0, 140.0, 150.0, 140.0);
        assert_eq!(last_stroke_erases(&h), Some(false), "{tool:?}: paints again");
    }
}

/// Other tools ignore `~`: the Mixer Brush and the retouching tools have no erasing stroke.
#[test]
fn tilde_is_not_an_eraser_for_other_tools() {
    let mut h = harness(Tool::Eraser);
    with_layer(&mut h);
    h.state_mut().ui.tool = Tool::CloneStamp;
    key(&mut h, Key::Backtick, true, Modifiers::NONE);
    assert!(!erase_held(h.state(), &h.ctx));
    key(&mut h, Key::Backtick, false, Modifiers::NONE);
}

/// The key is a Tools › Temporary binding: rebound, the new key erases and `~` no longer does.
#[test]
fn rebound_brush_eraser_uses_the_new_key_only() {
    let mut h = harness(Tool::Brush);
    with_layer(&mut h);
    let items = crate::prefs_ui::shortcut_items(h.state());
    let item = items.iter().find(|i| i.0 == "tools.temporary.erase").map(|i| (i.2.clone(), i.3.clone()));
    assert_eq!(item, Some((vec!["Tools".to_string(), "Temporary".to_string()], Some("`".into()))));
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"tools.temporary.erase": "F9"}})).unwrap();
    key(&mut h, Key::Backtick, true, Modifiers::NONE);
    drag_with(&mut h, Modifiers::NONE, 110.0, 130.0, 150.0, 130.0);
    key(&mut h, Key::Backtick, false, Modifiers::NONE);
    assert_eq!(last_stroke_erases(&h), Some(false), "` is no longer bound");
    key(&mut h, Key::F9, true, Modifiers::NONE);
    drag_with(&mut h, Modifiers::NONE, 110.0, 140.0, 150.0, 140.0);
    key(&mut h, Key::F9, false, Modifiers::NONE);
    assert_eq!(last_stroke_erases(&h), Some(true), "F9 erases");
}

/// The Mode dropdown's Clear (#3439): chosen there, every Pencil stroke erases with the Pencil
/// until another mode is chosen.
#[test]
fn clear_in_the_mode_dropdown_erases_until_another_mode_is_chosen() {
    use crate::canvas::{ToolEvent, tool_event};
    use egui_kittest::kittest::{NodeT, Queryable};
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
    app.ui.tool = Tool::Pencil;
    let mut h = Harness::builder().with_size(vec2(1600.0, 400.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::panels::options_bar(app, ui);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
    h.run_steps(4);
    let combo = |h: &Harness<'static, PhotocraftApp>, current: &str| {
        h.get_all_by_role(egui::accesskit::Role::ComboBox).find(|n| n.accesskit_node().value().as_deref() == Some(current)).unwrap().click();
    };
    combo(&h, "Normal");
    h.run_steps(3);
    h.get_by_label("Clear").click();
    h.run_steps(3);
    assert!(crate::paint_mouse::clears(h.state(), Tool::Pencil));
    let stroke = |h: &mut Harness<'static, PhotocraftApp>| {
        let (app, m) = (h.state_mut(), Modifiers::NONE);
        tool_event(app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: 100.0, y: 40.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: 100.0, y: 40.0 }, m);
    };
    stroke(&mut h);
    assert_eq!(last_stroke(h.state())["mode"], json!("clear"));
    assert_eq!(h.state().session.active().unwrap().history.undo_label(), Some("Pencil"));
    combo(&h, "Clear");
    h.run_steps(3);
    h.get_by_label("Normal").click();
    h.run_steps(3);
    assert!(!crate::paint_mouse::clears(h.state(), Tool::Pencil));
    stroke(&mut h);
    assert_eq!(last_stroke(h.state())["mode"], serde_json::Value::Null, "paints again");
    assert_eq!(last_stroke(h.state())["erase"], json!(false));
}

/// While `~` is held the Brush's Mode shows Clear, as in Photoshop.
#[test]
fn the_mode_shows_clear_while_tilde_is_held() {
    let mut h = harness(Tool::Brush);
    assert!(!crate::paint_mouse::clears(h.state(), Tool::Brush));
    key(&mut h, Key::Backtick, true, Modifiers::NONE);
    assert!(crate::paint_mouse::clears(h.state(), Tool::Brush));
    key(&mut h, Key::Backtick, false, Modifiers::NONE);
    h.run_steps(1);
    assert!(!crate::paint_mouse::clears(h.state(), Tool::Brush));
}

/// #3368 through the real key handling: holding Space (the temporary Hand) with a pending crop
/// keeps it pending, while V (another tool) commits it, as in Photoshop.
#[test]
fn a_held_key_keeps_a_pending_crop_and_another_tools_key_commits_it() {
    let mut h = harness(Tool::Crop);
    {
        let app = h.state_mut();
        app.ui.crop_rect = Some([100.0, 50.0, 300.0, 200.0]);
        app.crop.default_frame = false;
    }
    h.run_steps(2);
    assert!(crate::crop_ui::pending(h.state()));
    key(&mut h, Key::Space, true, Modifiers::NONE);
    h.run_steps(2);
    key(&mut h, Key::Space, false, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, Tool::Crop);
    assert_eq!(h.state().session.active().unwrap().doc.size, photocraft_doc::Size::new(400, 300));
    assert!(crate::crop_ui::pending(h.state()));
    tap(&mut h, Key::V, Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, Tool::Move);
    assert_eq!(h.state().session.active().unwrap().doc.size, photocraft_doc::Size::new(200, 150));
    assert_eq!(h.state().ui.crop_rect, None);
}
