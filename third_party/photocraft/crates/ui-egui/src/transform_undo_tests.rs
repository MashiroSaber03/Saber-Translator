//! Free Transform through the real canvas: Undo and Redo step through the box's own changes (each
//! drag one step) and never reach the document under the open box; and the
//! handles have a generous grab radius, so a press just off a corner scales instead of rotating.

use egui::{Key, Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

/// A 400 × 300 document with a 100 × 60 red rectangle on its own layer at (100, 100), the Move
/// tool active, at 100 %.
pub(crate) fn harness() -> Harness<'static, PhotocraftApp> {
    harness_tool(Tool::Move)
}

/// The same document with a chosen tool selected (e.g. Hand, #2302).
fn harness_tool(tool: Tool) -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.run("layer.new.layer", json!({})).unwrap();
    app.run("select.rect", json!({"x": 100, "y": 100, "width": 100, "height": 60})).unwrap();
    app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
    app.run("select.deselect", json!({})).unwrap();
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

fn screen(h: &Harness<'static, PhotocraftApp>, p: [f64; 2]) -> Pos2 {
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
    xf.to_screen(p[0] as f32, p[1] as f32)
}

/// Press at document point `from`, drag to `to` in a few steps and release.
pub(crate) fn drag(h: &mut Harness<'static, PhotocraftApp>, from: [f64; 2], to: [f64; 2]) {
    let (a, b) = (screen(h, from), screen(h, to));
    h.event(egui::Event::PointerMoved(a));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: a, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    for k in 1..=6 {
        h.event(egui::Event::PointerMoved(a + (b - a) * (k as f32 / 6.0)));
        h.run_steps(1);
    }
    h.event(egui::Event::PointerButton { pos: b, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

pub(crate) fn begin(h: &mut Harness<'static, PhotocraftApp>) {
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, "edit.freeTransform", json!({})).unwrap();
    h.run_steps(2);
}

fn quad(h: &Harness<'static, PhotocraftApp>) -> [[f64; 2]; 4] {
    h.state().ui.transform.as_ref().expect("transforming").quad
}

fn history(h: &Harness<'static, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().history.entries().len()
}

fn key(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers, k: Key) {
    h.key_press_modifiers(m, k);
    h.run_steps(2);
}

fn close(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> bool {
    a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-6)
}

#[test]
fn free_transform_starts_where_the_selected_pixels_were_moved() {
    let mut h = harness();
    h.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 100, "height": 60})).unwrap();
    drag(&mut h, [150.0, 130.0], [250.0, 180.0]);
    assert_eq!(photocraft_engine::float_cmds::floating(h.state().session.active().unwrap()).unwrap().offset, (100, 50));
    begin(&mut h);
    assert_eq!(quad(&h), [[200.0, 150.0], [300.0, 150.0], [300.0, 210.0], [200.0, 210.0]]);
    let st = h.state().session.active().unwrap();
    assert!(photocraft_engine::float_cmds::floating(st).is_none());
    assert_eq!(st.doc.selection.as_ref().unwrap().content_bounds(), photocraft_geom::Rect::new(200, 150, 300, 210));
}

/// Scale ×2, then down again, then ⌘Z: back to the ×2 box, not the fill before the transform.
#[test]
fn undo_steps_back_through_the_transform_not_the_document() {
    let mut h = harness();
    let steps = history(&h);
    begin(&mut h);
    let q0 = quad(&h);
    assert!(close(q0, [[100.0, 100.0], [200.0, 100.0], [200.0, 160.0], [100.0, 160.0]]), "{q0:?}");
    drag(&mut h, q0[2], [300.0, 220.0]);
    let q1 = quad(&h);
    assert!(!close(q1, q0), "the first drag scaled the box");
    drag(&mut h, q1[2], [150.0, 130.0]);
    let q2 = quad(&h);
    assert!(!close(q2, q1), "the second drag scaled it again");

    key(&mut h, Modifiers::COMMAND, Key::Z);
    assert!(close(quad(&h), q1), "⌘Z undoes the second drag: {:?}", quad(&h));
    key(&mut h, Modifiers::COMMAND, Key::Z);
    assert!(close(quad(&h), q0), "⌘Z again undoes the first");
    // Nothing left to undo in the session: the document under the box is left alone.
    key(&mut h, Modifiers::COMMAND, Key::Z);
    assert!(close(quad(&h), q0));
    assert_eq!(history(&h), steps, "the fill is still there");
    assert_eq!(width(&h), 100);

    key(&mut h, Modifiers::COMMAND | Modifiers::SHIFT, Key::Z);
    assert!(close(quad(&h), q1), "⇧⌘Z redoes the first drag");
    // A new change drops the redo; ↩ commits the session as one document step.
    key(&mut h, Modifiers::NONE, Key::ArrowRight);
    key(&mut h, Modifiers::COMMAND | Modifiers::SHIFT, Key::Z);
    assert!(close(quad(&h), q1.map(|p| [p[0] + 1.0, p[1]])), "nudged, nothing to redo: {:?}", quad(&h));
    key(&mut h, Modifiers::COMMAND, Key::Z);
    assert!(close(quad(&h), q1), "a nudge is a step too");
    key(&mut h, Modifiers::NONE, Key::Enter);
    assert!(h.state().ui.transform.is_none());
    assert_eq!(history(&h), steps + 1);
    assert_ne!(width(&h), 100);
    // Outside the transform, ⌘Z is the document's Undo again.
    key(&mut h, Modifiers::COMMAND, Key::Z);
    assert_eq!(width(&h), 100, "the committed transform is undone");
}

/// `engine.execute` over the control channel (and MCP, which goes through it) as a programmatic
/// caller: the result's `ok` flag.
fn execute(h: &mut Harness<'static, PhotocraftApp>, command: &str) -> bool {
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new("engine.execute", json!({"command": command}));
    let crate::control::Outcome::Done(reply) = crate::control::handle(h.state_mut(), &ctx, &req) else {
        panic!("{command}: expected an immediate reply");
    };
    h.run_steps(2);
    reply["ok"].as_bool().unwrap()
}

/// Undo and Redo sent over the control channel step through the open box like ⌘Z, never the
/// document under it, and Toggle Last State is refused until the transform ends.
#[test]
fn automation_undo_steps_through_the_transform() {
    let mut h = harness();
    let steps = history(&h);
    begin(&mut h);
    let q0 = quad(&h);
    drag(&mut h, q0[2], [300.0, 220.0]);
    let q1 = quad(&h);
    assert!(execute(&mut h, "edit.undo"));
    assert!(close(quad(&h), q0), "undid the drag: {:?}", quad(&h));
    assert!(execute(&mut h, "edit.undo"), "nothing left in the session: a no-op");
    assert!(close(quad(&h), q0));
    assert_eq!(history(&h), steps, "the fill is still there");
    assert_eq!(width(&h), 100);
    assert!(execute(&mut h, "edit.redo"));
    assert!(close(quad(&h), q1), "redid the drag");
    assert!(!execute(&mut h, "edit.toggleLastState"), "refused while transforming");
    assert!(close(quad(&h), q1));
    assert_eq!(history(&h), steps);
    // Once the transform ends, they reach the document again.
    key(&mut h, Modifiers::NONE, Key::Escape);
    assert!(h.state().ui.transform.is_none());
    assert!(execute(&mut h, "edit.undo"));
    assert_eq!(history(&h), steps - 1, "the fill is undone");
}

fn width(h: &Harness<'static, PhotocraftApp>) -> u32 {
    let st = h.state().session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds().width()
}

/// A press 10 pt diagonally off a corner (outside the old 8 pt radius) grabs the corner and
/// scales, whether the transform is already open or starts from the Move tool's controls.
#[test]
fn a_press_just_off_a_corner_scales_rather_than_rotates() {
    for via_controls in [false, true] {
        let mut h = harness();
        if via_controls {
            h.state_mut().ui.tool_options.move_show_transform = true;
            h.run_steps(1);
        } else {
            begin(&mut h);
        }
        drag(&mut h, [207.0, 167.0], [257.0, 197.0]);
        let q = quad(&h);
        assert!((q[0][1] - q[1][1]).abs() < 1e-6 && (q[1][0] - q[2][0]).abs() < 1e-6, "via controls {via_controls}: not rotated: {q:?}");
        assert!(q[2][0] > 230.0, "via controls {via_controls}: scaled: {q:?}");
        // Well outside the box still rotates.
        let before = quad(&h);
        drag(&mut h, [before[2][0] + 30.0, before[2][1] + 30.0], [before[2][0] - 10.0, before[2][1] + 60.0]);
        let r = quad(&h);
        assert!((r[0][1] - r[1][1]).abs() > 1.0, "via controls {via_controls}: rotated: {r:?}");
    }
}

/// With the Hand tool selected, a Free Transform drag resizes the box rather than panning the view
/// away: the open box owns the pointer (#2302).
#[test]
fn the_hand_tool_does_not_steal_a_free_transform_drag() {
    let mut h = harness_tool(Tool::Hand);
    begin(&mut h);
    let q0 = quad(&h);
    let center0 = h.state().ui.views[0].center;
    drag(&mut h, q0[2], [300.0, 220.0]);
    let q1 = quad(&h);
    assert!(!close(q1, q0), "the drag scaled the box, not the view: {q1:?}");
    assert_eq!(h.state().ui.views[0].center, center0, "the Hand did not pan");
    key(&mut h, Modifiers::NONE, Key::Enter);
    assert!(h.state().ui.transform.is_none(), "the transform committed");
    assert!(width(&h) > 150, "the committed resize applied: {}", width(&h));
}

/// With the Zoom tool selected, a Free Transform drag resizes the box rather than zooming the view:
/// the open box owns the pointer, as it does for the Hand (#4799).
#[test]
fn the_zoom_tool_does_not_steal_a_free_transform_drag() {
    let mut h = harness_tool(Tool::Zoom);
    begin(&mut h);
    let q0 = quad(&h);
    let (zoom0, center0) = (h.state().ui.views[0].zoom, h.state().ui.views[0].center);
    let inside = screen(&h, [150.0, 130.0]);
    h.event(egui::Event::PointerMoved(inside));
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton { pos: inside, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    h.run_steps(2);
    assert_eq!(h.state().ui.views[0].zoom, zoom0, "a click inside the box did not zoom");
    drag(&mut h, q0[2], [300.0, 220.0]);
    let q1 = quad(&h);
    assert!(!close(q1, q0), "the drag scaled the box, not the view: {q1:?}");
    assert_eq!(h.state().ui.views[0].zoom, zoom0, "the Zoom tool did not zoom");
    assert_eq!(h.state().ui.views[0].center, center0, "the Zoom tool did not move the view");
    key(&mut h, Modifiers::NONE, Key::Enter);
    assert!(h.state().ui.transform.is_none(), "the transform committed");
    assert!(width(&h) > 150, "the committed resize applied: {}", width(&h));
}

/// With the (freehand) Lasso selected, a Free Transform of a lasso selection follows the drag: the
/// Lasso reads its own pointer events, and dropped the moves and release of a drag it hadn't
/// started, so the box never changed (#2153).
#[test]
fn the_lasso_tool_does_not_swallow_a_free_transform_drag() {
    let mut h = harness_tool(Tool::Lasso);
    let outline = json!([[90, 90], [210, 90], [210, 170], [90, 170]]);
    h.state_mut().run("select.lasso", json!({"points": outline})).unwrap();
    begin(&mut h);
    let q0 = quad(&h);
    drag(&mut h, q0[2], [300.0, 220.0]);
    let q1 = quad(&h);
    assert!(!close(q1, q0), "the drag scaled the box: {q1:?}");
    key(&mut h, Modifiers::NONE, Key::Enter);
    assert!(h.state().ui.transform.is_none(), "the transform committed");
    assert!(width(&h) > 150, "the committed resize applied: {}", width(&h));
}

/// Holding Space (the temporary Hand, #249) still pans the view while a Free Transform box is open,
/// rather than resizing it (#2302 review): only the selected Hand yields to the box.
#[test]
fn holding_space_still_pans_while_transforming() {
    let mut h = harness();
    begin(&mut h);
    let q0 = quad(&h);
    let center0 = h.state().ui.views[0].center;
    let space = |h: &mut Harness<'static, PhotocraftApp>, down: bool| {
        h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: down, repeat: false, modifiers: Modifiers::NONE });
        h.run_steps(1);
    };
    space(&mut h, true);
    drag(&mut h, q0[2], [300.0, 220.0]);
    space(&mut h, false);
    h.run_steps(2);
    assert!(close(quad(&h), q0), "the space-drag did not touch the box: {:?}", quad(&h));
    assert_ne!(h.state().ui.views[0].center, center0, "the view panned instead");
}

#[test]
fn transform_preview_stays_visible_while_panning() {
    let mut h = harness();
    begin(&mut h);
    let texture = h.state().transform_preview.as_ref().unwrap().texture.pick(&h.ctx, 1.0, 1.0, false).0;
    let has_preview = |h: &Harness<'static, PhotocraftApp>| {
        h.output().shapes.iter().any(|clipped| matches!(&clipped.shape, egui::epaint::Shape::Mesh(mesh) if mesh.texture_id == texture))
    };
    assert!(has_preview(&h), "the transformed pixels are visible before panning");

    let center0 = h.state().ui.views[0].center;
    let from = screen(&h, [200.0, 160.0]);
    let to = from + vec2(50.0, 40.0);
    h.event(egui::Event::Key { key: Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(egui::Event::PointerMoved(from));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(egui::Event::PointerMoved(to));
    h.run_steps(1);

    assert_ne!(h.state().ui.views[0].center, center0, "the held Hand panned the view");
    assert!(has_preview(&h), "the transformed layer stays visible during the pan");
}

/// A transform started by dragging a Move-tool control: that first drag is undoable too.
#[test]
fn the_first_drag_from_the_move_tools_controls_undoes() {
    let mut h = harness();
    h.state_mut().ui.tool_options.move_show_transform = true;
    h.run_steps(1);
    drag(&mut h, [200.0, 160.0], [260.0, 196.0]);
    assert!(!close(quad(&h), [[100.0, 100.0], [200.0, 100.0], [200.0, 160.0], [100.0, 160.0]]));
    key(&mut h, Modifiers::COMMAND, Key::Z);
    assert!(close(quad(&h), [[100.0, 100.0], [200.0, 100.0], [200.0, 160.0], [100.0, 160.0]]), "{:?}", quad(&h));
}
