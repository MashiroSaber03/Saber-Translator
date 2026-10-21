//! #306: ⌘↩ (Ctrl+Enter off the Mac) loads a path as a selection, as in Photoshop: the path
//! being drawn with the Pen, the path selected in the Paths panel, or the work path. The key is
//! rebindable in Edit › Keyboard Shortcuts, and with no path it says why instead of doing nothing.

use egui::{Event, Key, Modifiers, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;
use crate::vector_ui::PenPath;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(6);
    h
}

/// ⌘ on the Mac, Ctrl elsewhere, as egui-winit reports them.
fn command() -> Modifiers {
    if cfg!(target_os = "macos") {
        Modifiers { command: true, mac_cmd: true, ..Default::default() }
    } else {
        Modifiers { command: true, ctrl: true, ..Default::default() }
    }
}

fn press(h: &mut Harness<'_, PhotocraftApp>, key: Key, m: Modifiers) {
    h.event(Event::ModifiersChanged(m));
    h.event(Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: m });
    h.event(Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers: m });
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
}

fn rect_path(x: f64, y: f64, w: f64, h: f64) -> serde_json::Value {
    let k = |x: f64, y: f64| json!({"anchor": [x, y]});
    json!({"subpaths": [{"closed": true, "knots": [k(x, y), k(x + w, y), k(x + w, y + h), k(x, y + h)]}]})
}

/// The selection's bounds (x, y, width, height), if any.
fn selection(h: &Harness<'_, PhotocraftApp>) -> Option<[i64; 4]> {
    let r = h.state().session.active()?.doc.selection.as_ref()?.content_bounds();
    Some([r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0].map(i64::from))
}

#[test]
fn cmd_enter_with_the_pen_loads_the_work_path() {
    let mut h = harness();
    h.state_mut().run("path.set", json!({"name": "work", "path": rect_path(10.0, 10.0, 50.0, 40.0)})).unwrap();
    h.state_mut().ui.tool = Tool::Pen;
    h.run_steps(2);
    assert!(h.state().session.active().unwrap().doc.selection.is_none());
    press(&mut h, Key::Enter, command());
    assert!(h.state().session.active().unwrap().doc.selection.is_some(), "⌘↩ made a selection: {}", h.state().ui.status);
    assert_eq!(selection(&h), Some([10, 10, 50, 40]));
}

#[test]
fn cmd_enter_while_drawing_finishes_the_path_and_loads_it() {
    let mut h = harness();
    h.state_mut().ui.tool = Tool::Pen;
    let p = |x: f64, y: f64| [[x, y], [x, y], [x, y]];
    h.state_mut().ui.pen = Some(PenPath { knots: vec![p(20.0, 20.0), p(120.0, 20.0), p(120.0, 80.0)], ..Default::default() });
    h.run_steps(2);
    press(&mut h, Key::Enter, command());
    let app = h.state();
    assert!(app.ui.pen.is_none(), "the pen path was finished");
    assert!(app.session.active().unwrap().doc.work_path.is_some(), "into the work path");
    assert!(app.session.active().unwrap().doc.selection.is_some(), "and loaded as a selection: {}", app.ui.status);
    // Plain ↩ still only finishes a path.
    let mut h = harness();
    h.state_mut().ui.tool = Tool::Pen;
    h.state_mut().ui.pen = Some(PenPath { knots: vec![p(20.0, 20.0), p(120.0, 20.0), p(120.0, 80.0)], ..Default::default() });
    h.run_steps(2);
    press(&mut h, Key::Enter, Modifiers::NONE);
    assert!(h.state().session.active().unwrap().doc.work_path.is_some());
    assert!(h.state().session.active().unwrap().doc.selection.is_none());
}

#[test]
fn cmd_enter_loads_the_path_selected_in_the_paths_panel() {
    let mut h = harness();
    h.state_mut().run("path.set", json!({"name": "work", "path": rect_path(0.0, 0.0, 30.0, 30.0)})).unwrap();
    h.state_mut().run("path.set", json!({"name": "Path 1", "path": rect_path(100.0, 40.0, 60.0, 50.0)})).unwrap();
    // Any tool, with a path selected in the Paths panel (as in Photoshop).
    h.state_mut().ui.tool = Tool::Brush;
    h.state_mut().ui.selected_path = Some("Path 1".into());
    h.run_steps(2);
    press(&mut h, Key::Enter, command());
    assert_eq!(selection(&h), Some([100, 40, 60, 50]), "{}", h.state().ui.status);
    // A selected path that went away falls back to the work path.
    h.state_mut().run("path.delete", json!({"name": "Path 1"})).unwrap();
    h.state_mut().run("select.deselect", json!({})).unwrap();
    press(&mut h, Key::Enter, command());
    assert_eq!(selection(&h), Some([0, 0, 30, 30]));
}

#[test]
fn without_a_path_it_says_why_and_the_key_can_be_rebound() {
    let mut h = harness();
    h.state_mut().ui.tool = Tool::Pen;
    press(&mut h, Key::Enter, command());
    let app = h.state();
    assert!(app.session.active().unwrap().doc.selection.is_none());
    assert!(app.ui.status_error && app.ui.status.contains("No path"), "status: {}", app.ui.status);
    // It's a listed, rebindable shortcut.
    let cmd_enter = crate::shortcuts::parse("Cmd+Enter").unwrap();
    assert!(crate::shortcut_dispatch::bindings(app).iter().any(|(id, sc)| id == "path.toSelection" && *sc == cmd_enter));
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"path.toSelection": "Cmd+Shift+Y"}})).unwrap();
    h.state_mut().run("path.set", json!({"name": "work", "path": rect_path(10.0, 10.0, 20.0, 20.0)})).unwrap();
    press(&mut h, Key::Enter, command());
    assert!(h.state().session.active().unwrap().doc.selection.is_none(), "⌘↩ was moved");
    let mut m = command();
    m.shift = true;
    press(&mut h, Key::Y, m);
    assert_eq!(selection(&h), Some([10, 10, 20, 20]), "{}", h.state().ui.status);
}

fn panel(h: &mut Harness<'_, PhotocraftApp>, tab: usize) {
    let ctx = h.ctx.clone();
    let (req, _) = crate::control::ControlRequest::new(
        "ui.set",
        json!({"theme":"promedium", "dockTabs":{"layers":tab}, "dock":{"collapsed":["color","properties","history","navigator"]}}),
    );
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(4);
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: egui::Pos2, modifiers: Modifiers) {
    h.hover_at(at);
    h.event(Event::ModifiersChanged(modifiers));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers });
    h.run_steps(1);
    h.event(Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: false, modifiers });
    h.run_steps(1);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
}

#[test]
fn thumbnail_selection_ctrl_click_shape_keeps_active_layer_and_supports_undo() {
    let mut h = harness();
    let shape =
        h.state_mut().run("shape.create", json!({"kind":"rect","rect":[40,20,60,40],"fill":"#cc3333","name":"Badge"})).unwrap()["layer"].as_u64().unwrap();
    let active = h.state_mut().run("layer.new.layer", json!({"name":"Other"})).unwrap()["layer"].as_u64().unwrap();
    panel(&mut h, 0);
    let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == shape).unwrap();
    click(&mut h, row.name.unwrap().center(), command());
    assert!(selection(&h).is_none(), "Ctrl-click on a shape's name only selects the layer");
    h.state_mut().run("layer.select", json!({"layer": active})).unwrap();
    h.run_steps(2);
    click(&mut h, egui::pos2(row.row.left() + 46.0, row.row.center().y), command());
    assert_eq!(selection(&h), Some([40, 20, 60, 40]), "{}", h.state().ui.status);
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(photocraft_doc::LayerId(active)));
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert!(selection(&h).is_none());
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(selection(&h), Some([40, 20, 60, 40]));
}

#[test]
fn thumbnail_selection_ctrl_click_paths_targets_thumbnail_and_clicked_path() {
    let mut h = harness();
    h.state_mut().run("path.set", json!({"name":"work","path":rect_path(10.0,10.0,50.0,40.0)})).unwrap();
    h.state_mut().run("path.set", json!({"name":"Saved","path":rect_path(100.0,40.0,60.0,50.0)})).unwrap();
    h.state_mut().ui.selected_path = Some("Saved".into());
    panel(&mut h, 2);
    let row = h.get_by_label("Work Path").rect();
    click(&mut h, egui::pos2(row.left() + 70.0, row.center().y), command());
    assert!(selection(&h).is_none(), "Ctrl-click on the name only selects the path row");
    h.state_mut().ui.selected_path = Some("Saved".into());
    h.run_steps(2);
    click(&mut h, egui::pos2(row.left() + 22.0, row.center().y), command());
    assert_eq!(selection(&h), Some([10, 10, 50, 40]), "{}", h.state().ui.status);
    assert_eq!(h.state().ui.selected_path.as_deref(), Some("Saved"), "loading a thumbnail does not change the selected path");
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert!(selection(&h).is_none());
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(selection(&h), Some([10, 10, 50, 40]));
    h.state_mut().run("select.deselect", json!({})).unwrap();
    let saved = h.get_by_label("Saved").rect();
    click(&mut h, egui::pos2(saved.left() + 22.0, saved.center().y), command());
    assert_eq!(selection(&h), Some([100, 40, 60, 50]));
}

#[test]
fn thumbnail_selection_ctrl_click_paths_combines_with_shift_and_alt() {
    let mut h = harness();
    h.state_mut().run("path.set", json!({"name":"work","path":rect_path(10.0,10.0,50.0,40.0)})).unwrap();
    panel(&mut h, 2);
    let row = h.get_by_label("Work Path").rect();
    let at = egui::pos2(row.left() + 22.0, row.center().y);
    h.state_mut().run("select.rect", json!({"x":100,"y":40,"width":20,"height":20})).unwrap();
    click(&mut h, at, command() | Modifiers::SHIFT);
    let selected = h.state().session.active().unwrap().doc.selection.as_ref().unwrap();
    assert!(selected.sample_channel(20, 20, 0) > 0.99 && selected.sample_channel(110, 50, 0) > 0.99);
    h.state_mut().run("select.all", json!({})).unwrap();
    click(&mut h, at, command() | Modifiers::ALT);
    let selected = h.state().session.active().unwrap().doc.selection.as_ref().unwrap();
    assert!(selected.sample_channel(20, 20, 0) < 0.01 && selected.sample_channel(110, 50, 0) > 0.99);
    h.state_mut().run("select.rect", json!({"x":30,"y":0,"width":100,"height":100})).unwrap();
    click(&mut h, at, command() | Modifiers::SHIFT | Modifiers::ALT);
    assert_eq!(selection(&h), Some([30, 10, 30, 40]));
}

#[test]
fn thumbnail_selection_ctrl_click_layer_path_preserves_panel_target() {
    let mut h = harness();
    h.state_mut().run("shape.create", json!({"kind":"rect","rect":[40,20,60,40],"fill":"#cc3333","name":"Badge"})).unwrap();
    h.state_mut().run("path.set", json!({"name":"work","path":rect_path(0.0,0.0,10.0,10.0)})).unwrap();
    h.state_mut().ui.selected_path = Some("work".into());
    panel(&mut h, 2);
    let row = h.get_by_label("Badge Shape Path").rect();
    click(&mut h, egui::pos2(row.left() + 22.0, row.center().y), command());
    assert_eq!(selection(&h), Some([40, 20, 60, 40]));
    assert_eq!(h.state().ui.selected_path.as_deref(), Some("work"));
    assert!(!h.state().ui.mask_target && !h.state().ui.vector_mask_target);
    if let Ok(path) = std::env::var("PHOTOCRAFT_THUMBNAIL_SELECTION_SCREENSHOT") {
        h.render().unwrap().save(path).unwrap();
    }
}

/// #3915: Clicking a shape with the Path Selection tool selects its layer.
#[test]
fn clicking_shape_with_path_selection_selects_its_layer() {
    let mut h = harness();
    let rect_layer =
        h.state_mut().run("shape.create", json!({"kind": "rect", "rect": [20, 20, 80, 60], "fill": "#ff0000"})).unwrap()["layer"].as_u64().unwrap();
    // Add a new empty layer above so rect is not the active layer.
    let empty_layer = h.state_mut().run("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
    assert_eq!(h.state().session.active().unwrap().active_layer.unwrap().0, empty_layer);

    // Clicking inside the rectangle with Path Selection selects the rect layer.
    h.state_mut().ui.tool = Tool::PathSelection;
    crate::vector_ui::path_selection_finish(h.state_mut(), [50.0, 50.0], [50.0, 50.0]);
    assert_eq!(h.state().session.active().unwrap().active_layer.unwrap().0, rect_layer);

    // Clicking outside does not change active layer.
    crate::vector_ui::path_selection_finish(h.state_mut(), [150.0, 90.0], [150.0, 90.0]);
    assert_eq!(h.state().session.active().unwrap().active_layer.unwrap().0, rect_layer);
}
