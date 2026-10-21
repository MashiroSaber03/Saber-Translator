//! The canvas layer context menu (#307): right-click with the Move tool, or ⌘/Ctrl+right-click
//! with any tool, lists the layers with visible pixels under the pointer, topmost first, as
//! Photoshop does. Choosing one selects it.
//!
//! The hit test is `layer.pickAt` with `"list": true` (non-transparent pixels, text bounds,
//! layer masks, hidden layers and hidden groups), and choosing runs `layer.select`, so the menu is the same
//! two commands an agent would run. Painting tools keep their plain right-click (the Brush
//! Preset picker, #268); the ⌘/Ctrl variant reaches the layer list from them too.

use egui::{Color32, Context, Modifiers};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::Tool;

/// An open layer menu: where (screen points) and the layers under that point, topmost first.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerMenu {
    pub pos: [f32; 2],
    /// (layer id, name).
    pub layers: Vec<(u64, String)>,
}

/// Does a right-click with `tool` and `mods` open the layer menu (rather than the tool's own)?
pub fn is_gesture(tool: Tool, mods: Modifiers) -> bool {
    tool == Tool::Move || mods.command
}

/// The layers at document point (x, y), topmost first, including text between glyphs.
pub fn layers_at(app: &mut PhotocraftApp, x: f64, y: f64) -> Vec<(u64, String)> {
    if !(x.is_finite() && y.is_finite()) {
        return Vec::new();
    }
    let Ok(v) = app.session.execute("layer.pickAt", json!({"x": x, "y": y, "list": true})) else { return Vec::new() };
    v.get("layers")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|l| Some((l.get("layer")?.as_u64()?, l.get("name").and_then(Value::as_str).unwrap_or_default().to_string()))).collect())
        .unwrap_or_default()
}

/// Open the menu at screen point `at` for document point (x, y). False (and nothing opens) when
/// no layer has pixels there.
pub fn open(app: &mut PhotocraftApp, ctx: &Context, at: [f32; 2], x: f64, y: f64) -> bool {
    let layers = layers_at(app, x, y);
    if layers.is_empty() {
        app.ui.layer_menu = None;
        return false;
    }
    app.ui.brush_picker = None;
    app.ui.layer_menu = Some(LayerMenu { pos: at, layers });
    crate::canvas_popup::mark_open(ctx, menu_id());
    true
}

/// Choose a layer from the menu: select it (one `layer.select`) and close the menu.
pub fn choose(app: &mut PhotocraftApp, layer: u64) {
    app.ui.layer_menu = None;
    if let Err(e) = app.run("layer.select", json!({"layer": layer, "mode": "replace"})) {
        app.ui.status = e;
    }
}

fn menu_id() -> egui::Id {
    egui::Id::new("canvas-layer-menu")
}

/// Show the open menu. Escape or a click outside closes it.
pub fn show(app: &mut PhotocraftApp, ctx: &Context) {
    let Some(menu) = app.ui.layer_menu.clone() else { return };
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        app.ui.layer_menu = None;
        return;
    }
    let active = app.session.active().and_then(|st| st.active_layer).map(|l| l.0);
    let id = menu_id();
    let screen = ctx.content_rect();
    let size = ctx.memory(|m| m.area_rect(id)).map_or(egui::vec2(200.0, 24.0 * menu.layers.len() as f32 + 12.0), |r| r.size());
    let pos = egui::pos2(menu.pos[0].min(screen.right() - size.x).max(screen.left()), menu.pos[1].min(screen.bottom() - size.y).max(screen.top()));
    let mut chosen = None;
    let area = egui::Area::new(id).order(egui::Order::Foreground).fixed_pos(pos).show(ctx, |ui| {
        egui::Frame::menu(ui.style()).show(ui, |ui| {
            // As wide as the longest name (long names are cut short), like a menu.
            let font = egui::TextStyle::Button.resolve(ui.style());
            let widest = menu.layers.iter().map(|(_, n)| ui.painter().layout_no_wrap(n.clone(), font.clone(), Color32::WHITE).size().x).fold(0.0, f32::max);
            let w = (widest + 2.0 * ui.spacing().button_padding.x + 8.0).clamp(140.0, 360.0);
            ui.set_width(w);
            ui.spacing_mut().item_spacing.y = 0.0;
            // A long list scrolls instead of running off the screen.
            egui::ScrollArea::vertical().max_height((screen.height() - 24.0).max(48.0)).show(ui, |ui| {
                ui.set_width(w);
                for (layer, name) in &menu.layers {
                    let b = egui::Button::selectable(Some(*layer) == active, name.as_str()).truncate().min_size(egui::vec2(w, 22.0));
                    if ui.add(b).clicked() {
                        chosen = Some(*layer);
                    }
                }
            });
        });
    });
    if let Some(layer) = chosen {
        choose(app, layer);
        return;
    }
    if crate::canvas_popup::outside_pressed(ctx, id, area.response.rect) {
        app.ui.layer_menu = None;
    }
}

#[cfg(test)]
mod tests {
    use egui::{Modifiers, PointerButton, Pos2, vec2};
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
    use serde_json::json;

    use super::*;

    /// 200 × 200: a white Background, "Red" (a square in the middle), and "Hidden" (all over,
    /// hidden). Returns the app and the (background, red, hidden) ids.
    fn app() -> (PhotocraftApp, u64, u64, u64) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 200})).unwrap();
        let bg = app.session.active().unwrap().active_layer.unwrap().0;
        app.run("layer.new.layer", json!({"name": "Red"})).unwrap();
        app.run("select.rect", json!({"x": 50, "y": 50, "width": 100, "height": 100})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        let red = app.session.active().unwrap().active_layer.unwrap().0;
        app.run("layer.new.layer", json!({"name": "Hidden"})).unwrap();
        app.run("edit.fill", json!({"color": "#0000ff"})).unwrap();
        let hidden = app.session.active().unwrap().active_layer.unwrap().0;
        app.run("layer.setProps", json!({"layer": hidden, "visible": false})).unwrap();
        (app, bg, red, hidden)
    }

    fn names(app: &PhotocraftApp) -> Vec<String> {
        app.ui.layer_menu.as_ref().map(|m| m.layers.iter().map(|l| l.1.clone()).collect()).unwrap_or_default()
    }

    #[test]
    fn lists_visible_layers_with_pixels_topmost_first() {
        let (mut a, bg, red, _) = app();
        assert!(open(&mut a, &Context::default(), [10.0, 10.0], 100.0, 100.0));
        assert_eq!(a.ui.layer_menu.as_ref().unwrap().layers.iter().map(|l| l.0).collect::<Vec<_>>(), [red, bg]);
        assert!(open(&mut a, &Context::default(), [10.0, 10.0], 10.0, 10.0));
        assert_eq!(names(&a), ["Background"]);
        choose(&mut a, red);
        assert_eq!(a.session.active().unwrap().active_layer.unwrap().0, red);
        assert!(a.ui.layer_menu.is_none());
    }

    #[test]
    fn nothing_there_or_bad_input_opens_nothing() {
        let (mut a, _, _, _) = app();
        for (x, y) in [(-5.0, 10.0), (500.0, 500.0), (f64::NAN, 1.0), (1.0, f64::INFINITY), (1e300, -1e300)] {
            assert!(!open(&mut a, &Context::default(), [0.0, 0.0], x, y), "({x}, {y})");
            assert!(a.ui.layer_menu.is_none());
        }
        // A transparent document: nothing to list.
        let mut b = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert!(!open(&mut b, &Context::default(), [0.0, 0.0], 1.0, 1.0), "no document");
        b.run("file.new", json!({"width": 20, "height": 20, "background": "transparent"})).unwrap();
        assert!(!open(&mut b, &Context::default(), [0.0, 0.0], 1.0, 1.0));
        // Choosing a layer that no longer exists reports it and keeps the app running.
        choose(&mut a, 987_654);
        assert!(!a.ui.status.is_empty());
    }

    fn harness(a: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
        let mut a = a;
        a.sync_views();
        let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            },
            a,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
        h.run_steps(4);
        h
    }

    fn right_click(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, modifiers: Modifiers) {
        h.event(egui::Event::ModifiersChanged(modifiers));
        h.event(egui::Event::PointerMoved(p));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Secondary, pressed: true, modifiers });
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Secondary, pressed: false, modifiers });
        h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
        h.run_steps(2);
    }

    #[test]
    fn move_tool_right_click_lists_and_selects() {
        let (mut a, bg, _, _) = app();
        a.ui.tool = Tool::Move;
        let mut h = harness(a);
        let c = h.state().last_canvas_rect.center();
        right_click(&mut h, c, Modifiers::NONE);
        assert_eq!(names(h.state()), ["Red", "Background"]);
        let j0 = h.state().session.journal.len();
        h.get_by_label("Background").click();
        h.run_steps(3);
        assert_eq!(h.state().session.active().unwrap().active_layer.unwrap().0, bg);
        assert!(h.state().ui.layer_menu.is_none(), "choosing closes the menu");
        assert!(h.state().session.journal[j0..].iter().any(|(id, _)| id == "layer.select"));
        // Escape closes it without choosing.
        right_click(&mut h, c, Modifiers::NONE);
        assert!(h.state().ui.layer_menu.is_some());
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert!(h.state().ui.layer_menu.is_none());
        assert_eq!(h.state().session.active().unwrap().active_layer.unwrap().0, bg);
    }

    #[test]
    fn painting_tools_keep_the_brush_picker_and_command_right_click_lists_layers() {
        let (mut a, _, _, _) = app();
        a.ui.tool = Tool::Brush;
        let mut h = harness(a);
        let c = h.state().last_canvas_rect.center();
        right_click(&mut h, c, Modifiers::NONE);
        assert!(h.state().ui.brush_picker.is_some() && h.state().ui.layer_menu.is_none(), "plain right-click: the Brush Preset picker");
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        right_click(&mut h, c, Modifiers::COMMAND);
        assert!(h.state().ui.brush_picker.is_none(), "⌘/Ctrl+right-click never opens the picker");
        assert_eq!(names(h.state()), ["Red", "Background"]);
        assert!(!h.state().session.journal.iter().any(|(id, _)| id == "paint.stroke"));
    }

    #[test]
    fn erase_preference_does_not_take_the_command_right_click() {
        let (mut a, _, red, _) = app();
        a.run("prefs.set", json!({"path": "tools.rightClickWithPaintingTools", "value": "erase"})).unwrap();
        a.ui.tool = Tool::Brush;
        let mut h = harness(a);
        let c = h.state().last_canvas_rect.center();
        right_click(&mut h, c, Modifiers::COMMAND);
        assert_eq!(h.state().ui.layer_menu.as_ref().map(|m| m.layers[0].0), Some(red));
        assert!(!h.state().session.journal.iter().any(|(id, _)| id == "paint.stroke"), "nothing erased");
    }

    #[test]
    fn agents_open_it_with_a_right_pointer_press() {
        let (mut a, _, red, _) = app();
        let ctx = egui::Context::default();
        let events = json!([{"kind": "down", "x": 100, "y": 100}, {"kind": "up", "x": 100, "y": 100}]);
        let (req, _rx) = crate::control::ControlRequest::new("ui.pointer", json!({"tool": "move", "button": "right", "events": events}));
        let _ = crate::control::handle(&mut a, &ctx, &req);
        assert_eq!(a.ui.layer_menu.as_ref().map(|m| m.layers[0].0), Some(red));
        let (req, _rx) = crate::control::ControlRequest::new("ui.inspect", Value::Null);
        let crate::control::Outcome::Done(v) = crate::control::handle(&mut a, &ctx, &req) else { panic!("inspect") };
        assert_eq!(v["result"]["layerMenu"]["layers"][0][1], "Red");
    }

    /// #512: a `ui.pointer` right-click opens the layer menu (and, with a painting tool, the Brush
    /// Preset picker) at its document point on screen, where a mouse right-click there opens it.
    #[test]
    fn agents_open_it_at_the_pointed_point() {
        let (mut a, _, _, _) = app();
        a.ui.tool = Tool::Move;
        let mut h = harness(a);
        let ctx = h.ctx.clone();
        let pointer = |h: &mut Harness<'static, PhotocraftApp>, tool: &str, x: f64, y: f64| {
            let events = json!([{"kind": "down", "x": x, "y": y}, {"kind": "up", "x": x, "y": y}]);
            let (req, _rx) = crate::control::ControlRequest::new("ui.pointer", json!({"tool": tool, "button": "right", "events": events}));
            let _ = crate::control::handle(h.state_mut(), &ctx, &req);
            crate::canvas::ViewXform::active(h.state()).unwrap().to_screen(x as f32, y as f32)
        };
        let near = |a: Option<[f32; 2]>, b: Pos2| a.is_some_and(|a| (a[0] - b.x).abs() < 0.5 && (a[1] - b.y).abs() < 0.5);
        let mut opened = Vec::new();
        for (x, y, listed) in [(100.0, 100.0, vec!["Red", "Background"]), (20.0, 170.0, vec!["Background"])] {
            let at = pointer(&mut h, "move", x, y);
            let pos = h.state().ui.layer_menu.as_ref().map(|m| m.pos);
            assert!(near(pos, at), "({x}, {y}): menu at {pos:?}, point on screen {at:?}");
            assert_eq!(names(h.state()), listed);
            h.state_mut().ui.layer_menu = None;
            right_click(&mut h, at, Modifiers::NONE);
            assert!(near(h.state().ui.layer_menu.as_ref().map(|m| m.pos), at), "the mouse opens it there too");
            assert_eq!(names(h.state()), listed);
            h.key_press(egui::Key::Escape);
            h.run_steps(2);
            opened.push(at);
        }
        assert!(opened[0].distance(opened[1]) > 50.0, "{opened:?}");
        let at = pointer(&mut h, "brush", 20.0, 170.0);
        assert!(near(h.state().ui.brush_picker, at), "picker at {:?}, point on screen {at:?}", h.state().ui.brush_picker);
    }
}
