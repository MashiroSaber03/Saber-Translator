//! Mouse buttons on the canvas. Tools follow the left button. The right button never paints with
//! the tool: with a painting tool it opens the Brush Preset picker at the pointer (Photoshop), or,
//! with Preferences › Tools › Right-click with painting tools set to Erase, a right drag with the
//! Brush erases with the current brush (Krita, Paint). Each right stroke is one `paint.stroke`
//! with `"erase": true`, so one undo step, with the pen pressure and tilt of a normal stroke.

use egui::{PointerButton, Response};
use photocraft_engine::BrushSettings;
use photocraft_engine::prefs::RightClickPaint;

use crate::PhotocraftApp;
use crate::state::Tool;

/// The brush is a tool option, kept per tool like Photoshop's options bar: switching from the
/// Brush to the Eraser saves the session brush for the old tool and loads the new tool's, so each
/// keeps its own size, hardness, mode, opacity, dynamics and smoothing. A tool seen for the
/// first time takes what the brush already has, at Photoshop's 10 % smoothing. Other tools leave
/// the brush alone.
///
/// The foreground and background colours, the erase flag and the stroke seed belong to the
/// session rather than to the tool, so a swap never carries them between tools: picking a colour
/// or painting with the Eraser works with any tool's brush.
pub fn sync_tool_brush(app: &mut PhotocraftApp) {
    let tool = app.ui.tool;
    if !has_brush_picker(tool) || app.ui.brush_tool == tool {
        return;
    }
    // The live brush is the outgoing tool's, so save it there before the new one takes over: any
    // edit it made since it was last active is in it, so its copy never goes stale.
    let current = app.session.tools.brush.clone();
    save_brush(app, app.ui.brush_tool, current.clone());
    app.session.tools.brush = match app.ui.tool_brushes.iter().find(|(t, _)| *t == tool) {
        Some((_, b)) => session_fields(&current, b),
        // A tool seen for the first time takes what the brush has, at Photoshop's smoothing.
        None => BrushSettings { smoothing: first_smoothing(), ..current },
    };
    // The picked preset was the outgoing tool's brush: forget it, so Update Brush can't overwrite
    // that preset with this tool's brush.
    app.session.tools.current_preset = None;
    app.session.tools.brush_changed();
    app.ui.brush_tool = tool;
}

/// The brush `tool` had when it was last active.
fn save_brush(app: &mut PhotocraftApp, tool: Tool, brush: BrushSettings) {
    match app.ui.tool_brushes.iter_mut().find(|(t, _)| *t == tool) {
        Some(e) => e.1 = brush,
        None => app.ui.tool_brushes.push((tool, brush)),
    }
}

/// The session's own brush fields, which a tool swap never carries: the colours, the erase flag
/// and the stroke seed (a stroke's replay seed is the command's, not the tool's).
fn session_fields(current: &BrushSettings, brush: &BrushSettings) -> BrushSettings {
    BrushSettings { color: current.color, background: current.background, erase: current.erase, seed: current.seed, ..brush.clone() }
}

/// A tool's brush as it starts out (Reset Tool, #2825): the default brush at Photoshop's
/// smoothing, with the session's own fields of `current`.
pub(crate) fn fresh_brush(current: &BrushSettings) -> BrushSettings {
    session_fields(current, &BrushSettings { smoothing: first_smoothing(), ..BrushSettings::default() })
}

/// Photoshop's 10 % stroke smoothing, what a tool's brush starts at.
fn first_smoothing() -> photocraft_engine::paint::brush::Smoothing {
    photocraft_engine::paint::brush::Smoothing { amount: 0.1, ..Default::default() }
}

/// Does a stroke with `tool` starting now paint in Photoshop's Clear mode, erasing with its own
/// brush (#3439)? Clear chosen in its Mode dropdown, or `~` held (`hold_keys::erase_held`).
pub fn clears(app: &PhotocraftApp, tool: Tool) -> bool {
    let chosen = match tool {
        Tool::Brush => app.ui.tool_options.brush_clear,
        Tool::Pencil => app.ui.tool_options.pencil_clear,
        _ => false,
    };
    crate::hold_keys::erases_while_held(tool) && (chosen || app.held_erase)
}

/// Which tool events this frame's canvas response produces.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Buttons {
    pub started: bool,
    pub dragged: bool,
    pub stopped: bool,
    pub clicked: bool,
}

/// Tools whose right-click opens the Brush Preset picker (those with the options-bar brush chip).
pub fn has_brush_picker(tool: Tool) -> bool {
    tool.is_brushlike() || tool == Tool::QuickSelection
}

/// Does a right-button drag with `tool` erase (rather than open the picker)?
pub fn right_erases(app: &PhotocraftApp, tool: Tool) -> bool {
    matches!(tool, Tool::Brush | Tool::Eraser) && app.session.prefs().tools.right_click_with_painting_tools == RightClickPaint::Erase
}

/// Route the canvas response's buttons: the left one drives the tool; the right one resizes the
/// brush with Alt held (#297), erases (Erase preference) or opens the Brush Preset picker. Arms
/// `secondary_erase` or `brush_resize_armed` for this frame's `Down`.
pub fn canvas_buttons(app: &mut PhotocraftApp, response: &Response, tool: Tool) -> Buttons {
    let (mods, right_down) = response.ctx.input(|i| (i.modifiers, i.pointer.secondary_down()));
    // Alt+right-drag resizes the brush (brush_resize.rs); its events reach `tool_event` like a
    // left drag's, and nothing paints. A resize whose release was missed ends here.
    crate::brush_resize::release_stale(app, right_down || response.drag_stopped_by(PointerButton::Secondary));
    let resize_start = crate::brush_resize::applies(tool)
        && app.drag.is_none()
        && crate::brush_resize::is_right_gesture(crate::workspace_ui::sticky_mods(app, mods))
        && response.drag_started_by(PointerButton::Secondary);
    let resizing = app.brush_resize.is_some_and(|r| r.secondary);
    app.brush_resize_armed = resize_start;
    // ⌘/Ctrl+right-click lists the layers under the pointer instead (layer_pick_ui.rs, #307).
    let layer_menu = crate::layer_pick_ui::is_gesture(tool, mods);
    let erase = right_erases(app, tool) && !resize_start && !resizing && !layer_menu;
    let right_stroke = erase && app.drag.is_some();
    let right_start = erase && response.drag_started_by(PointerButton::Secondary);
    let right_click = response.secondary_clicked();
    if right_click
        && !erase
        && !layer_menu
        && has_brush_picker(tool)
        && let Some(p) = response.interact_pointer_pos()
    {
        crate::brush_picker::open(&mut app.ui, &response.ctx, [p.x, p.y]);
    }
    let erase_click = erase && right_click;
    app.secondary_erase = right_start || erase_click;
    let right_drag = right_stroke || resizing;
    Buttons {
        started: response.drag_started_by(PointerButton::Primary) || right_start || resize_start,
        dragged: response.dragged_by(PointerButton::Primary) || (right_drag && response.dragged_by(PointerButton::Secondary)),
        stopped: response.drag_stopped_by(PointerButton::Primary) || (right_drag && response.drag_stopped_by(PointerButton::Secondary)),
        clicked: response.clicked() || erase_click,
    }
}

/// `ui.pointer` with `"button": "secondary"`: true when its events should reach the tool (an
/// Alt+right-drag brush resize, `brush_resize_armed` for its `Down`; or an erasing right stroke,
/// `secondary_erase` armed for its `Down`). Otherwise a right-click with a painting tool opens
/// the Brush Preset picker at screen point `at`, and nothing paints.
pub fn pointer_secondary(app: &mut PhotocraftApp, ctx: &egui::Context, down: bool, mods: egui::Modifiers, at: [f32; 2]) -> bool {
    let tool = app.ui.tool;
    if crate::brush_resize::applies(tool) && (crate::brush_resize::is_right_gesture(mods) || app.brush_resize.is_some_and(|r| r.secondary)) {
        app.brush_resize_armed = down && app.drag.is_none();
        return true;
    }
    if right_erases(app, tool) {
        app.secondary_erase = down;
        return true;
    }
    if down && has_brush_picker(tool) {
        crate::brush_picker::open(&mut app.ui, ctx, at);
    }
    false
}

/// The Brush Preset picker a right-click or the options-bar brush chip opened, at the pointer. It
/// edits the session brush like the Brushes panel; a press outside, Escape, Enter or a
/// double-click on a preset closes it.
pub fn show_picker(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some([x, y]) = app.ui.brush_picker else { return };
    if !has_brush_picker(app.ui.tool) {
        app.ui.brush_picker = None;
        return;
    }
    // Escape and Enter close it once this frame's edits are in (a typed size applies), unless they
    // end a rename in its rename bar.
    let key_close = app.ui.brush_picker_list.renaming.is_none() && ctx.input(|i| i.key_pressed(egui::Key::Escape) || i.key_pressed(egui::Key::Enter));
    let screen = ctx.content_rect();
    // The picker's content size: the one its grip was dragged to, or the one it last closed with
    // (kept in egui's own memory, which outlives the app's), or the default. The frame's chrome
    // sits around it, and together they keep the whole picker on screen.
    let id = crate::brush_picker::picker_id();
    let size_id = id.with("content-size");
    let content = app.ui.brush_picker_size.or_else(|| ctx.data_mut(|d| d.get_persisted::<[f32; 2]>(size_id))).map_or_else(
        || egui::vec2(crate::brush_picker::DEFAULT_SIZE[0], crate::brush_picker::DEFAULT_SIZE[1]),
        |s| crate::brush_picker::clamp_size(s, screen.size()),
    );
    // Remember it for the picker's next open, and — egui's memory being saved — the next run.
    ctx.data_mut(|d| d.insert_persisted(size_id, [content.x, content.y]));
    let style = ctx.global_style();
    let size = content + style.spacing.menu_margin.sum() + egui::Vec2::splat(2.0 * style.visuals.window_stroke().width);
    let pos = egui::pos2(x.min(screen.right() - size.x).max(screen.left()), y.min(screen.bottom() - size.y).max(screen.top()));
    let area = egui::Area::new(id).order(egui::Order::Foreground).fixed_pos(pos).show(ctx, |ui| {
        let before = app.session.tools.brush.clone();
        let mut b = before.clone();
        let list = &mut app.ui.brush_picker_list;
        let frame = egui::Frame::popup(ui.style())
            .show(ui, |ui| crate::brush_picker::body(ui, &mut b, &app.session.tools.presets, app.session.tools.current_preset.as_deref(), list, content));
        let picks = frame.inner;
        // The corner grip resizes the picker, and the size is remembered across opens.
        if let Some(new) = crate::brush_picker::resize_grip(ui, frame.response.rect, content) {
            app.ui.brush_picker_size = Some([new.x, new.y]);
        }
        crate::brush_panel::commit_gesture(app, ui.ctx(), &before, &b);
        let closes = picks.iter().any(crate::brush_picker::Pick::closes);
        crate::brush_picker::apply(app, ui.ctx(), picks);
        closes
    });
    // Not a press on a menu the picker opened (the gear, a preset's context menu), nor on the
    // options-bar chip, whose click toggles the picker. A menu floats over the picker in the
    // foreground order, so the press is told apart by the layer it lands on, not by any popup
    // just being open: a press elsewhere closes the picker in the same click that closes the menu.
    let press = ctx.input(|i| i.pointer.any_pressed().then(|| i.pointer.interact_pos()).flatten());
    let on_menu = press.is_some_and(|p| ctx.layer_id_at(p).is_some_and(|l| l.order == egui::Order::Foreground));
    let outside = crate::canvas_popup::outside_pressed(ctx, id, area.response.rect) && press.is_some_and(|p| !crate::brush_picker::on_chip(ctx, p)) && !on_menu;
    if outside || key_close || area.inner {
        crate::brush_picker::close(&mut app.ui);
    }
}

#[cfg(test)]
mod tests {
    use egui::{Modifiers, PointerButton, Pos2, vec2};
    use egui_kittest::Harness;
    use serde_json::json;

    use super::*;
    use crate::canvas::{ToolEvent, tool_event};

    fn harness(prefs: Option<&str>) -> Harness<'static, PhotocraftApp> {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("tools.setColors", json!({"foreground": "#0000ff"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 20, "hardness": 1.0}})).unwrap();
        if let Some(v) = prefs {
            app.run("prefs.set", json!({"path": "tools.rightClickWithPaintingTools", "value": v})).unwrap();
        }
        app.ui.tool = Tool::Brush;
        app.sync_views();
        let mut h = Harness::builder().with_size(vec2(1200.0, 800.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(4);
        h
    }

    fn press(h: &mut Harness<'static, PhotocraftApp>, pos: Pos2, button: PointerButton, pressed: bool) {
        h.event(egui::Event::PointerButton { pos, button, pressed, modifiers: Modifiers::NONE });
        h.run_steps(2);
    }

    /// Drag across the canvas centre with `button`; returns the document points it covered.
    fn drag(h: &mut Harness<'static, PhotocraftApp>, button: PointerButton) -> (Pos2, Pos2) {
        let c = h.state().last_canvas_rect.center();
        let (a, b) = (c - vec2(80.0, 0.0), c + vec2(80.0, 0.0));
        h.event(egui::Event::PointerMoved(a));
        h.run_steps(1);
        press(h, a, button, true);
        for i in 1..=8 {
            h.event(egui::Event::PointerMoved(a + (b - a) * (i as f32 / 8.0)));
            h.run_steps(1);
        }
        press(h, b, button, false);
        (a, b)
    }

    fn alpha_at(h: &Harness<'static, PhotocraftApp>, screen: Pos2) -> f32 {
        let app = h.state();
        let v = app.ui.views[0].clone();
        let r = app.last_canvas_rect;
        let d = (screen - r.center()) / v.zoom;
        let (x, y) = ((d.x + v.center[0]) as i32, (d.y + v.center[1]) as i32);
        let st = app.session.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)[3]
    }

    fn strokes(h: &Harness<'static, PhotocraftApp>) -> Vec<serde_json::Value> {
        h.state().session.journal.iter().filter(|(id, _)| id == "paint.stroke").map(|(_, p)| p.clone()).collect()
    }

    #[test]
    fn left_drag_paints() {
        let mut h = harness(None);
        let (a, b) = drag(&mut h, PointerButton::Primary);
        assert_eq!(strokes(&h).len(), 1);
        assert!(alpha_at(&h, a + (b - a) * 0.5) > 0.9);
    }

    #[test]
    fn shift_click_connects_to_the_previous_brush_stroke() {
        use egui::{Event, Modifiers};

        let mut h = harness(None);
        let (_, end) = drag(&mut h, PointerButton::Primary);
        let previous = h
            .state()
            .session
            .journal
            .iter()
            .rev()
            .find(|(id, _)| id == "paint.stroke")
            .map(|(_, p)| p["points"].as_array().unwrap().last().unwrap().clone())
            .unwrap();
        let target = end + vec2(120.0, 80.0);
        h.event(Event::PointerMoved(target));
        h.run_steps(1);
        for pressed in [true, false] {
            h.event(Event::PointerButton { pos: target, button: PointerButton::Primary, pressed, modifiers: Modifiers::SHIFT });
            h.run_steps(2);
        }

        let strokes = strokes(&h);
        assert_eq!(strokes.len(), 2, "a Shift-click commits one connected stroke");
        let points = strokes[1]["points"].as_array().unwrap();
        // The previous endpoint, then the clicked one. While the button is held, smoothing that is
        // still catching up repeats the clicked point with later times.
        assert_eq!(points[0][0], previous[0]);
        assert_eq!(points[0][1], previous[1]);
        let clicked = &points[1];
        assert!(points[1..].iter().all(|p| p[0] == clicked[0] && p[1] == clicked[1]), "{points:?}");
        assert!(alpha_at(&h, end + vec2(60.0, 40.0)) > 0.9, "the segment between the strokes is painted");
    }

    #[test]
    fn right_click_opens_the_brush_picker_and_never_paints() {
        let mut h = harness(None);
        let undo = h.state().session.active().unwrap().history.past_len();
        let c = h.state().last_canvas_rect.center();
        h.event(egui::Event::PointerMoved(c));
        h.run_steps(1);
        press(&mut h, c, PointerButton::Secondary, true);
        press(&mut h, c, PointerButton::Secondary, false);
        assert_eq!(h.state().ui.brush_picker, Some([c.x, c.y]), "opened at the pointer");
        assert!(strokes(&h).is_empty());
        // The picker shows over the canvas at the pointer, moved up just enough to stay on screen.
        h.run_steps(2);
        let shown = h.ctx.memory(|m| m.area_rect(egui::Id::new("canvas-brush-picker"))).expect("picker shown");
        let screen = h.ctx.content_rect();
        assert!((shown.min.x - c.x).abs() < 1.0 && shown.min.y <= c.y + 1.0, "{shown:?}");
        assert!(screen.contains_rect(shown.shrink(0.5)) && shown.width() > 250.0 && shown.height() > 150.0, "{shown:?}");
        // It is the real preset library, not a fixed set of round tips (#258).
        let names: Vec<String> = h.state().session.tools.presets.iter().take(3).map(|p| p.name.clone()).collect();
        {
            use egui_kittest::kittest::Queryable;
            for n in &names {
                assert!(h.query_by_label(n).is_some(), "{n} listed");
            }
        }
        // A right drag doesn't paint either.
        drag(&mut h, PointerButton::Secondary);
        assert!(strokes(&h).is_empty(), "right-drag never paints with the brush picker preference");
        assert_eq!(h.state().session.active().unwrap().history.past_len(), undo, "no edit");
        // Escape closes it.
        h.state_mut().ui.brush_picker = Some([c.x, c.y]);
        h.run_steps(1);
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert_eq!(h.state().ui.brush_picker, None);
    }

    #[test]
    fn click_outside_closes_the_picker() {
        let mut h = harness(None);
        let c = h.state().last_canvas_rect.center();
        h.state_mut().ui.brush_picker = Some([c.x, c.y]);
        h.run_steps(2);
        let far = c - vec2(300.0, 200.0);
        h.event(egui::Event::PointerMoved(far));
        press(&mut h, far, PointerButton::Primary, true);
        press(&mut h, far, PointerButton::Primary, false);
        assert_eq!(h.state().ui.brush_picker, None);
    }

    /// #1031: the right-click picker picks brushes, not just the size. A click picks a preset and
    /// keeps the picker open, a double-click picks one and closes it, and Enter applies a typed
    /// size before it closes the picker.
    #[test]
    fn picker_click_picks_a_preset_and_double_click_closes_it() {
        use egui_kittest::kittest::Queryable;
        let mut h = harness(None);
        let c = h.state().last_canvas_rect.center();
        h.event(egui::Event::PointerMoved(c));
        h.run_steps(1);
        press(&mut h, c, PointerButton::Secondary, true);
        press(&mut h, c, PointerButton::Secondary, false);
        h.run_steps(2);
        let names: Vec<String> = h.state().session.tools.presets.iter().skip(1).take(2).map(|p| p.name.clone()).collect();
        let picks = |h: &Harness<'static, PhotocraftApp>, name: &str| {
            h.state().session.journal.iter().filter(|(id, p)| id == "tools.setBrush" && *p == json!({ "preset": name })).count()
        };
        h.get_by_label(&names[0]).click();
        h.run_steps(3);
        assert_eq!(picks(&h, &names[0]), 1);
        assert!(h.state().ui.brush_picker.is_some(), "a click keeps the picker open");
        let at = h.get_by_label(&names[1]).rect().center();
        h.event(egui::Event::PointerMoved(at));
        h.run_steps(1);
        // Two clicks 2 frames apart; well after the first click, so not a triple click.
        h.run_steps(40);
        for pressed in [true, false, true, false] {
            h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
            h.step();
        }
        h.run_steps(2);
        assert_eq!(picks(&h, &names[1]), 1, "picked once: the second click only closes");
        assert_eq!(h.state().ui.brush_picker, None, "a double-click closes the picker");
        assert!(strokes(&h).is_empty(), "nothing painted under the picker");
        // Enter applies the typed size, then closes the picker.
        h.state_mut().ui.brush_picker = Some([c.x, c.y]);
        h.run_steps(2);
        h.get_all_by_role(egui::accesskit::Role::SpinButton).next().expect("the Size field").click();
        h.run_steps(1);
        h.event(egui::Event::Text("30*2".into()));
        h.run_steps(1);
        h.key_press(egui::Key::Enter);
        h.run_steps(2);
        assert_eq!((h.state().session.tools.brush.size, h.state().ui.brush_picker), (60.0, None));
    }

    /// A sampled tip has no hardness: the picker shows only its size (Photoshop).
    #[test]
    fn picker_shows_hardness_only_for_round_tips() {
        use egui_kittest::kittest::Queryable;
        let mut h = harness(None);
        let c = h.state().last_canvas_rect.center();
        h.state_mut().ui.brush_picker = Some([c.x, c.y]);
        h.run_steps(2);
        assert!(h.query_by_label("Hardness").is_some());
        let sampled = h.state().session.tools.presets.iter().find(|p| p.brush.tip != photocraft_engine::paint::TipShape::Round).map(|p| p.name.clone());
        h.state_mut().run("tools.setBrush", json!({ "preset": sampled.expect("a sampled preset") })).unwrap();
        h.run_steps(2);
        assert!(h.query_by_label("Hardness").is_none());
        assert!(h.state().ui.brush_picker.is_some());
    }

    /// The gear menu's New Brush Preset… opens the New Brush Preset dialog (#4490), which saves
    /// the preset under the typed name on OK.
    #[test]
    fn picker_gear_menu_saves_and_names_a_new_preset() {
        use egui_kittest::kittest::Queryable;
        let mut h = harness(None);
        let c = h.state().last_canvas_rect.center();
        h.state_mut().ui.brush_picker = Some([c.x, c.y]);
        h.run_steps(2);
        h.get_by_label("Brush Preset Options").click();
        h.run_steps(2);
        let before = h.state().session.tools.presets.len();
        h.get_by_label("New Brush Preset…").click();
        h.run_steps(3);
        let dialog = h.state().ui.dialogs.iter().find(|d| crate::brush_preset_dialog::owns(&d.fields)).map(|d| d.id).expect("the dialog asks for a name");
        assert_eq!(h.state().session.tools.presets.len(), before, "nothing is saved before OK");
        assert!(h.state().ui.brush_picker.is_none(), "the dialog takes the picker's place");
        // This harness draws the canvas only: OK as automation does (the dialog's own tests type).
        h.state_mut().ui.dialog_mut(dialog).unwrap().fields.insert("name".into(), json!("Inky"));
        crate::dialogs::confirm(h.state_mut(), dialog).unwrap();
        h.run_steps(2);
        assert!(h.state().session.tools.presets.iter().any(|p| p.name == "Inky"), "saved under the typed name");
        assert_eq!(h.state().session.tools.current_preset.as_deref(), Some("Inky"));
        h.state_mut().ui.brush_picker = Some([c.x, c.y]);
        h.run_steps(2);
        // The gear's boxes pick the card parts; the last one on can't be turned off.
        h.get_by_label("Brush Preset Options").click();
        h.run_steps(2);
        h.get_by_label("Brush Tip").click();
        h.run_steps(2);
        assert!(!h.state().ui.brush_picker_list.show_tip);
        h.get_by_label("Brush Stroke").click();
        h.run_steps(2);
        h.get_by_label("Brush Name").click();
        h.run_steps(2);
        assert!(h.state().ui.brush_picker_list.show_name, "the last part on stays on");
        assert!(!h.state().ui.brush_picker_list.show_stroke && !h.state().ui.brush_picker_list.show_tip);
        assert!(h.state().ui.brush_picker.is_some());
        // A rename left open goes with the picker.
        h.state_mut().ui.brush_picker_list.renaming =
            Some(crate::brush_panel::Renaming { group: false, name: "Inky".into(), folder: Vec::new(), text: String::new() });
        h.run_steps(2);
        let far = c - vec2(300.0, 200.0);
        h.event(egui::Event::PointerMoved(far));
        press(&mut h, far, PointerButton::Primary, true);
        press(&mut h, far, PointerButton::Primary, false);
        assert!(h.state().ui.brush_picker.is_none() && h.state().ui.brush_picker_list.renaming.is_none());
    }

    #[test]
    fn right_drag_erases_with_the_erase_preference() {
        let mut h = harness(Some("erase"));
        let (a, b) = drag(&mut h, PointerButton::Primary);
        let mid = a + (b - a) * 0.5;
        assert!(alpha_at(&h, mid) > 0.9);
        let undo = h.state().session.active().unwrap().history.past_len();
        drag(&mut h, PointerButton::Secondary);
        let s = strokes(&h);
        assert_eq!(s.len(), 2);
        assert_eq!(s[1]["erase"], json!(true), "the right stroke erases");
        assert_eq!(alpha_at(&h, mid), 0.0, "erased to transparency");
        assert_eq!(h.state().session.active().unwrap().history.past_len(), undo + 1, "one undo step");
        assert_eq!(h.state().ui.brush_picker, None, "no picker in erase mode");
        // The next left stroke paints again.
        drag(&mut h, PointerButton::Primary);
        assert_eq!(strokes(&h)[2]["erase"], json!(false));
        assert!(alpha_at(&h, mid) > 0.9);
    }

    #[test]
    fn right_stroke_erases_with_pen_pressure_through_the_control_channel_path() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
        app.run("prefs.set", json!({"path": "tools.rightClickWithPaintingTools", "value": "erase"})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = Modifiers::NONE;
        assert!(pointer_secondary(&mut app, &egui::Context::default(), true, egui::Modifiers::NONE, [0.0, 0.0]));
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 0.4 }, m);
        assert!(!app.secondary_erase, "armed for one stroke only");
        tool_event(&mut app, ToolEvent::Move { x: 100.0, y: 40.0, pressure: 0.8 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 40.0 }, m);
        let p = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").map(|(_, p)| p.clone()).unwrap();
        assert_eq!(p["erase"], json!(true));
        assert_eq!(p["points"][0], json!([10.0, 40.0, 0.4f32 as f64]), "pressure reaches the erasing stroke");
        // On the Background the erasing stroke paints the background colour (#76).
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers[0].surface().unwrap().rgba(50, 40), [1.0, 1.0, 1.0, 1.0]);
        // With the default preference the right button opens the picker and nothing paints.
        app.run("prefs.set", json!({"path": "tools.rightClickWithPaintingTools", "value": "brushPicker"})).unwrap();
        assert!(!pointer_secondary(&mut app, &egui::Context::default(), true, egui::Modifiers::NONE, [0.0, 0.0]));
        assert!(app.ui.brush_picker.is_some() && !app.secondary_erase);
    }

    #[test]
    fn the_brush_is_a_per_tool_option_that_presets_keep() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        let amount = |app: &PhotocraftApp| app.session.tools.brush.smoothing.amount;
        app.ui.tool = Tool::Brush;
        sync_tool_brush(&mut app);
        assert_eq!(amount(&app), 0.1, "Photoshop's default");
        // Picking a preset (whose own smoothing is 0) keeps the tool's smoothing.
        let preset = app.session.tools.presets.iter().find(|p| p.brush.smoothing.amount == 0.0).map(|p| p.name.clone()).unwrap();
        app.run("tools.setBrush", json!({"preset": preset})).unwrap();
        assert_eq!(amount(&app), 0.1);
        // A user value (the options-bar field) survives a preset too, mode included.
        app.session.tools.brush.smoothing.amount = 0.42;
        app.session.tools.brush.smoothing.pulled_string = true;
        app.run("tools.setBrush", json!({"preset": preset})).unwrap();
        assert_eq!((amount(&app), app.session.tools.brush.smoothing.pulled_string), (0.42, true));
        // The Eraser has its own (10 % at first); switching back restores the Brush's.
        app.ui.tool = Tool::Eraser;
        sync_tool_brush(&mut app);
        assert_eq!(amount(&app), 0.1);
        app.session.tools.brush.smoothing.amount = 0.0;
        app.ui.tool = Tool::Brush;
        sync_tool_brush(&mut app);
        assert_eq!((amount(&app), app.session.tools.brush.smoothing.pulled_string), (0.42, true));
        // Other tools leave it alone; the Eraser kept its 0 %.
        app.ui.tool = Tool::Move;
        sync_tool_brush(&mut app);
        assert_eq!(amount(&app), 0.42);
        app.ui.tool = Tool::Eraser;
        // A stroke picks up the tool's brush even without a frame in between.
        tool_event(&mut app, ToolEvent::Down { x: 5.0, y: 5.0, pressure: 1.0 }, Modifiers::NONE);
        assert_eq!(amount(&app), 0.0);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 5.0 }, Modifiers::NONE);
    }

    /// #218: Photoshop keeps a brush per painting tool, so the size, hardness, mode, opacity and
    /// dynamics of one tool never leak into another's, and each tool gets its own back.
    #[test]
    fn each_tool_keeps_its_own_brush() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        let brush = |app: &PhotocraftApp| app.session.tools.brush.clone();
        let size = |b: &BrushSettings| b.size;
        let hardness = |b: &BrushSettings| b.hardness;
        let opacity = |b: &BrushSettings| b.opacity;
        let mode = |b: &BrushSettings| b.mode;

        app.ui.tool = Tool::Brush;
        sync_tool_brush(&mut app);
        app.run("tools.setBrush", json!({"brush": {"size": 40, "hardness": 0.2, "opacity": 0.7, "mode": "Multiply"}})).unwrap();
        assert_eq!(size(&brush(&app)), 40.0);

        // A tool seen for the first time starts from the brush the old tool left (Photoshop's
        // 10 % smoothing), so nothing is lost by switching.
        app.ui.tool = Tool::Eraser;
        sync_tool_brush(&mut app);
        assert_eq!(size(&brush(&app)), 40.0, "a new tool inherits the current brush");

        // The Eraser edits its own size; the Brush keeps 40 px.
        app.run("tools.setBrush", json!({"brush": {"size": 8, "hardness": 1.0}})).unwrap();
        app.ui.tool = Tool::Brush;
        sync_tool_brush(&mut app);
        assert_eq!(size(&brush(&app)), 40.0, "the Brush keeps its own size");
        assert_eq!(hardness(&brush(&app)), 0.2, "and its hardness");
        app.ui.tool = Tool::Eraser;
        sync_tool_brush(&mut app);
        assert_eq!(size(&brush(&app)), 8.0, "the Eraser keeps its own size");
        assert_eq!(hardness(&brush(&app)), 1.0, "and its hardness");

        // Mode, opacity, flow and the dynamics sections are per tool too.
        app.ui.tool = Tool::CloneStamp;
        sync_tool_brush(&mut app);
        app.run("tools.setBrush", json!({"brush": {"mode": "Screen", "opacity": 0.3}})).unwrap();
        app.ui.tool = Tool::Brush;
        sync_tool_brush(&mut app);
        assert_eq!((mode(&brush(&app)), opacity(&brush(&app))), (photocraft_color::BlendMode::Multiply, 0.7), "the Brush's own mode and opacity");
        app.ui.tool = Tool::CloneStamp;
        sync_tool_brush(&mut app);
        assert_eq!((mode(&brush(&app)), opacity(&brush(&app))), (photocraft_color::BlendMode::Screen, 0.3), "the Clone Stamp keeps its own");

        // The colours, the erase flag and the seed belong to the session, so a swap keeps what is
        // live now instead of restoring another tool's copy (a picked colour reaches every tool).
        app.run("tools.setBrush", json!({"brush": {"color": [0.0, 1.0, 0.0, 1.0], "background": [1.0, 0.0, 1.0, 1.0], "erase": true, "seed": 42}})).unwrap();
        app.ui.tool = Tool::Brush;
        sync_tool_brush(&mut app);
        let b = brush(&app);
        assert_eq!(
            (b.color, b.background, b.erase, b.seed),
            ([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0], true, 42),
            "the session's own brush fields survive the swap"
        );

        // Tools with no brush of their own (the Move) leave the brush alone.
        let before = brush(&app);
        app.ui.tool = Tool::Move;
        sync_tool_brush(&mut app);
        assert_eq!(brush(&app), before);
    }

    #[test]
    fn eraser_tool_on_the_background_paints_the_background_colour() {
        // #76: the Eraser on the Background (locked transparency) paints the background colour.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
        app.run("tools.setColors", json!({"background": "#ff0000"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 12, "hardness": 1.0}})).unwrap();
        app.ui.tool = Tool::Eraser;
        let m = Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 100.0, y: 40.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 40.0 }, m);
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers[0].name, "Background");
        assert_eq!(st.doc.layers[0].surface().unwrap().rgba(50, 40), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(st.history.undo_label(), Some("Eraser"));
        assert!(!app.ui.status_error, "{}", app.ui.status);
    }

    /// Assert the one committed stroke contains a point at each queued screen position.
    fn assert_committed_points(h: &Harness<'static, PhotocraftApp>, queued: &[Pos2]) {
        let s = strokes(h);
        assert_eq!(s.len(), 1);
        let points = s[0]["points"].as_array().unwrap();
        let app = h.state();
        let v = app.ui.views[0].clone();
        let r = app.last_canvas_rect;
        for p in queued {
            let d = (*p - r.center()) / v.zoom;
            let want = [d.x as f64 + v.center[0] as f64, d.y as f64 + v.center[1] as f64];
            assert!(
                points.iter().any(|q| {
                    let q = q.as_array().unwrap();
                    (q[0].as_f64().unwrap() - want[0]).abs() < 1.0 && (q[1].as_f64().unwrap() - want[1]).abs() < 1.0
                }),
                "queued move {want:?} missing from {points:?}"
            );
        }
    }

    #[test]
    fn one_frame_feeds_every_pointer_move_not_just_the_latest() {
        // Fast strokes used to lose the moves between two frames: the canvas read only
        // `interact_pointer_pos()` once per frame, so the stroke was a coarse polyline. Every
        // `PointerMoved` egui-winit delivered in one frame must now reach the stroke.
        let mut h = harness(None);
        let c = h.state().last_canvas_rect.center();
        let a = c - vec2(90.0, 0.0);
        h.event(egui::Event::PointerMoved(a));
        h.run_steps(1);
        press(&mut h, a, PointerButton::Primary, true);
        // Cross egui's drag threshold on its own frame.
        h.event(egui::Event::PointerMoved(a + vec2(20.0, 0.0)));
        h.run_steps(1);
        // Several moves in ONE frame: `Harness::event` would make one frame per event, so push
        // them straight into the next frame's input. All of them must reach the committed stroke.
        let queued: Vec<Pos2> = (1..=6).map(|i| a + vec2(20.0 + i as f32 * 8.0, 0.0)).collect();
        for p in &queued {
            h.input_mut().events.push(egui::Event::PointerMoved(*p));
        }
        h.run_steps(1);
        let end = *queued.last().unwrap();
        press(&mut h, end, PointerButton::Primary, false);
        assert_committed_points(&h, &queued);
    }
}
