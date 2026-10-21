//! The right-click edit menu of every text field (#2921): Undo, Cut, Copy, Paste, Delete and
//! Select All at the pointer, acting on that field, as native text fields and Photoshop's own do.
//!
//! egui's `TextEdit` has no context menu, and fields are built all over the UI (layer rename,
//! number fields being typed into, dialogs, panels), so this is an egui plugin rather than a
//! wrapper each field has to use. A secondary press in the focused text field (egui focuses a
//! field on any press in it) opens the menu at the end of that pass. The app draws the menu
//! ([`show`]) at the start of its UI, before the fields: a chosen item gives the field its focus
//! back and hands it the matching input (egui's `Copy` / `Cut` events, ⌘A, ⌘Z, Delete; Paste asks
//! the window for the clipboard text, as ⌘V in a field does), which the field handles later in
//! the same pass. The application shortcuts have already run by then and never see that input.
//!
//! While the menu is open the field keeps its focus when the menu is clicked (egui would drop it
//! on any click outside the field, and fields such as the layer rename commit when they lose
//! it); a press anywhere else closes the menu and gives that press its usual effect, and Esc closes
//! only the menu. A right-click inside the field's selection keeps the selection (egui moves the
//! caret to the pointer on any press), so Copy and Cut act on it.
//!
//! The right-click belongs to the field: the canvas and its tool menus only see pointer presses
//! on the canvas, which a field drawn above it takes. Spelling suggestions are not in this menu:
//! the spell checker (`edit.checkSpelling`) checks type layers, so they head the Type tool's edit
//! menu instead (`canvas_tool_menu`, over a misspelled word of the text being edited, #4370).

use egui::text::CCursorRange;
use egui::{Context, Event, Id, Key, Modifiers, PointerButton, Pos2, RawInput, SurrenderFocusOn, Ui};

/// One menu item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Undo,
    Cut,
    Copy,
    Paste,
    Delete,
    SelectAll,
}

/// The menu, in its order: label, the keys that do the same in a field (none shown for Delete,
/// whose key is its name), and the action. `None` is a separator.
const ROWS: &[Option<(&str, &str, Action)>] = &[
    Some(("Undo", "Cmd+Z", Action::Undo)),
    None,
    Some(("Cut", "Cmd+X", Action::Cut)),
    Some(("Copy", "Cmd+C", Action::Copy)),
    Some(("Paste", "Cmd+V", Action::Paste)),
    Some(("Delete", "", Action::Delete)),
    None,
    Some(("Select All", "Cmd+A", Action::SelectAll)),
];

/// The open menu.
#[derive(Clone, Copy, Debug)]
struct Open {
    /// The text field it edits.
    field: Id,
    /// Where it was opened (its top-left corner).
    pos: Pos2,
    /// The field had a selection: Cut, Copy and Delete have something to act on.
    has_selection: bool,
    /// The focus rule to put back when the menu closes.
    surrender: SurrenderFocusOn,
}

/// What the menu remembers between passes (in the context's data: the plugin's hooks and the
/// app's [`show`] both use it).
#[derive(Clone, Debug, Default)]
struct State {
    open: Option<Open>,
    /// The focused text field's selection at the end of the last pass.
    last: Option<(Id, CCursorRange)>,
    /// The focus rule to put back at the start of the next pass.
    restore: Option<SurrenderFocusOn>,
}

impl State {
    fn load(ctx: &Context) -> State {
        ctx.data(|d| d.get_temp::<State>(menu_id())).unwrap_or_default()
    }

    fn store(self, ctx: &Context) {
        ctx.data_mut(|d| d.insert_temp(menu_id(), self));
    }

    /// Close the menu; its field's focus rule comes back at the start of the next pass.
    fn close(&mut self) {
        if let Some(o) = self.open.take() {
            self.restore = Some(o.surrender);
        }
    }

    /// Put back the focus rule the menu replaced, now.
    fn restore_now(&mut self, ctx: &Context) {
        if let Some(rule) = self.restore.take() {
            ctx.options_mut(|o| o.input_options.surrender_focus_on = rule);
        }
    }
}

/// The menu's area.
pub fn menu_id() -> Id {
    Id::new("text-field-menu")
}

/// Is the edit menu open, and on which text field?
pub fn open_on(ctx: &Context) -> Option<Id> {
    State::load(ctx).open.map(|o| o.field)
}

/// Opens and closes the menu. It draws nothing: a plugin's hooks run with every plugin locked,
/// and a widget under the pointer calls them. The app draws the menu with [`show`].
/// Installed by `PhotocraftApp::setup_context`.
pub struct TextFieldMenu;

impl egui::plugin::Plugin for TextFieldMenu {
    fn debug_name(&self) -> &'static str {
        "PhotoCraft text field menu"
    }

    /// Esc closes only the menu: the field (a layer rename cancels on Esc) and egui (which drops
    /// the focus on Esc) never see it.
    fn input_hook(&mut self, ctx: &Context, input: &mut RawInput) {
        let mut state = State::load(ctx);
        if state.open.is_none() {
            return;
        }
        let before = input.events.len();
        input.events.retain(|e| !matches!(e, Event::Key { key: Key::Escape, .. }));
        if input.events.len() != before {
            state.close();
            state.store(ctx);
        }
    }

    fn on_begin_pass(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let mut state = State::load(&ctx);
        state.restore_now(&ctx);
        if let Some(open) = state.open {
            // The field went away (its dialog closed, the rename ended) or lost its focus some
            // other way; or a press elsewhere, which closes the menu and does what it does
            // without it. (The menu's rect is last pass's: none before it was first drawn.)
            let menu = ctx.memory(|m| m.area_rect(menu_id()));
            let elsewhere = ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().zip(menu).is_some_and(|(p, r)| !r.contains(p)));
            if ctx.read_response(open.field).is_none() || !ctx.memory(|m| m.has_focus(open.field)) || elsewhere {
                state.close();
                state.restore_now(&ctx);
            }
        }
        state.store(&ctx);
    }

    fn on_end_pass(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let mut state = State::load(&ctx);
        let focused = ctx.memory(|m| m.focused());
        if state.open.is_none()
            && let Some(field) = focused
            && let Some((pos, has_selection)) = secondary_press_in(&ctx, field, state.last)
        {
            let surrender = ctx.options(|o| o.input_options.surrender_focus_on);
            ctx.options_mut(|o| o.input_options.surrender_focus_on = SurrenderFocusOn::Never);
            state.open = Some(Open { field, pos, has_selection, surrender });
            ctx.move_to_top(egui::LayerId::new(egui::Order::Foreground, menu_id()));
            ctx.request_repaint();
        }
        state.last = focused.and_then(|id| egui::TextEdit::load_state(&ctx, id)?.cursor.char_range().map(|r| (id, r)));
        state.store(&ctx);
    }
}

/// A secondary press this pass inside the focused text field `field`: where, and whether the
/// field has a selection. A press inside the selection the field had at the end of the last pass
/// (`last`) keeps it: egui moved the caret to the pointer.
fn secondary_press_in(ctx: &Context, field: Id, last: Option<(Id, CCursorRange)>) -> Option<(Pos2, bool)> {
    let (pressed, origin) = ctx.input(|i| (i.pointer.button_pressed(PointerButton::Secondary), i.pointer.press_origin()));
    let pos = origin.filter(|_| pressed)?;
    let mut state = egui::TextEdit::load_state(ctx, field)?;
    let response = ctx.read_response(field)?;
    if !response.interact_rect.contains(pos) || ctx.layer_id_at(pos) != Some(response.layer_id) {
        return None;
    }
    let mut range = state.cursor.char_range();
    if let Some((id, before)) = last
        && id == field
        && !before.is_empty()
        && range.is_some_and(|now| before.as_sorted_char_range().contains(&now.primary.index) || before.as_sorted_char_range().end == now.primary.index)
    {
        state.cursor.set_char_range(Some(before));
        state.store(ctx, field);
        range = Some(before);
    }
    Some((pos, range.is_some_and(|r| !r.is_empty())))
}

/// Is `action` available? Cut, Copy and Delete need a selection.
fn enabled(open: &Open, action: Action) -> bool {
    match action {
        Action::Cut | Action::Copy | Action::Delete => open.has_selection,
        Action::Undo | Action::Paste | Action::SelectAll => true,
    }
}

/// Draw the open menu and do the item chosen in it. The app calls this first thing in a pass,
/// before any text field is drawn, so the field handles the chosen item in the same pass.
pub fn show(ctx: &Context) {
    let mut state = State::load(ctx);
    let Some(open) = state.open else { return };
    if let Some(action) = draw(ctx, &open) {
        // The field keeps its focus through this pass's click; the rule comes back next pass.
        state.close();
        state.store(ctx);
        apply(ctx, open.field, action);
    }
}

/// Draw the menu; returns the item chosen this pass.
fn draw(ctx: &Context, open: &Open) -> Option<Action> {
    let id = menu_id();
    let screen = ctx.content_rect();
    let size = ctx.memory(|m| m.area_rect(id)).map_or(egui::vec2(180.0, 170.0), |r| r.size());
    let pos = egui::pos2(open.pos.x.min(screen.right() - size.x).max(screen.left()), open.pos.y.min(screen.bottom() - size.y).max(screen.top()));
    let mut chosen = None;
    egui::Area::new(id).order(egui::Order::Foreground).fixed_pos(pos).show(ctx, |ui| {
        egui::Frame::menu(ui.style()).show(ui, |ui| {
            let t = crate::theme::Tokens::get(ui.ctx());
            let v = &mut ui.style_mut().visuals;
            v.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
            v.widgets.inactive.bg_stroke = egui::Stroke::NONE;
            v.widgets.hovered.weak_bg_fill = t.accent;
            v.widgets.hovered.bg_fill = t.accent;
            v.widgets.hovered.bg_stroke = egui::Stroke::NONE;
            v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
            v.widgets.hovered.corner_radius = egui::CornerRadius::same(3);
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.set_width(170.0);
            for row in ROWS {
                let Some((label, keys, action)) = *row else {
                    ui.separator();
                    continue;
                };
                let mut item = egui::Button::new(tl!(label)).min_size(egui::vec2(170.0, 20.0));
                if !keys.is_empty() {
                    item = item.shortcut_text(crate::shortcuts::pretty(keys));
                }
                if ui.add_enabled(enabled(open, action), item).clicked() {
                    chosen = Some(action);
                }
            }
        });
    });
    chosen
}

/// Do `action` in `field`: give it its focus back and the input that does the same from the
/// keyboard, which it handles later in this pass.
fn apply(ctx: &Context, field: Id, action: Action) {
    ctx.memory_mut(|m| m.request_focus(field));
    let key = |key, modifiers| Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers };
    let event = match action {
        Action::Undo => key(Key::Z, Modifiers::COMMAND),
        Action::Cut => Event::Cut,
        Action::Copy => Event::Copy,
        Action::Delete => key(Key::Delete, Modifiers::NONE),
        Action::SelectAll => key(Key::A, Modifiers::COMMAND),
        Action::Paste => {
            // The clipboard text arrives as a Paste event next pass, as for ⌘V in a field.
            ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaste);
            return;
        }
    };
    ctx.input_mut(|i| i.events.push(event));
}

#[cfg(test)]
mod tests {
    use egui::{Event, Id, Key, Modifiers, OutputCommand, PointerButton, Pos2, vec2};
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
    use serde_json::json;

    use super::open_on;
    use crate::PhotocraftApp;
    use crate::layer_row_ui::{rename, start_rename};

    /// The app with a layer "Alpha" being renamed (its name selected, as a rename starts) and the
    /// Rectangular Marquee, whose canvas has a context menu of its own.
    fn renaming() -> (Harness<'static, PhotocraftApp>, Id) {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        let layer = s.execute("layer.new.layer", json!({"name": "Alpha"})).unwrap()["layer"].as_u64().unwrap();
        let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            PhotocraftApp::new(s, crate::Services::default())
        });
        h.state_mut().ui.tool = crate::state::Tool::RectMarquee;
        h.run_steps(8);
        let ctx = h.ctx.clone();
        assert!(start_rename(&ctx, layer, "Alpha").is_none());
        h.run_steps(4);
        (h, Id::new(("layer-rename-field", layer)))
    }

    fn button(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, button: PointerButton) {
        h.hover_at(at);
        h.step();
        for pressed in [true, false] {
            h.event(Event::PointerButton { pos: at, button, pressed, modifiers: Modifiers::NONE });
            h.step();
        }
    }

    /// Right-click in the middle of `field`; the menu opens on it.
    fn right_click(h: &mut Harness<'_, PhotocraftApp>, field: Id) {
        let at = h.ctx.read_response(field).expect("field drawn").rect.center();
        button(h, at, PointerButton::Secondary);
        assert_eq!(open_on(&h.ctx), Some(field), "a right-click in a text field opens its edit menu");
        // A new area is measured in an invisible first pass; it is placed in the next one.
        h.step();
    }

    /// Where the open menu's `label` item is.
    fn item(h: &Harness<'_, PhotocraftApp>, label: &str) -> Option<Pos2> {
        let menu = h.ctx.memory(|m| m.area_rect(super::menu_id()))?;
        h.query_all_by_label_contains(label).map(|n| n.rect().center()).find(|p| menu.contains(*p))
    }

    /// Choose `label` in the open menu; returns the text the pass that ran it copied, if any.
    fn choose(h: &mut Harness<'_, PhotocraftApp>, label: &str) -> Option<String> {
        let at = item(h, label).expect("item in the menu");
        h.hover_at(at);
        h.step();
        h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.step();
        h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.step();
        let copied = h.output().platform_output.commands.iter().find_map(|c| match c {
            OutputCommand::CopyText(t) => Some(t.clone()),
            _ => None,
        });
        h.run_steps(2);
        copied
    }

    fn text(h: &Harness<'_, PhotocraftApp>) -> String {
        rename(&h.ctx).expect("still renaming").text
    }

    #[test]
    fn right_click_in_a_text_field_opens_its_edit_menu_not_the_canvas_menu() {
        let (mut h, field) = renaming();
        right_click(&mut h, field);
        assert!(h.state().ui.canvas_tool_menu.is_none() && h.state().ui.brush_picker.is_none(), "the canvas and tool menus stay shut");
        assert!(!egui::Popup::is_any_open(&h.ctx), "and so does the layer row's menu under the field");
        for label in ["Undo", "Cut", "Copy", "Paste", "Delete", "Select All"] {
            assert!(item(&h, label).is_some(), "{label} is in the menu");
        }
        // Esc closes the menu only: the rename stays open with its text.
        h.event(Event::Key { key: Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
        h.run_steps(2);
        assert_eq!(open_on(&h.ctx), None);
        assert_eq!(text(&h), "Alpha");
        assert!(h.ctx.memory(|m| m.has_focus(field)), "the field keeps its focus");
    }

    #[test]
    fn text_field_menu_select_all_then_cut_moves_the_text_to_the_clipboard() {
        let (mut h, field) = renaming();
        // A click in the field drops the selection the rename started with.
        let at = h.ctx.read_response(field).unwrap().rect.center();
        button(&mut h, at, PointerButton::Primary);
        right_click(&mut h, field);
        assert_eq!(choose(&mut h, "Select All"), None);
        assert_eq!(open_on(&h.ctx), None, "choosing closes the menu");
        // A right-click inside the selection keeps it, so Cut takes the whole name.
        right_click(&mut h, field);
        assert_eq!(choose(&mut h, "Cut").as_deref(), Some("Alpha"), "Cut copies the selected text");
        assert_eq!(text(&h), "", "and removes it from the field");
        assert!(h.ctx.memory(|m| m.has_focus(field)), "the rename is still being typed");
    }
}
