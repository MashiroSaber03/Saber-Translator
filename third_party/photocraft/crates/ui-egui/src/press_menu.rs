//! Popup menu buttons that behave like native menus (#2071), as Photoshop's Layers panel footer
//! buttons do: the menu opens on the press, so one gesture can press the button, drag onto an item
//! and release to choose it. A plain click still opens the menu and leaves it open for a second
//! click; a release on nothing (or back on the button) chooses nothing. A secondary click opens it
//! too. The menu bar has its own copy of this gesture in `menus.rs`, which also spans titles.

use egui::{Popup, PopupCloseBehavior, Response, SetOpenCommand, Ui};

fn gesture_id() -> egui::Id {
    egui::Id::new("press-menu-gesture")
}

/// The popup whose button's press started the current press-drag gesture, if any. Several press
/// menus run in one frame (the Layers footer has two): each owns only its own gesture.
fn gesture_owner(ctx: &egui::Context) -> Option<egui::Id> {
    ctx.data(|d| d.get_temp::<egui::Id>(gesture_id()))
}

/// This frame's open/close command for `button`'s popup: a primary press opens a closed menu
/// (starting a press-drag gesture) or closes an open one, a secondary click opens it, and the
/// primary release never toggles, so the menu doesn't blink.
fn open_command(button: &Response, popup: egui::Id) -> Option<SetOpenCommand> {
    let ctx = &button.ctx;
    if button.secondary_clicked() {
        return Some(SetOpenCommand::Bool(true));
    }
    // A press this frame on the button (still down, or a whole click within one frame).
    let pressed = ctx.input(|i| i.pointer.primary_pressed()) && (button.is_pointer_button_down_on() || button.clicked());
    if !pressed {
        return None;
    }
    let open = Popup::is_id_open(ctx, popup);
    if !open {
        ctx.data_mut(|d| d.insert_temp(gesture_id(), popup));
    }
    Some(SetOpenCommand::Bool(!open))
}

/// One press menu, handed to [`show`]'s content to choose its items with.
pub struct PressMenu {
    popup: egui::Id,
}

impl PressMenu {
    fn owns_gesture(&self, ctx: &egui::Context) -> bool {
        gesture_owner(ctx) == Some(self.popup)
    }

    /// Was `item` chosen this frame: clicked, or the press-drag gesture that opened this menu was
    /// released on it?
    pub fn chosen(&self, ui: &Ui, item: &Response) -> bool {
        item.clicked() || (item.enabled() && item.contains_pointer() && ui.input(|i| i.pointer.primary_released()) && self.owns_gesture(ui.ctx()))
    }

    /// A menu item button; true (and the menu closes) when it is chosen.
    pub fn item(&self, ui: &mut Ui, text: impl Into<egui::WidgetText>) -> bool {
        self.item_response(ui, text).1
    }

    /// Is the press-drag gesture that opened this menu over `item`, still held? egui doesn't
    /// hover other widgets while one is dragged, so the item would show no highlight (#4601).
    fn dragged_over(&self, ui: &Ui, item: &Response) -> bool {
        item.enabled() && !item.hovered() && item.contains_pointer() && ui.input(|i| i.pointer.primary_down()) && self.owns_gesture(ui.ctx())
    }

    fn item_response(&self, ui: &mut Ui, text: impl Into<egui::WidgetText>) -> (Response, bool) {
        // Under the item: the highlight it would have if hovered, while the gesture drags over it.
        let highlight = ui.painter().add(egui::Shape::Noop);
        let response = ui.button(text);
        if self.dragged_over(ui, &response) {
            let v = &ui.visuals().widgets.hovered;
            ui.painter().set(highlight, egui::Shape::rect_filled(response.rect.expand(v.expansion), v.corner_radius, v.weak_bg_fill));
        }
        let hit = self.chosen(ui, &response);
        if hit {
            ui.close();
        }
        (response, hit)
    }
}

/// Shows `button`'s menu (an `egui::Popup::menu`) with press-drag-release; `content` chooses its
/// items with [`PressMenu::item`].
pub fn show<R>(button: &Response, content: impl FnOnce(&mut Ui, &PressMenu) -> R) -> Option<R> {
    let ctx = button.ctx.clone();
    let menu = PressMenu { popup: Popup::default_response_id(button) };
    let open = open_command(button, menu.popup);
    // The release ending the press that opened the menu is a click outside the popup: it must not
    // close it again.
    let close = if button.clicked() && menu.owns_gesture(&ctx) { PopupCloseBehavior::IgnoreClicks } else { PopupCloseBehavior::CloseOnClick };
    let shown = Popup::menu(button).open_memory(open).close_behavior(close).show(|ui| content(ui, &menu)).map(|r| r.inner);
    // This menu's gesture ends with the button (its release was handled by the items above).
    if menu.owns_gesture(&ctx) && ctx.input(|i| i.pointer.primary_released() || !i.pointer.primary_down()) {
        ctx.data_mut(|d| d.remove::<egui::Id>(gesture_id()));
    }
    shown
}

/// The same press-drag-release for an `egui::ComboBox` (#2735): the list opens on the press, so one
/// gesture can press the box, drag onto an option and release to choose it, and a plain click
/// still opens the list for a second click. egui's combo box shows its own list and toggles it on
/// a whole click, so this works around it: [`PressCombo::before`] runs before the box (and gives
/// it its close behaviour), [`PressCombo::chosen`] picks its options, [`PressCombo::after`] runs
/// after it.
pub struct PressCombo {
    popup: egui::Id,
}

impl PressCombo {
    /// Before showing the combo box made with `ComboBox::from_id_salt(salt)` in `ui`; returns the
    /// close behaviour to give it.
    pub fn before(ui: &Ui, salt: impl egui::AsIdSalt) -> (Self, PopupCloseBehavior) {
        // egui's ids for the box (`ComboBox::show_ui`) and its list (`widget_to_popup_id`).
        let combo = PressCombo { popup: ui.make_persistent_id(egui::IdSalt::new(salt)).with("popup") };
        let ctx = ui.ctx();
        // The click ending the press that opened the list (the pointer didn't leave the box): egui
        // toggles the list on it, so close it first and the toggle leaves it open, drawn this
        // frame. No click may close it either.
        if combo.owns_gesture(ctx) && ctx.input(|i| i.pointer.primary_clicked()) {
            Popup::close_id(ctx, combo.popup);
            return (combo, PopupCloseBehavior::IgnoreClicks);
        }
        (combo, PopupCloseBehavior::CloseOnClick)
    }

    fn owns_gesture(&self, ctx: &egui::Context) -> bool {
        gesture_owner(ctx) == Some(self.popup)
    }

    /// Was option `item` (in the open list `ui`) chosen this frame: clicked, or the press-drag
    /// gesture that opened the list released on it? The list closes when it was.
    pub fn chosen(&self, ui: &Ui, item: &Response) -> bool {
        let released_on = item.enabled() && item.contains_pointer() && ui.input(|i| i.pointer.primary_released()) && self.owns_gesture(ui.ctx());
        if released_on {
            ui.close();
        }
        item.clicked() || released_on
    }

    /// After the combo box (`button`: its response). A primary press on the box opens a closed
    /// list, starting a press-drag gesture; the gesture ends with the button.
    pub fn after(&self, button: &Response) {
        let ctx = &button.ctx;
        let pressed = ctx.input(|i| i.pointer.primary_pressed()) && button.is_pointer_button_down_on();
        if pressed && !Popup::is_id_open(ctx, self.popup) {
            // Opened now, the list counts as open this frame and egui shows it from the next.
            Popup::open_id(ctx, self.popup);
            ctx.data_mut(|d| d.insert_temp(gesture_id(), self.popup));
            ctx.request_repaint();
        } else if self.owns_gesture(ctx) && ctx.input(|i| i.pointer.primary_released() || !i.pointer.primary_down()) {
            ctx.data_mut(|d| d.remove::<egui::Id>(gesture_id()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, Modifiers, PointerButton, Pos2, Rect, pos2, vec2};

    /// A bare context with two menu buttons side by side (like the Layers footer's adjustment and
    /// fx buttons), each with two items; records what each frame chose.
    struct Rig {
        ctx: egui::Context,
        buttons: [Rect; 2],
        items: [Vec<Rect>; 2],
    }

    #[derive(Debug, Default)]
    struct Frame {
        open: bool,
        chosen: Option<&'static str>,
        /// The items drawn with a (hover) fill under them.
        highlighted: Vec<&'static str>,
    }

    const LABELS: [[&str; 2]; 2] = [["Levels", "Curves"], ["Drop Shadow", "Stroke"]];

    impl Rig {
        fn new() -> Self {
            let mut rig = Rig { ctx: egui::Context::default(), buttons: [Rect::NOTHING; 2], items: [Vec::new(), Vec::new()] };
            rig.frame(Vec::new());
            rig
        }

        fn frame(&mut self, events: Vec<Event>) -> Frame {
            let mut out = Frame::default();
            let mut items = [Vec::new(), Vec::new()];
            let rect = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 400.0));
            let mut buttons = [Rect::NOTHING; 2];
            let mut labels = Vec::new();
            let mut output = self.ctx.run_ui(egui::RawInput { screen_rect: Some(rect), events, ..Default::default() }, |ui| {
                ui.add_space(20.0);
                ui.horizontal(|ui| {
                    for (m, name) in ["First menu", "Second menu"].into_iter().enumerate() {
                        let b = ui.add_sized([100.0, 28.0], egui::Button::new(name));
                        buttons[m] = b.rect;
                        show(&b, |ui, menu| {
                            out.open = true;
                            for label in LABELS[m] {
                                let (r, hit) = menu.item_response(ui, label);
                                if hit {
                                    out.chosen = Some(label);
                                }
                                items[m].push(r.rect);
                                labels.push((label, r.rect));
                            }
                        });
                    }
                });
            });
            output.textures_delta.clear();
            for (label, rect) in labels {
                // An item's own frame is transparent unless hovered; the popup's fade-in can soften
                // the fill's alpha a little.
                let lit = output.shapes.iter().any(|c| matches!(&c.shape, egui::Shape::Rect(r) if r.fill.a() > 200 && r.rect.contains_rect(rect)));
                if lit {
                    out.highlighted.push(label);
                }
            }
            self.buttons = buttons;
            for (m, list) in items.into_iter().enumerate() {
                if !list.is_empty() {
                    self.items[m] = list;
                }
            }
            out
        }

        fn press(&mut self, pos: Pos2, pressed: bool) -> Frame {
            self.frame(vec![Event::PointerMoved(pos), Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE }])
        }

        fn drag_to(&mut self, from: Pos2, to: Pos2) {
            for k in 1..=8 {
                self.frame(vec![Event::PointerMoved(from + (to - from) * (k as f32 / 8.0))]);
            }
        }

        fn item_center(&self, i: usize) -> Pos2 {
            self.items[0][i].center()
        }
    }

    #[test]
    fn press_drag_release_chooses_an_item() {
        let mut rig = Rig::new();
        let at = rig.buttons[0].center();
        rig.frame(vec![Event::PointerMoved(at)]);
        assert!(rig.press(at, true).open, "the menu opens on the press");
        rig.frame(Vec::new());
        let curves = rig.item_center(1);
        rig.drag_to(at, rig.item_center(0));
        assert_eq!(rig.frame(Vec::new()).highlighted, ["Levels"], "the item under the drag is highlighted (#4601)");
        rig.drag_to(rig.item_center(0), curves);
        assert_eq!(rig.frame(Vec::new()).highlighted, ["Curves"], "and only that one");
        let f = rig.press(curves, false);
        assert_eq!(f.chosen, Some("Curves"), "releasing on Curves chose it");
        assert!(!rig.frame(Vec::new()).open, "and closed the menu");
    }

    #[test]
    fn a_click_leaves_the_menu_open_for_a_second_click() {
        let mut rig = Rig::new();
        let at = rig.buttons[0].center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        let f = rig.press(at, false);
        assert!(f.open && f.chosen.is_none(), "a click opens the menu and chooses nothing");
        assert!(rig.frame(Vec::new()).open, "the menu stays open");
        let levels = rig.item_center(0);
        rig.frame(vec![Event::PointerMoved(levels)]);
        rig.press(levels, true);
        assert_eq!(rig.press(levels, false).chosen, Some("Levels"), "a second click chooses");
        assert!(!rig.frame(Vec::new()).open);
    }

    #[test]
    fn releasing_on_nothing_chooses_nothing() {
        let mut rig = Rig::new();
        let at = rig.buttons[0].center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        rig.frame(Vec::new());
        let away = pos2(350.0, 350.0);
        rig.drag_to(at, away);
        let f = rig.press(away, false);
        assert!(f.chosen.is_none(), "a release off the items chooses nothing");
        assert!(rig.frame(Vec::new()).open, "and keeps the menu open");
        // Dragging back onto the button and releasing chooses nothing either.
        let mut rig = Rig::new();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        rig.frame(Vec::new());
        rig.drag_to(at, rig.item_center(0));
        rig.drag_to(rig.item_center(0), at);
        let f = rig.press(at, false);
        assert!(f.chosen.is_none() && f.open, "a release back on the button keeps the menu open");
    }

    #[test]
    fn secondary_click_opens_and_pressing_the_button_again_closes() {
        let mut rig = Rig::new();
        let at = rig.buttons[0].center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.frame(vec![Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed: true, modifiers: Modifiers::NONE }]);
        let f = rig.frame(vec![Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed: false, modifiers: Modifiers::NONE }]);
        assert!(f.open, "a secondary click opens the menu");
        rig.press(at, true);
        rig.press(at, false);
        assert!(!rig.frame(Vec::new()).open, "a click on the button closes an open menu");
    }

    /// Both menus run every frame: the first menu's end-of-frame cleanup must not end a gesture
    /// that started on the second one before its items see the release.
    #[test]
    fn press_drag_release_works_on_the_second_of_two_menus() {
        let mut rig = Rig::new();
        let at = rig.buttons[1].center();
        rig.frame(vec![Event::PointerMoved(at)]);
        assert!(rig.press(at, true).open, "the second menu opens on the press");
        rig.frame(Vec::new());
        let stroke = rig.items[1][1].center();
        rig.drag_to(at, stroke);
        assert_eq!(rig.press(stroke, false).chosen, Some("Stroke"), "releasing on Stroke chose it");
        assert!(!rig.frame(Vec::new()).open, "and closed the menu");
    }

    /// A bare context with one `egui::ComboBox` of three options, driven with [`PressCombo`].
    struct ComboRig {
        ctx: egui::Context,
        value: usize,
        button: Rect,
        options: Vec<Rect>,
    }

    const OPTIONS: [&str; 3] = ["RGB", "Red", "Green"];

    impl ComboRig {
        fn new() -> Self {
            let mut rig = ComboRig { ctx: egui::Context::default(), value: 0, button: Rect::NOTHING, options: Vec::new() };
            rig.frame(Vec::new());
            rig
        }

        fn frame(&mut self, events: Vec<Event>) -> Frame {
            let mut out = Frame::default();
            let mut options = Vec::new();
            let rect = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 400.0));
            let mut value = self.value;
            let mut button = Rect::NOTHING;
            let mut output = self.ctx.run_ui(egui::RawInput { screen_rect: Some(rect), events, ..Default::default() }, |ui| {
                ui.add_space(20.0);
                let (combo, close) = PressCombo::before(ui, "channel");
                let shown = egui::ComboBox::from_id_salt("channel").selected_text(OPTIONS[value]).close_behavior(close).show_ui(ui, |ui| {
                    out.open = true;
                    for (i, label) in OPTIONS.into_iter().enumerate() {
                        let item = ui.selectable_label(value == i, label);
                        options.push(item.rect);
                        if combo.chosen(ui, &item) {
                            value = i;
                            out.chosen = Some(label);
                        }
                    }
                });
                combo.after(&shown.response);
                button = shown.response.rect;
            });
            output.textures_delta.clear();
            self.value = value;
            self.button = button;
            if !options.is_empty() {
                self.options = options;
            }
            out
        }

        fn press(&mut self, pos: Pos2, pressed: bool) -> Frame {
            self.frame(vec![Event::PointerMoved(pos), Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE }])
        }

        fn drag_to(&mut self, from: Pos2, to: Pos2) {
            for k in 1..=8 {
                self.frame(vec![Event::PointerMoved(from + (to - from) * (k as f32 / 8.0))]);
            }
        }
    }

    #[test]
    fn press_drag_release_on_a_combo_box_chooses_an_option() {
        let mut rig = ComboRig::new();
        let at = rig.button.center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        assert!(rig.frame(Vec::new()).open, "the list opens on the press");
        // its first frame only sizes it: the options are where they are drawn from the next
        rig.frame(Vec::new());
        let red = rig.options[1].center();
        rig.drag_to(at, red);
        let f = rig.press(red, false);
        assert_eq!((f.chosen, rig.value), (Some("Red"), 1), "releasing on Red chose it");
        assert!(!rig.frame(Vec::new()).open, "and closed the list");
    }

    #[test]
    fn a_click_on_a_combo_box_leaves_its_list_open_for_a_second_click() {
        let mut rig = ComboRig::new();
        let at = rig.button.center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        let f = rig.press(at, false);
        assert!(f.open && f.chosen.is_none(), "the click's release keeps the list drawn and chooses nothing");
        assert!(rig.frame(Vec::new()).open, "the list stays open");
        let green = rig.options[2].center();
        rig.frame(vec![Event::PointerMoved(green)]);
        rig.press(green, true);
        assert_eq!(rig.press(green, false).chosen, Some("Green"), "a second click chooses");
        assert_eq!(rig.value, 2);
        assert!(!rig.frame(Vec::new()).open);
    }

    #[test]
    fn releasing_off_a_combo_box_list_chooses_nothing() {
        let mut rig = ComboRig::new();
        let at = rig.button.center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        rig.frame(Vec::new());
        rig.frame(Vec::new());
        let away = pos2(380.0, 380.0);
        rig.drag_to(at, away);
        let f = rig.press(away, false);
        assert!(f.chosen.is_none() && rig.value == 0, "a release off the options chooses nothing");
        assert!(rig.frame(Vec::new()).open, "and keeps the list open");
    }

    #[test]
    fn clicking_a_combo_box_again_closes_its_list() {
        let mut rig = ComboRig::new();
        let at = rig.button.center();
        rig.frame(vec![Event::PointerMoved(at)]);
        rig.press(at, true);
        rig.press(at, false);
        assert!(rig.frame(Vec::new()).open);
        rig.press(at, true);
        rig.press(at, false);
        assert!(!rig.frame(Vec::new()).open, "a second click on the box closes the list");
        assert_eq!(rig.value, 0);
    }
}
