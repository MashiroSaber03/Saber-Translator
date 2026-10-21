//! Drag-selecting in a single-line text field follows only the pointer's x, as in Photoshop and
//! native text fields: the pointer can drift above or below the field without the selection
//! jumping to the start or the end of the text (#1946).
//!
//! egui's `TextEdit` picks the character under the pointer with `Galley::cursor_from_pos`, which
//! returns the start of the text when the pointer is more than a few points above the first row
//! and the end when it is below the last one. That is right for a multi-line field and wrong for a
//! single-line one, and `TextEdit` doesn't tell them apart. So while the pointer is pressed in the
//! focused single-line text field, this plugin moves the pointer's y onto that field's text row
//! before egui sees the input; the x, and so the selection, is left alone. Every single-line
//! field goes through it, including number fields being typed into (a `DragValue` edits its text
//! with a single-line `TextEdit`). egui keeps no record of which kind a field is, so a field is
//! taken as single-line by its height: no taller than two widget heights (a multi-line field
//! shows four rows unless told otherwise).

use egui::{Context, Event, Pos2, RawInput, Rect};

/// Installed by `PhotocraftApp::setup_context`.
#[derive(Default)]
pub struct SingleLineDrag {
    /// The y the pointer is kept on during the next input, found at the end of the last pass.
    row_y: Option<f32>,
}

impl egui::plugin::Plugin for SingleLineDrag {
    fn debug_name(&self) -> &'static str {
        "PhotoCraft single-line text drag"
    }

    fn on_end_pass(&mut self, ui: &mut egui::Ui) {
        self.row_y = dragged_row_y(ui.ctx());
    }

    fn input_hook(&mut self, _ctx: &Context, input: &mut RawInput) {
        let Some(row_y) = self.row_y else {
            return;
        };
        for event in &mut input.events {
            if let Event::PointerMoved(pos) | Event::PointerButton { pos, .. } = event {
                *pos = Pos2::new(pos.x, row_y);
            }
        }
    }
}

/// The y to keep the pointer on: on the text row of the focused single-line text field the
/// primary button was pressed in, while it is still held. `None` when no such drag is going on.
/// Read at the end of a pass, when the field's rect is this pass's.
fn dragged_row_y(ctx: &Context) -> Option<f32> {
    let (down, origin) = ctx.input(|i| (i.pointer.primary_down(), i.pointer.press_origin()));
    let origin = origin.filter(|_| down)?;
    let id = ctx.memory(|m| m.focused())?;
    egui::TextEdit::load_state(ctx, id)?;
    let rect = ctx.read_response(id)?.rect;
    if !rect.contains(origin) {
        return None;
    }
    row_y(rect, ctx.global_style().spacing.interact_size.y)
}

/// A y on the text row of a single-line field laid out in `rect`, or `None` when `rect` is too
/// tall to be one (a multi-line field, whose vertical drag must keep working).
///
/// A single-line `TextEdit` puts its row at the top of its rect (a small margin below the top
/// edge), or centred when the field asks for it (number fields); `interact` is the theme's widget
/// height, which a single-line field is never much taller than. A point half a widget height
/// below the top, or the centre of a shorter rect, is on the row in both cases, and egui accepts a
/// few points of slack either way.
fn row_y(rect: Rect, interact: f32) -> Option<f32> {
    let (h, interact) = (rect.height(), interact.max(1.0));
    if !(h.is_finite() && interact.is_finite()) || h <= 0.0 || h > 2.0 * interact {
        return None;
    }
    Some(rect.top() + (h / 2.0).min(interact / 2.0))
}

#[cfg(test)]
mod tests {
    use egui::{Context, Event, Id, Modifiers, PointerButton, Pos2, RawInput, Rect, pos2, vec2};

    const TEXT: &str = "abcdefghijklmnopqrst";
    const LINES: &str = "first line\nsecond line\nthird line";

    #[derive(Clone, Copy)]
    enum Field {
        SingleLine,
        MultiLine,
    }

    /// A context set up as the app sets it up (fonts, theme, plugins).
    fn ctx() -> Context {
        let ctx = Context::default();
        crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        ctx
    }

    /// Run one frame with `events` over a text field; returns its rect.
    fn frame(ctx: &Context, field: Field, id: Id, events: Vec<Event>) -> Rect {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))), events, ..Default::default() };
        let mut rect = Rect::NOTHING;
        let mut out = ctx.run_ui(input, |ui| {
            ui.add_space(100.0);
            rect = match field {
                Field::SingleLine => ui.add(egui::TextEdit::singleline(&mut TEXT.to_string()).id(id).desired_width(400.0)).rect,
                Field::MultiLine => ui.add(egui::TextEdit::multiline(&mut LINES.to_string()).id(id).desired_width(400.0)).rect,
            };
        });
        out.textures_delta.clear();
        rect
    }

    fn press(pos: Pos2, pressed: bool) -> Event {
        Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE }
    }

    /// Press at `from`, drag a little sideways, then to `to` and release; the selection as
    /// (anchor, end) character indices. Both points get the field's rect and the width of
    /// [`TEXT`].
    fn drag(field: Field, from: impl Fn(Rect, f32) -> Pos2, to: impl Fn(Rect, f32) -> Pos2) -> (usize, usize) {
        let ctx = ctx();
        let id = Id::new("single-line-drag-test");
        let rect = frame(&ctx, field, id, vec![]);
        frame(&ctx, field, id, vec![]);
        let text_w = text_width(&ctx);
        let (from, to) = (from(rect, text_w), to(rect, text_w));
        frame(&ctx, field, id, vec![Event::PointerMoved(from), press(from, true)]);
        frame(&ctx, field, id, vec![Event::PointerMoved(from + vec2(1.0, 0.0))]);
        let anchor = selection(&ctx, id).0;
        frame(&ctx, field, id, vec![Event::PointerMoved(from + vec2(12.0, 0.0))]);
        frame(&ctx, field, id, vec![Event::PointerMoved(to)]);
        frame(&ctx, field, id, vec![Event::PointerMoved(to)]);
        let during = selection(&ctx, id);
        frame(&ctx, field, id, vec![press(to, false)]);
        let after = selection(&ctx, id);
        assert_eq!(during, after, "releasing keeps the selection made by the drag");
        assert_eq!(after.0, anchor, "the anchor stays where the drag started");
        after
    }

    fn text_width(ctx: &Context) -> f32 {
        let font = egui::TextStyle::Body.resolve(&ctx.global_style());
        ctx.fonts_mut(|f| f.layout_no_wrap(TEXT.to_string(), font, egui::Color32::WHITE).size().x)
    }

    fn selection(ctx: &Context, id: Id) -> (usize, usize) {
        let state = egui::TextEdit::load_state(ctx, id).expect("the field stored its state");
        let range = state.cursor.char_range().expect("the field has a cursor");
        (range.secondary.index.into(), range.primary.index.into())
    }

    /// `frac` of the way along [`TEXT`] on the row of a single-line field.
    fn on_row(frac: f32) -> impl Fn(Rect, f32) -> Pos2 {
        move |r, w| pos2(r.left() + 4.0 + frac * w, r.center().y)
    }

    #[test]
    fn dragging_up_and_right_selects_right_of_the_anchor() {
        let n = TEXT.chars().count();
        // Up out of the field, to three quarters of the text.
        let (anchor, end) = drag(Field::SingleLine, on_row(0.5), |r, w| pos2(r.left() + 4.0 + 0.75 * w, r.top() - 40.0));
        assert!(anchor > 0 && anchor < n, "anchor {anchor} is mid-text");
        assert!(end > anchor && end < n, "selection ends right of the anchor at the pointer's x ({anchor}..{end}), not at the start");
        // Up and past the end of the text: everything right of the anchor.
        let (anchor, end) = drag(Field::SingleLine, on_row(0.5), |r, _| pos2(r.right() + 50.0, r.top() - 60.0));
        assert!(anchor > 0 && end == n, "{anchor}..{end}");
    }

    #[test]
    fn dragging_down_and_left_selects_left_of_the_anchor() {
        let n = TEXT.chars().count();
        let (anchor, end) = drag(Field::SingleLine, on_row(0.75), |r, w| pos2(r.left() + 4.0 + 0.25 * w, r.bottom() + 40.0));
        assert!(anchor > 0 && anchor < n, "anchor {anchor} is mid-text");
        assert!(end < anchor && end > 0, "selection ends left of the anchor at the pointer's x ({anchor}..{end}), not at the end");
        let (anchor, end) = drag(Field::SingleLine, on_row(0.75), |r, _| pos2(r.left() - 50.0, r.bottom() + 60.0));
        assert!(anchor > 0 && end == 0, "{anchor}..{end}");
    }

    #[test]
    fn a_multi_line_field_keeps_its_vertical_drag() {
        // Press on the last line, drag up above the field: the selection runs to the start.
        let last_line = |r: Rect, _| pos2(r.left() + 30.0, r.top() + r.height() * 0.6);
        let (anchor, end) = drag(Field::MultiLine, last_line, |r, _| pos2(r.left() + 30.0, r.top() - 40.0));
        assert!(anchor > "first line\nsecond line\n".len(), "anchor {anchor} is on the last line");
        assert_eq!(end, 0, "dragging above a multi-line field selects to its start");
        assert_eq!(super::row_y(Rect::from_min_size(Pos2::ZERO, vec2(200.0, 100.0)), 24.0), None);
    }

    #[test]
    fn the_row_is_inside_a_single_line_rect() {
        assert_eq!(super::row_y(Rect::from_min_size(pos2(0.0, 10.0), vec2(200.0, 20.0)), 24.0), Some(20.0));
        assert_eq!(super::row_y(Rect::from_min_size(pos2(0.0, 10.0), vec2(200.0, 40.0)), 24.0), Some(22.0));
        assert_eq!(super::row_y(Rect::NOTHING, 24.0), None);
        assert_eq!(super::row_y(Rect::from_min_size(Pos2::ZERO, vec2(10.0, f32::NAN)), 24.0), None);
    }
}
