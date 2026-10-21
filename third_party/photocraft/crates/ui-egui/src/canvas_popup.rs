//! Outside-click dismissal shared by canvas menus and the Brush Preset picker.

use egui::{Context, Id, Rect};

pub(crate) fn mark_open(ctx: &Context, menu: Id) {
    let frame = ctx.cumulative_frame_nr();
    ctx.data_mut(|d| d.insert_temp(menu.with("opening-frame"), frame));
}

pub(crate) fn outside_pressed(ctx: &Context, menu: Id, rect: Rect) -> bool {
    // A quick tap's opening press can fall outside the menu after physical-pixel rounding.
    // Use the frame, rather than the pass, so a layout retry cannot dismiss it either.
    let opened = ctx.data(|d| d.get_temp::<u64>(menu.with("opening-frame")));
    opened != Some(ctx.cumulative_frame_nr()) && ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, Modifiers, PointerButton, RawInput, pos2, vec2};

    fn outside_press() -> RawInput {
        RawInput {
            events: vec![
                Event::PointerMoved(pos2(5.75, 5.75)),
                Event::PointerButton { pos: pos2(5.75, 5.75), button: PointerButton::Secondary, pressed: true, modifiers: Modifiers::NONE },
                Event::PointerButton { pos: pos2(5.75, 5.75), button: PointerButton::Secondary, pressed: false, modifiers: Modifiers::NONE },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn opening_frame_survives_layout_retries_and_later_presses_close() {
        let ctx = Context::default();
        let id = Id::new("menu");
        let rect = Rect::from_min_size(pos2(6.0, 6.0), vec2(100.0, 100.0));
        for _ in 0..3 {
            let mut passes = 0;
            let mut opening_pointer = None;
            let mut decisions = Vec::new();
            ctx.run_ui(outside_press(), |ui| {
                let ctx = ui.ctx();
                if passes == 0 {
                    opening_pointer = Some(ctx.input(|i| i.pointer.clone()));
                    mark_open(ctx, id);
                    ctx.request_discard("test the menu's extra layout pass");
                } else {
                    // egui consumes events before retrying. Replay the opening press so this
                    // assertion tests the frame guard, rather than the absence of input.
                    ctx.input_mut(|i| i.pointer = opening_pointer.clone().unwrap());
                }
                let outside_input = ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p)));
                decisions.push((outside_input, outside_pressed(ctx, id, rect)));
                passes += 1;
            })
            .drop_without_applying_deltas();
            assert_eq!(passes, 2);
            assert_eq!(decisions, [(true, false), (true, false)], "both passes must protect an outside opening press");
            let mut later_outside = false;
            ctx.run_ui(outside_press(), |ui| later_outside = outside_pressed(ui.ctx(), id, rect)).drop_without_applying_deltas();
            assert!(later_outside, "a later frame's outside press still dismisses the menu");
        }
    }

    #[test]
    fn opening_one_menu_does_not_protect_another() {
        let ctx = Context::default();
        let first = Id::new("first");
        let second = Id::new("second");
        let rect = Rect::from_min_size(pos2(6.0, 6.0), vec2(100.0, 100.0));
        ctx.run_ui(RawInput::default(), |ui| mark_open(ui.ctx(), first)).drop_without_applying_deltas();
        ctx.run_ui(outside_press(), |ui| {
            let ctx = ui.ctx();
            mark_open(ctx, second);
            assert!(!outside_pressed(ctx, second, rect));
            assert!(outside_pressed(ctx, first, rect));
        })
        .drop_without_applying_deltas();
    }
}
