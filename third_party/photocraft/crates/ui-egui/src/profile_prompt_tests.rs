//! Stacked dialogs of the same size in the same place (Embedded Profile Mismatch prompts used to
//! stack like this when several files opened at once): each can be answered with a real click,
//! one after another, even when an older one has been raised above the live one.

use crate::{PhotocraftApp, Services};
use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::{
    Harness,
    kittest::{NodeT, Queryable},
};
use photocraft_engine::Session;

fn harness(background: bool) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1400.0, 900.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(
            Session::new(),
            Services {
                import: Some(Box::new(|name, bytes, depth| {
                    photocraft_io::import_with_svg_group_depth(name, bytes, depth).map(|r| (r.document, r.warnings)).map_err(|e| e.to_string())
                })),
                ..Default::default()
            },
        )
    });
    h.state_mut().background_jobs = background;
    h.run_steps(3);
    h
}

/// Stack `n` identical Embedded Profile Mismatch dialogs (same size, same place; only the newest
/// is live) and wait until they are up and laid out. Opening files no longer stacks them (they ask
/// one at a time, `profile_open_tests`), so they are opened directly.
fn open_together(h: &mut Harness<'static, PhotocraftApp>, n: usize) {
    let message = "The document “shot.png” has an embedded color profile that does not match the current RGB working space.";
    for _ in 0..n {
        crate::prefs_ui::open_mismatch(h.state_mut(), message, "preserve", false, "__stacked");
    }
    for _ in 0..100 {
        h.run_steps(2);
        if h.state().ui.dialogs.len() >= n {
            h.run_steps(6);
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("{n} prompts, got {}", h.state().ui.dialogs.len());
}

fn press(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, pressed: bool) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
}

/// A real click (pointer events, hit-tested like a mouse) rather than an accessibility action.
fn click_at(h: &mut Harness<'static, PhotocraftApp>, p: Pos2) {
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    for pressed in [true, false] {
        press(h, p, pressed);
        h.run_steps(1);
    }
    h.run_steps(3);
}

/// Where the live prompt's `button` is: the one prompt that isn't greyed out.
fn live_button(h: &Harness<'static, PhotocraftApp>, button: &str) -> Pos2 {
    let live: Vec<_> = h.query_all_by_label(button).filter(|n| !n.accesskit_node().is_disabled()).map(|n| n.rect().center()).collect();
    assert_eq!(live.len(), 1, "exactly one prompt is live, {} open", h.state().ui.dialogs.len());
    live[0]
}

#[test]
fn prompts_for_files_opened_together_answer_one_at_a_time_with_real_clicks() {
    for background in [false, true] {
        for button in ["OK", "Cancel"] {
            let mut h = harness(background);
            open_together(&mut h, 3);
            for left in (0..3).rev() {
                let at = live_button(&h, button);
                click_at(&mut h, at);
                assert_eq!(h.state().ui.dialogs.len(), left, "background {background}: {button} answers the prompt in front");
            }
        }
    }
}

#[test]
fn an_older_prompt_raised_in_front_gives_way_to_the_live_one() {
    // Only the newest dialog takes input; egui only lets the modal drawn on top take input. When
    // egui raised an older prompt (it raises an area that is pressed or wasn't drawn the frame
    // before), the one in front was greyed out and the live one was unreachable behind it: no
    // click or key could answer either, so the app had to be quit (three screenshots opened
    // together from File › Open).
    for background in [false, true] {
        let mut h = harness(background);
        open_together(&mut h, 3);
        let ok = live_button(&h, "OK");
        let oldest = h.state().ui.dialogs[0].id;
        h.ctx.memory_mut(|m| m.areas_mut().move_to_top(egui::LayerId::new(egui::Order::Foreground, egui::Id::new(("dialog", oldest)))));
        h.run_steps(4);
        // All three are the same size in the same place: a click there must reach the live one.
        for left in [2, 1, 0] {
            click_at(&mut h, ok);
            assert_eq!(h.state().ui.dialogs.len(), left, "background {background}: OK on the prompt in front answers it");
        }
    }
}
