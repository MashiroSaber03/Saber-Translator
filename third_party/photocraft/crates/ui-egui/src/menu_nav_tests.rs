//! Menus on small displays (#138): every row of the longest menus is reachable by keyboard, wheel
//! and the scroll arrows, and the keyboard drives submenus and runs commands.

use super::*;
use crate::PhotocraftApp;
use egui_kittest::{Harness, kittest::Queryable};
use serde_json::json;

/// Window size in pixels and UI scale: the small displays of #138.
const DISPLAYS: [(f32, f32, f32); 4] = [(1280.0, 720.0, 1.0), (1024.0, 600.0, 1.0), (1280.0, 720.0, 1.5), (1024.0, 600.0, 1.5)];
/// The tallest menus.
const LONGEST: [&str; 4] = ["Filter", "Layer", "Image", "Edit"];

/// #3737: on Wayland egui knows the monitor's size but never where the window is (no
/// `inner_rect`). A window taller than the monitor less its taskbar or panel runs under it, so
/// Layer, View and Window (and every other long menu) must end above it and scroll to their last
/// row, at 1× and 1.5×.
#[test]
fn reported_menus_stay_above_the_panel_when_the_window_position_is_unknown() {
    for scale in [1.0, 1.5] {
        for top in ["Layer", "View", "Window", "Filter", "Image", "Edit"] {
            let mut h = harness((1366.0 * scale, 740.0 * scale, scale));
            let v = h.input_mut().viewports.entry(egui::ViewportId::ROOT).or_default();
            v.monitor_size = Some(egui::vec2(1366.0, 768.0));
            v.inner_rect = None;
            v.outer_rect = None;
            v.maximized = Some(false);
            h.run_steps(3);
            let bottom = 768.0 - crate::work_area::TASKBAR;
            let what = format!("{scale}x {top}");
            open(&mut h, top);
            for r in popups(&h) {
                assert!(r.bottom() <= bottom + 0.5, "{what}: popup {r:?} runs under the panel (from {bottom})");
            }
            check_bar_clear(&h, &what);
            let last = rows(&h, 0).len() - 1;
            let in_view = |h: &Harness<'static, PhotocraftApp>| {
                let nav = Nav::current(&h.ctx);
                nav.views[0].expand(0.5).contains_rect(nav.rows[0][last].rect) && nav.views[0].bottom() <= bottom + 0.5
            };
            let view = Nav::current(&h.ctx).views[0];
            h.hover_at(view.center());
            for _ in 0..80 {
                if in_view(&h) {
                    break;
                }
                h.event(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, -60.0),
                    modifiers: egui::Modifiers::NONE,
                    phase: egui::TouchPhase::Move,
                });
                h.run_steps(2);
            }
            assert!(in_view(&h), "{what}: the wheel reaches the last row, above the panel");
        }
    }
}

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
    app.sync_views();
    app
}

fn harness((w, h, scale): (f32, f32, f32)) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(egui::vec2(w / scale, h / scale)).with_pixels_per_point(scale).build_ui_state(
        |ui, app| {
            crate::menus::menu_bar(app, ui);
        },
        app(),
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.ctx.all_styles_mut(|s| s.scroll_animation = egui::style::ScrollAnimation::none());
    h.run_steps(3);
    h
}

fn open(h: &mut Harness<'static, PhotocraftApp>, top: &str) {
    h.get_by_label(top).click();
    h.run_steps(4);
    assert!(Nav::current(&h.ctx).rows.first().is_some_and(|r| !r.is_empty()), "{top} did not open");
}

/// Is row `i` of `level` entirely inside its level's visible (scrolled) area, on screen?
fn visible(h: &Harness<'static, PhotocraftApp>, level: usize, i: usize) -> bool {
    let nav = Nav::current(&h.ctx);
    let (Some(row), Some(view)) = (nav.rows.get(level).and_then(|r| r.get(i)), nav.views.get(level)) else { return false };
    view.expand(0.5).contains_rect(row.rect) && h.ctx.content_rect().expand(0.5).contains_rect(*view)
}

fn rows(h: &Harness<'static, PhotocraftApp>, level: usize) -> Vec<Row> {
    Nav::current(&h.ctx).rows.get(level).cloned().unwrap_or_default()
}

fn key(h: &mut Harness<'static, PhotocraftApp>, k: egui::Key) {
    h.key_press(k);
    h.run_steps(3);
}

#[test]
fn every_row_of_the_longest_menus_is_reachable_with_the_arrow_keys() {
    for display in DISPLAYS {
        for top in LONGEST {
            let mut h = harness(display);
            open(&mut h, top);
            let all = rows(&h, 0);
            let enabled: Vec<usize> = all.iter().enumerate().filter(|(_, r)| r.enabled).map(|(i, _)| i).collect();
            assert!(enabled.len() > 5, "{top}: {} enabled rows", enabled.len());
            let mut seen = Vec::new();
            for _ in 0..enabled.len() {
                key(&mut h, egui::Key::ArrowDown);
                let i = Nav::current(&h.ctx).highlighted(0).expect("a highlighted row");
                assert!(
                    visible(&h, 0, i),
                    "{display:?} {top}: highlighted row {i} ({:?}) is scrolled out of view",
                    rows(&h, 0).get(i).map(|r| r.command.clone())
                );
                seen.push(i);
            }
            seen.sort_unstable();
            assert_eq!(seen, enabled, "{display:?} {top}: ↓ visits every enabled row once");
            // ↓ wraps to the first row (scrolling back up); ↑ wraps to the last.
            key(&mut h, egui::Key::ArrowDown);
            assert_eq!(Nav::current(&h.ctx).highlighted(0), enabled.first().copied());
            assert!(visible(&h, 0, enabled[0]));
            key(&mut h, egui::Key::ArrowUp);
            let last = *enabled.last().unwrap();
            assert_eq!(Nav::current(&h.ctx).highlighted(0), Some(last));
            assert!(visible(&h, 0, last), "{display:?} {top}: the last row is reachable");
        }
    }
}

#[test]
fn small_displays_overflow_and_the_menu_stays_in_the_window() {
    // 1024 × 600 at 150 %: 683 × 400 points, far shorter than the Filter menu.
    let mut h = harness((1024.0, 600.0, 1.5));
    open(&mut h, "Filter");
    let all = rows(&h, 0);
    let last = all.len() - 1;
    assert!(!visible(&h, 0, last), "the Filter menu must overflow at 683 × 400 pt for this test to mean anything");
    let nav = Nav::current(&h.ctx);
    let screen = h.ctx.content_rect();
    assert!(nav.views[0].bottom() <= screen.bottom() && nav.views[0].top() >= screen.top());
    let bar = h.get_by_label("Filter").rect();
    assert!(nav.views[0].top() >= bar.bottom(), "the menu hangs below the menu bar instead of sliding over it: {:?} vs {bar:?}", nav.views[0]);
    // Scroll arrows show when the rows overflow.
    assert!(h.query_by_label("Scroll menu down").is_some() && h.query_by_label("Scroll menu up").is_some());
}

#[test]
fn the_mouse_wheel_scrolls_a_long_menu() {
    for display in DISPLAYS {
        let mut h = harness(display);
        open(&mut h, "Filter");
        let last = rows(&h, 0).len() - 1;
        let view = Nav::current(&h.ctx).views[0];
        h.hover_at(view.center());
        h.run_steps(2);
        for _ in 0..60 {
            if visible(&h, 0, last) {
                break;
            }
            h.event(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -60.0),
                modifiers: egui::Modifiers::NONE,
                phase: egui::TouchPhase::Move,
            });
            h.run_steps(2);
        }
        assert!(visible(&h, 0, last), "{display:?}: the wheel reaches the last Filter row");
        assert!(rows(&h, 0).len() == last + 1, "the menu stays open while scrolling");
    }
}

#[test]
fn resting_on_the_scroll_arrows_scrolls() {
    let mut h = harness((1024.0, 600.0, 1.5));
    open(&mut h, "Filter");
    let last = rows(&h, 0).len() - 1;
    assert!(visible(&h, 0, 0) && !visible(&h, 0, last));
    let view = Nav::current(&h.ctx).views[0];
    h.hover_at(egui::pos2(view.center().x, view.bottom() + ARROW / 2.0));
    for _ in 0..240 {
        h.run_steps(1);
        if visible(&h, 0, last) {
            break;
        }
    }
    assert!(visible(&h, 0, last), "resting on ▼ scrolls to the end");
    assert!(!visible(&h, 0, 0));
    let view = Nav::current(&h.ctx).views[0];
    h.hover_at(egui::pos2(view.center().x, view.top() - ARROW / 2.0));
    for _ in 0..240 {
        h.run_steps(1);
        if visible(&h, 0, 0) {
            break;
        }
    }
    assert!(visible(&h, 0, 0), "resting on ▲ scrolls back to the top");
}

#[test]
fn arrows_open_and_close_submenus_and_enter_runs_a_command() {
    let mut h = harness((1024.0, 600.0, 1.5));
    open(&mut h, "Filter");
    // Walk down to the first submenu (Blur, Distort, …) and open it with →.
    for _ in 0..rows(&h, 0).len() {
        key(&mut h, egui::Key::ArrowDown);
        let nav = Nav::current(&h.ctx);
        if nav.highlighted(0).and_then(|i| nav.rows[0].get(i)).is_some_and(|r| r.command.is_none()) {
            break;
        }
    }
    key(&mut h, egui::Key::ArrowRight);
    h.run_steps(2);
    let nav = Nav::current(&h.ctx);
    assert_eq!(nav.level, 1, "→ moves into the submenu");
    assert!(nav.rows.get(1).is_some_and(|r| !r.is_empty()), "the submenu is open");
    let first = nav.rows[1].iter().position(|r| r.enabled);
    assert_eq!(nav.highlighted(1), first, "its first enabled row is highlighted");
    assert!(visible(&h, 1, first.unwrap()));
    key(&mut h, egui::Key::ArrowDown);
    assert_eq!(Nav::current(&h.ctx).level, 1);
    // ← closes it again.
    key(&mut h, egui::Key::ArrowLeft);
    h.run_steps(2);
    let nav = Nav::current(&h.ctx);
    assert_eq!((nav.level, nav.rows.len()), (0, 1), "← closes the submenu");
    // ↩ runs the highlighted command and closes the menu: Select › All.
    key(&mut h, egui::Key::Escape);
    open(&mut h, "Select");
    for _ in 0..rows(&h, 0).len() {
        let nav = Nav::current(&h.ctx);
        if nav.highlighted(0).and_then(|i| nav.rows[0].get(i)).is_some_and(|r| r.command.as_deref() == Some("select.all")) {
            break;
        }
        key(&mut h, egui::Key::ArrowDown);
    }
    assert!(h.state().session.active().unwrap().doc.selection.is_none());
    key(&mut h, egui::Key::Enter);
    assert!(h.state().session.active().unwrap().doc.selection.is_some(), "↩ ran Select › All");
    assert!(!is_open(&h.ctx), "and closed the menu");
}

#[test]
fn left_and_right_switch_menus_and_shortcuts_wait_while_one_is_open() {
    let mut h = harness((1280.0, 720.0, 1.0));
    assert!(!is_open(&h.ctx));
    open(&mut h, "File");
    assert!(is_open(&h.ctx), "shortcuts leave the keys to an open menu");
    key(&mut h, egui::Key::ArrowRight);
    h.run_steps(2);
    assert!(h.query_by_label_contains("Open…").is_none(), "→ leaves File");
    assert!(h.query_by_label_contains("Undo").is_some(), "→ opens Edit");
    key(&mut h, egui::Key::ArrowLeft);
    h.run_steps(2);
    assert!(h.query_by_label_contains("Open…").is_some(), "← back to File");
    key(&mut h, egui::Key::Escape);
    h.run_steps(2);
    assert!(h.query_by_label_contains("Open…").is_none(), "Esc closes the menu");
    assert!(!is_open(&h.ctx), "shortcuts work again");
}

/// A level's visible height (its view) and the full height of its rows: equal means no scrolling.
fn fits(h: &Harness<'static, PhotocraftApp>, level: usize) -> (f32, f32) {
    let nav = Nav::current(&h.ctx);
    (nav.views.get(level).map_or(0.0, |v| v.height()), nav.contents.get(level).copied().unwrap_or(0.0))
}

/// The on-screen rect of the popup holding `level` (the smallest area around its view).
fn popup(h: &Harness<'static, PhotocraftApp>, level: usize) -> egui::Rect {
    let Some(view) = Nav::current(&h.ctx).views.get(level).copied() else { return egui::Rect::NOTHING };
    h.ctx
        .memory(|m| m.areas().visible_layer_ids())
        .into_iter()
        .filter_map(|layer| h.ctx.memory(|m| m.area_rect(layer.id)))
        .filter(|r| r.contains(view.center()) && r.width() < 600.0)
        .min_by(|a, b| a.area().total_cmp(&b.area()))
        .unwrap_or(egui::Rect::NOTHING)
}

/// The popup is as tall as its rows plus the menu frame: no blank space, no clipped rows.
fn hugs_rows(h: &Harness<'static, PhotocraftApp>, level: usize) -> Result<(), String> {
    let (_, content) = fits(h, level);
    let frame = h.ctx.global_style().spacing.menu_margin.sum().y + 2.0;
    let area = popup(h, level);
    ((area.height() - content - frame).abs() < 2.0).then_some(()).ok_or(format!("popup {} pt tall for {content} pt of rows + {frame} frame", area.height()))
}

/// Open the `n`th enabled submenu of the open top-level menu with the keyboard.
fn open_submenu(h: &mut Harness<'static, PhotocraftApp>, n: usize) -> bool {
    let subs: Vec<usize> = rows(h, 0).iter().enumerate().filter(|(_, r)| r.enabled && r.command.is_none()).map(|(i, _)| i).collect();
    let Some(&target) = subs.get(n) else { return false };
    for _ in 0..rows(h, 0).len() {
        if Nav::current(&h.ctx).highlighted(0) == Some(target) {
            break;
        }
        key(h, egui::Key::ArrowDown);
    }
    key(h, egui::Key::ArrowRight);
    h.run_steps(3);
    Nav::current(&h.ctx).rows.get(1).is_some_and(|r| !r.is_empty())
}

#[test]
fn menus_are_as_tall_as_their_rows_on_a_tall_window() {
    // #235: every menu opened at egui's 400 pt default area height and scrolled, even at 1440 × 900.
    let mut checked = 0;
    for top in crate::menus::TOP_MENUS {
        let mut h = harness((1440.0, 900.0, 1.0));
        open(&mut h, top);
        let (view, content) = fits(&h, 0);
        if content > h.ctx.content_rect().height() - 60.0 {
            continue; // Taller than the window itself: scrolling is right (see the short-window tests).
        }
        checked += 1;
        assert!((view - content).abs() < 1.0, "{top}: the menu is {view} pt tall for {content} pt of rows");
        if let Err(e) = hugs_rows(&h, 0) {
            panic!("{top}: {e}");
        }
        assert!(h.query_by_label("Scroll menu down").is_none(), "{top}: no scroll arrows when the rows fit");
        assert!((0..rows(&h, 0).len()).all(|i| visible(&h, 0, i)), "{top}: every row is on screen without scrolling");
    }
    assert!(checked >= 6, "most menus fit a 900 px window: {checked}");
    // The File menu is taller than the old 400 pt cap and fits at 1440 × 900.
    let mut h = harness((1440.0, 900.0, 1.0));
    open(&mut h, "File");
    let (view, content) = fits(&h, 0);
    assert!(content > 420.0 && (view - content).abs() < 1.0, "File: {view} pt view for {content} pt of rows");
    assert_eq!(hugs_rows(&h, 0), Ok(()));
}

#[test]
fn submenus_are_as_tall_as_their_rows_on_a_tall_window() {
    let mut checked = 0;
    for top in ["Filter", "Image", "Layer"] {
        for n in 0..3 {
            let mut h = harness((1440.0, 900.0, 1.0));
            open(&mut h, top);
            if !open_submenu(&mut h, n) {
                continue;
            }
            checked += 1;
            let (view, content) = fits(&h, 1);
            assert!(content > 0.0 && (view - content).abs() < 1.0, "{top} submenu {n}: {view} pt view for {content} pt of rows");
            if let Err(e) = hugs_rows(&h, 1) {
                panic!("{top} submenu {n}: {e}");
            }
        }
    }
    assert!(checked >= 6, "opened {checked} submenus");
}

#[test]
fn a_short_window_still_scrolls_the_file_menu_to_its_last_row() {
    let mut h = harness((1000.0, 500.0, 1.0));
    open(&mut h, "File");
    let (view, content) = fits(&h, 0);
    assert!(view < content - 1.0, "File must overflow a 500 px window: {view} vs {content}");
    let nav = Nav::current(&h.ctx);
    assert!(nav.views[0].bottom() <= h.ctx.content_rect().bottom(), "the menu stays in the window");
    assert!(h.query_by_label("Scroll menu down").is_some());
    let last = rows(&h, 0).len() - 1;
    assert!(!visible(&h, 0, last));
    h.hover_at(nav.views[0].center());
    h.run_steps(2);
    for _ in 0..60 {
        if visible(&h, 0, last) {
            break;
        }
        h.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -60.0),
            modifiers: egui::Modifiers::NONE,
            phase: egui::TouchPhase::Move,
        });
        h.run_steps(2);
    }
    assert!(visible(&h, 0, last), "the wheel reaches File's last row");
}

/// Every popup on screen (menus and submenus).
fn popups(h: &Harness<'static, PhotocraftApp>) -> Vec<egui::Rect> {
    h.ctx
        .memory(|m| m.areas().visible_layer_ids())
        .into_iter()
        .filter(|l| l.order == egui::Order::Foreground)
        .filter_map(|layer| h.ctx.memory(|m| m.area_rect(layer.id)))
        .collect()
}

/// No popup covers the menu bar or runs off the window, and every menu title stays visible.
fn check_bar_clear(h: &Harness<'static, PhotocraftApp>, what: &str) {
    let bar = Nav::current(&h.ctx).bar_bottom.expect("the menu bar's bottom");
    let screen = h.ctx.content_rect();
    for r in popups(h) {
        assert!(r.top() >= bar - 0.5, "{what}: popup {r:?} covers the menu bar (bottom {bar})");
        assert!(r.bottom() <= screen.bottom() + 0.5, "{what}: popup {r:?} runs off the window {screen:?}");
    }
    for title in crate::menus::TOP_MENUS {
        // The title is the topmost node of that name (a submenu row can share it).
        let t = h.query_all_by_label(title).map(|n| n.rect()).min_by(|a, b| a.top().total_cmp(&b.top())).expect("menu title");
        assert!(popups(h).iter().all(|r| !r.intersects(t.shrink(1.0))), "{what}: {title} is covered");
    }
}

#[test]
fn no_menu_or_submenu_covers_the_menu_bar() {
    // #319: on a 1280 × 720 display (the reporter's, at 1× and 2×, less the title bar) tall menus,
    // then tall submenus (Image › Adjustments), slid up over the menu bar and hid its titles.
    let displays = [(1280.0, 703.0, 1.0), (2560.0, 1406.0, 2.0), (1366.0, 768.0, 1.0)];
    let mut checked = 0;
    for display in displays {
        for top in LONGEST {
            let mut h = harness(display);
            open(&mut h, top);
            check_bar_clear(&h, &format!("{display:?} {top}"));
            // Hover each submenu row in view, top to bottom: rows low in the menu open their
            // submenus upward.
            let view = Nav::current(&h.ctx).views.first().copied().unwrap_or(egui::Rect::NOTHING);
            let subs: Vec<egui::Rect> = rows(&h, 0).iter().filter(|r| r.enabled && r.command.is_none() && view.contains_rect(r.rect)).map(|r| r.rect).collect();
            for (i, row) in subs.iter().enumerate() {
                h.hover_at(row.center() + egui::vec2(-20.0, 0.0));
                h.run_steps(6);
                if Nav::current(&h.ctx).rows.get(1).is_none_or(Vec::is_empty) {
                    continue;
                }
                checked += 1;
                check_bar_clear(&h, &format!("{display:?} {top} submenu {i}"));
            }
        }
    }
    assert!(checked > 30, "checked {checked} submenus");
}

#[test]
fn level_room_keeps_every_level_below_the_bar() {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 703.0));
    let (bar, frame) = (Some(30.0), 14.0);
    let below = 703.0 - EDGE - (30.0 + EDGE) - frame;
    assert_eq!(level_room(screen, bar, 1, None, frame), below);
    // A submenu from a row near the top opens downward with the room under it…
    let row = egui::Rect::from_min_size(egui::pos2(100.0, 80.0), egui::vec2(200.0, 30.0));
    assert_eq!(level_room(screen, bar, 2, Some(row), frame), 703.0 - EDGE - 80.0 - frame);
    // …and from a row near the bottom, upward to the bar.
    let low = egui::Rect::from_min_size(egui::pos2(100.0, 600.0), egui::vec2(200.0, 30.0));
    assert_eq!(level_room(screen, bar, 2, Some(low), frame), 630.0 - (30.0 + EDGE) - frame);
    // Never more than the space under the bar, never less than the scroll arrows need.
    let odd = egui::Rect::from_min_size(egui::pos2(100.0, -50.0), egui::vec2(200.0, 30.0));
    assert!(level_room(screen, bar, 2, Some(odd), frame) <= below);
    assert_eq!(level_room(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 40.0)), bar, 1, None, frame), 4.0 * ARROW);
    assert_eq!(level_room(screen, Some(f32::NAN), 1, None, frame), level_room(screen, None, 1, None, frame));
    assert_eq!(level_room(egui::Rect::NOTHING, bar, 2, Some(row), frame), 4.0 * ARROW);
    // A window running under the taskbar: a top-level menu leaves a scrolling submenu's room above
    // it, so the submenu of its last row (which egui opens downward) stays above the taskbar too.
    let visible = egui::Rect::from_min_max(screen.min, egui::pos2(screen.right(), 600.0));
    let top = level_room_in(screen, visible, bar, 1, None, None, frame);
    assert_eq!(top, 600.0 - EDGE - (30.0 + EDGE) - (4.0 * ARROW + frame) - frame);
    let last_row = egui::Rect::from_min_size(egui::pos2(100.0, 30.0 + EDGE + top - 24.0), egui::vec2(200.0, 24.0));
    // A 200 pt submenu fits the window downward, so egui opens it that way: it scrolls in what is left.
    let room = level_room_in(screen, visible, bar, 2, Some(last_row), Some(200.0), frame);
    assert!(room >= 4.0 * ARROW && last_row.top() - (frame - 2.0) / 2.0 + room + frame <= 600.0 - EDGE + 0.5, "room {room}");
}

/// #315: a 1366 × 768 pt display with a Windows taskbar. `window` is the window's content size and
/// `top` where its content starts on the desktop (below the title bar), in points; the taskbar
/// covers the monitor's bottom [`crate::work_area::TASKBAR`] points. `scale` is the UI scale.
fn taskbar_harness(window: (f32, f32), top: f32, scale: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = harness((window.0 * scale, window.1 * scale, scale));
    let v = h.input_mut().viewports.entry(egui::ViewportId::ROOT).or_default();
    v.monitor_size = Some(egui::vec2(1366.0, 768.0));
    v.inner_rect = Some(egui::Rect::from_min_size(egui::pos2(0.0, top), egui::vec2(window.0, window.1)));
    v.maximized = Some(false);
    h.run_steps(3);
    h
}

/// The visible bottom of a taskbar harness: the monitor's bottom less the taskbar, in the window.
fn taskbar_bottom(h: &Harness<'static, PhotocraftApp>) -> f32 {
    crate::work_area::visible_rect(&h.ctx).bottom()
}

#[test]
fn menus_stay_above_the_taskbar_and_scroll_to_their_last_row() {
    // The default 1440 × 900 window on a 1366 × 768 display runs past the screen; a 1366 × 737
    // window below a 31 pt title bar reaches under the taskbar. At 1× and 2× (a 2732 × 1536 px
    // display at 200%).
    let mut scrolled = 0;
    for (window, top, scale) in [((1440.0, 900.0), 0.0, 1.0), ((1366.0, 737.0), 31.0, 1.0), ((1366.0, 737.0), 31.0, 2.0)] {
        for menu in LONGEST {
            let mut h = taskbar_harness(window, top, scale);
            let bottom = taskbar_bottom(&h);
            let what = format!("{window:?}@{scale}x {menu}");
            assert!((bottom - (768.0 - crate::work_area::TASKBAR - top)).abs() < 1.0, "{what}: visible bottom {bottom}");
            open(&mut h, menu);
            for r in popups(&h) {
                assert!(r.bottom() <= bottom + 0.5, "{what}: popup {r:?} runs under the taskbar (from {bottom})");
            }
            let last = rows(&h, 0).len() - 1;
            let in_view = |h: &Harness<'static, PhotocraftApp>, i: usize| {
                let nav = Nav::current(&h.ctx);
                nav.views[0].expand(0.5).contains_rect(nav.rows[0][i].rect) && nav.views[0].bottom() <= bottom + 0.5
            };
            // A menu that fits above the taskbar needs no scrolling.
            let Some(down) = h.query_by_label("Scroll menu down").map(|n| n.rect()) else {
                assert!(in_view(&h, last), "{what}: fits, so its last row is in view");
                continue;
            };
            scrolled += 1;
            // The ▼ arrow is above the taskbar, and resting on it reaches the last row.
            assert!(down.bottom() <= bottom, "{what}: ▼ at {down:?}");
            assert!(!in_view(&h, last));
            h.hover_at(down.center());
            for _ in 0..400 {
                h.run_steps(1);
                if in_view(&h, last) {
                    break;
                }
            }
            assert!(in_view(&h, last), "{what}: ▼ scrolls to the last row, above the taskbar");
            // And the wheel scrolls back up.
            let view = Nav::current(&h.ctx).views[0];
            h.hover_at(view.center());
            for _ in 0..60 {
                h.event(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, 120.0),
                    modifiers: egui::Modifiers::NONE,
                    phase: egui::TouchPhase::Move,
                });
                h.run_steps(1);
                if in_view(&h, 0) {
                    break;
                }
            }
            assert!(in_view(&h, 0), "{what}: the wheel scrolls back to the first row");
        }
    }
    assert!(scrolled >= 6, "the long menus overflow above the taskbar: {scrolled}");
}

#[test]
fn submenus_stay_above_the_taskbar() {
    for menu in ["Image", "Layer", "Filter"] {
        let mut h = taskbar_harness((1440.0, 900.0), 0.0, 1.0);
        let bottom = taskbar_bottom(&h);
        open(&mut h, menu);
        let view = Nav::current(&h.ctx).views.first().copied().unwrap_or(egui::Rect::NOTHING);
        let subs: Vec<egui::Rect> = rows(&h, 0).iter().filter(|r| r.enabled && r.command.is_none() && view.contains_rect(r.rect)).map(|r| r.rect).collect();
        assert!(!subs.is_empty());
        for (i, row) in subs.iter().enumerate() {
            h.hover_at(row.center() + egui::vec2(-20.0, 0.0));
            h.run_steps(6);
            for r in popups(&h) {
                assert!(r.bottom() <= bottom + 0.5, "{menu} submenu {i}: popup {r:?} runs under the taskbar (from {bottom})");
            }
            check_bar_clear(&h, &format!("{menu} submenu {i}"));
        }
    }
}

/// #3737 on X11 (Linux Mint, 1920 × 1080 px): the window manager maximized the window to the whole
/// monitor without keeping it clear of the bottom panel, so the window's position is known but the
/// panel covers its bottom. At 1× and at a 150% UI scale (1280 × 720 pt).
fn maximized_under_a_panel(scale: f32) -> Harness<'static, PhotocraftApp> {
    let size = egui::vec2(1920.0, 1080.0) / scale;
    let mut h = harness((1920.0, 1080.0, scale));
    let v = h.input_mut().viewports.entry(egui::ViewportId::ROOT).or_default();
    v.monitor_size = Some(size);
    v.inner_rect = Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), size));
    v.maximized = Some(true);
    h.run_steps(3);
    h
}

#[test]
fn menus_of_a_maximized_window_stay_above_a_panel_it_runs_under() {
    let mut scrolled = 0;
    for scale in [1.0_f32, 1.5] {
        for menu in LONGEST.iter().copied().chain(["View", "Window"]) {
            let mut h = maximized_under_a_panel(scale);
            let bottom = taskbar_bottom(&h);
            let what = format!("{scale}x {menu}");
            assert!((bottom - (1080.0 / scale - crate::work_area::TASKBAR)).abs() < 1.0, "{what}: visible bottom {bottom}");
            open(&mut h, menu);
            for r in popups(&h) {
                assert!(r.bottom() <= bottom + 0.5, "{what}: popup {r:?} runs under the panel (from {bottom})");
            }
            let last = rows(&h, 0).len() - 1;
            let in_view = |h: &Harness<'static, PhotocraftApp>| {
                let nav = Nav::current(&h.ctx);
                nav.views[0].expand(0.5).contains_rect(nav.rows[0][last].rect) && nav.views[0].bottom() <= bottom + 0.5
            };
            let Some(down) = h.query_by_label("Scroll menu down").map(|n| n.rect()) else {
                assert!(in_view(&h), "{what}: fits, so its last row is in view");
                continue;
            };
            scrolled += 1;
            assert!(down.bottom() <= bottom, "{what}: ▼ at {down:?} is under the panel");
            h.hover_at(down.center());
            for _ in 0..400 {
                h.run_steps(1);
                if in_view(&h) {
                    break;
                }
            }
            assert!(in_view(&h), "{what}: ▼ scrolls to the last row, above the panel");
        }
    }
    assert!(scrolled >= 3, "at 150% the long menus overflow above the panel: {scrolled}");
}

#[test]
fn a_maximized_window_or_an_unknown_monitor_uses_the_whole_window() {
    // Maximized inside the work area (the window manager kept it clear of the taskbar).
    let mut h = taskbar_harness((1366.0, 720.0), 0.0, 1.0);
    h.input_mut().viewports.entry(egui::ViewportId::ROOT).or_default().maximized = Some(true);
    h.run_steps(2);
    assert_eq!(crate::work_area::visible_rect(&h.ctx), h.ctx.content_rect());
    let h = harness((1366.0, 768.0, 1.0));
    assert_eq!(crate::work_area::visible_rect(&h.ctx), h.ctx.content_rect());
}

/// One gesture: press a menu title (it opens on the press, not the release), drag down to an
/// item and release on it to run it. A plain click still opens the menu and leaves it open.
#[test]
fn press_drag_release_runs_a_menu_item() {
    use egui::{Modifiers, PointerButton};
    let mut h = harness((1280.0, 720.0, 1.0));
    let rulers = h.state().ui.extras.rulers;
    let title = h.get_by_label("View").rect().center();
    h.event(egui::Event::PointerMoved(title));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: title, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert!(Nav::current(&h.ctx).rows.first().is_some_and(|r| !r.is_empty()), "the menu opens on the press");
    let item = h.get_by_label_contains("Rulers").rect().center();
    for k in 1..=8 {
        h.event(egui::Event::PointerMoved(title + (item - title) * (k as f32 / 8.0)));
        h.run_steps(1);
    }
    h.event(egui::Event::PointerButton { pos: item, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    assert_eq!(h.state().ui.extras.rulers, !rulers, "releasing on Rulers ran it");
    assert!(Nav::current(&h.ctx).rows.first().is_none_or(|r| r.is_empty()), "and closed the menu");
    // A click (press and release on the title) opens the menu and leaves it open.
    h.get_by_label("View").click();
    h.run_steps(4);
    assert!(Nav::current(&h.ctx).rows.first().is_some_and(|r| !r.is_empty()), "a click leaves it open");
}

/// Windows and Linux menus too (#775): one press-drag-release gesture can move across titles and
/// into a submenu. Press File, drag over to View, down to Proof Setup, right into its submenu and
/// release on Working CMYK.
#[test]
fn press_drag_release_reaches_other_menus_and_submenus() {
    use egui::{Modifiers, PointerButton};
    let mut h = harness((1280.0, 720.0, 1.0));
    let mut at = h.get_by_label("File").rect().center();
    h.event(egui::Event::PointerMoved(at));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(2);
    let mut drag_to = |h: &mut Harness<'static, PhotocraftApp>, to: egui::Pos2| {
        for k in 1..=8 {
            h.event(egui::Event::PointerMoved(at + (to - at) * (k as f32 / 8.0)));
            h.run_steps(1);
        }
        h.run_steps(2);
        at = to;
    };
    let view = h.get_by_label("View").rect().center();
    drag_to(&mut h, view);
    assert!(h.query_by_label_contains("Proof Setup").is_some(), "dragging over View switched to its menu");
    let proof = h.get_by_label_contains("Proof Setup").rect();
    drag_to(&mut h, egui::pos2(proof.left() + 30.0, proof.center().y));
    drag_to(&mut h, egui::pos2(proof.right() - 6.0, proof.center().y));
    let item = h.get_by_label_contains("Working CMYK").rect();
    drag_to(&mut h, egui::pos2(item.left() + 30.0, proof.center().y));
    drag_to(&mut h, item.center());
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    let proofing = crate::menus::menu_items(h.state()).into_iter().find(|i| i.id == "view.proofColors").and_then(|i| i.checked);
    assert_eq!(proofing, Some(true), "releasing on Working CMYK ran it");
    assert!(Nav::current(&h.ctx).rows.first().is_none_or(|r| r.is_empty()), "and closed the menus");
}

/// A menu bar whose frames are `dt` seconds apart, as with a real pointer sending moves.
fn harness_dt(dt: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 720.0)).with_step_dt(dt).build_ui_state(
        |ui, app| {
            crate::menus::menu_bar(app, ui);
        },
        app(),
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.ctx.all_styles_mut(|s| s.scroll_animation = egui::style::ScrollAnimation::none());
    h.run_steps(3);
    h
}

/// Which of Image › Mode and Image › Adjustments has its submenu open.
fn image_submenu(h: &Harness<'static, PhotocraftApp>) -> &'static str {
    match (h.query_by_label_contains("16 Bits/Channel").is_some(), h.query_by_label_contains("Brightness/Contrast").is_some()) {
        (true, false) => "Mode",
        (false, true) => "Adjustments",
        (false, false) => "none",
        (true, true) => "both",
    }
}

/// Open Image › Mode with the pointer resting on it, at 120 frames a second; returns where the
/// pointer is, the Adjustments row below it, and Mode's 16 Bits/Channel row.
fn rest_on_mode(h: &mut Harness<'static, PhotocraftApp>) -> (egui::Pos2, egui::Rect, egui::Rect) {
    open(h, "Image");
    let mode = h.get_by_label_contains("Mode").rect();
    let from = egui::pos2(mode.right() - 40.0, mode.center().y);
    for k in 0..=12 {
        h.event(egui::Event::PointerMoved(from - egui::vec2(30.0 * (1.0 - k as f32 / 12.0), 0.0)));
        h.run_steps(1);
    }
    h.run_steps(30);
    assert_eq!(image_submenu(h), "Mode", "resting on Mode opens it");
    (from, h.get_by_label_contains("Adjustments").rect(), h.get_by_label_contains("16 Bits/Channel").rect())
}

/// Move the pointer along `path` (0…1) in `frames` frames; `pause` holds it still for that many
/// frames at that fraction of the way.
fn glide(h: &mut Harness<'static, PhotocraftApp>, frames: usize, pause: Option<(f32, usize)>, mut path: impl FnMut(f32) -> egui::Pos2) {
    let mut paused = false;
    for k in 0..=frames {
        let t = k as f32 / frames as f32;
        h.event(egui::Event::PointerMoved(path(t)));
        h.run_steps(1);
        if let Some((_, n)) = pause.filter(|(at, _)| !paused && t >= *at) {
            paused = true;
            h.run_steps(n);
        }
    }
}

/// A person's path from `a` to `b`: it bows, down first and then across, by `bow`.
fn bowed(a: egui::Pos2, b: egui::Pos2, bow: f32) -> impl Fn(f32) -> egui::Pos2 {
    let c = egui::pos2(a.x + (b.x - a.x) * (0.5 - bow), a.y + (b.y - a.y) * (0.5 + bow));
    move |t| {
        let u = 1.0 - t;
        egui::pos2(u * u * a.x + 2.0 * u * t * c.x + t * t * b.x, u * u * a.y + 2.0 * u * t * c.y + t * t * b.y)
    }
}

/// #1218: heading from Image › Mode for a row low in its submenu, the pointer crosses the
/// Adjustments row. On a diagonal, a bowed path or with a short pause on the way, Mode's submenu
/// stays open and Adjustments' doesn't open; the crossed row isn't highlighted.
#[test]
fn a_submenu_stays_open_while_the_pointer_heads_for_it() {
    struct Path {
        name: &'static str,
        /// Frames at 120 a second.
        frames: usize,
        /// Hold still this many frames at this fraction of the way.
        pause: Option<(f32, usize)>,
        bow: f32,
    }
    let paths = [
        Path { name: "straight", frames: 36, pause: None, bow: 0.0 },
        Path { name: "hesitate", frames: 36, pause: Some((0.3, 18)), bow: 0.0 },
        Path { name: "bowed", frames: 42, pause: None, bow: 0.2 },
        Path { name: "slow and bowed", frames: 96, pause: None, bow: 0.2 },
    ];
    let mut wrong = Vec::new();
    for Path { name, frames, pause, bow } in paths {
        let mut h = harness_dt(1.0 / 120.0);
        let (from, adjustments, target) = rest_on_mode(&mut h);
        let to = egui::pos2(target.left() + 40.0, target.center().y);
        let path = bowed(from, to, bow);
        let mode_row = rows(&h, 0).iter().position(|r| r.command.is_none() && r.rect.contains(from)).expect("the Mode row");
        let mut crossed = false;
        glide(&mut h, frames, pause, |t| {
            let p = path(t);
            crossed |= adjustments.contains(p);
            p
        });
        assert!(crossed, "{name}: the path crosses the Adjustments row");
        // Once Mode's submenu closes on the way, the pointer arrives over Adjustments' instead.
        let open = image_submenu(&h);
        if open != "Mode" {
            wrong.push(format!("{name}: arrived with {open} open, not Mode"));
        }
        if let Some(row) = Nav::current(&h.ctx).highlighted(0).filter(|&row| row != mode_row) {
            wrong.push(format!("{name}: row {row} is highlighted, not Mode"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Resting on the row crossed, or moving away from the submenu onto it, still opens its submenu,
/// and the highlight moves to it (only it is lit).
#[test]
fn resting_on_another_row_or_moving_away_switches_submenus() {
    let only_adjustments_lit = |h: &Harness<'static, PhotocraftApp>, adjustments: egui::Rect, what: &str| {
        let row = rows(h, 0).iter().position(|r| r.rect.contains(adjustments.center()));
        assert_eq!(Nav::current(&h.ctx).highlighted(0), row, "{what}: Adjustments is the highlighted row, not Mode");
    };
    let mut h = harness_dt(1.0 / 120.0);
    let (from, adjustments, _) = rest_on_mode(&mut h);
    // Head for the submenu, but stop on Adjustments on the way.
    let stop = egui::pos2(from.x + 15.0, adjustments.center().y);
    glide(&mut h, 12, None, |t| from + (stop - from) * t);
    assert_eq!(image_submenu(&h), "Mode", "just arrived on Adjustments, still heading for Mode's submenu");
    h.run_steps((super::AIM_REST * 120.0) as usize + 12);
    assert_eq!(image_submenu(&h), "Adjustments", "resting on Adjustments opens it");
    only_adjustments_lit(&h, adjustments, "resting");

    // Moving away from Mode's submenu switches (as soon as egui does): straight down, or down and
    // to the left.
    for dx in [0.0, -60.0] {
        let mut h = harness_dt(1.0 / 120.0);
        let (from, adjustments, _) = rest_on_mode(&mut h);
        let to = egui::pos2(from.x + dx, adjustments.center().y);
        glide(&mut h, 10, None, |t| from + (to - from) * t);
        h.run_steps(30);
        assert_eq!(image_submenu(&h), "Adjustments", "moving away ({dx}) from Mode's submenu opens Adjustments'");
        only_adjustments_lit(&h, adjustments, "moving away");
    }
}

#[test]
fn heading_is_the_triangle_to_the_submenus_near_edge() {
    let sub = egui::Rect::from_min_max(egui::pos2(200.0, 100.0), egui::pos2(400.0, 400.0));
    let from = egui::pos2(150.0, 110.0);
    assert!(heading(from, from, sub));
    assert!(heading(from, egui::pos2(170.0, 160.0), sub), "down and right, toward the submenu");
    assert!(heading(from, egui::pos2(190.0, 105.0), sub), "straight across");
    assert!(!heading(from, egui::pos2(150.0, 160.0), sub), "straight down");
    assert!(!heading(from, egui::pos2(120.0, 140.0), sub), "away to the left");
    assert!(!heading(from, egui::pos2(170.0, 60.0), sub), "up past the submenu's top");
    // A submenu that opened to the left.
    let left = egui::Rect::from_min_max(egui::pos2(0.0, 100.0), egui::pos2(100.0, 400.0));
    assert!(heading(egui::pos2(150.0, 110.0), egui::pos2(130.0, 160.0), left));
    assert!(!heading(egui::pos2(150.0, 110.0), egui::pos2(170.0, 160.0), left));
    assert!(!heading(from, from, egui::Rect::NOTHING));
}
