//! The part of the window the user can actually see and reach (#315).
//!
//! A window can be larger than its display: PhotoCraft opens at 1440 × 900 pt, which runs past
//! the bottom of a 1366 × 768 screen and under the Windows taskbar. egui sizes and places menus
//! by the window's content rect, so their last items (and their ▼ scroll arrow) were out of
//! reach. egui doesn't expose the monitor's work area (the screen minus the taskbar or Dock), only
//! the monitor's size and the window's position on the desktop: the visible part is the content
//! rect cut to the monitor, keeping a taskbar-sized margin clear at the monitor's bottom.
//!
//! [`fit_window`] also fixes the cause at startup: a window that doesn't fit its monitor is
//! maximized, which puts it in the work area (Photoshop opens maximized on small displays).

use egui::{Rect, Vec2, pos2, vec2};

/// Room kept free at the monitor's bottom edge for a taskbar or Dock egui can't see (the Windows
/// taskbar is 40–48 pt tall; a bottom Dock about as much).
pub const TASKBAR: f32 = 48.0;
/// Below this, the guess about where the monitor is went wrong: use the whole window.
const MIN_VISIBLE: f32 = 160.0;

/// Where the window's content can be seen, in egui points. `monitor` is the monitor's size,
/// `inner` the content's rect on the desktop (both from [`egui::ViewportInfo`]). `state` says
/// whether the window is maximized or full screen.
pub fn visible_part(content: Rect, monitor: Option<Vec2>, inner: Option<Rect>, state: WindowState) -> Rect {
    let Some(m) = monitor else { return content };
    if state == WindowState::FullScreen || !m.x.is_finite() || !m.y.is_finite() || m.x < 1.0 || m.y < 1.0 {
        return content;
    }
    // A maximized window is usually inside the work area already, but not always (#3737): on X11
    // a window manager that doesn't reserve a panel's space (Cinnamon's Muffin with PhotoCraft's
    // own title bar, a panel without a strut) maximizes it to the whole monitor, and the panel then
    // covers its bottom. Where the window is known, it is checked like any other; only a maximized
    // window with no position (Wayland) is taken to fit.
    if state == WindowState::Maximized && inner.is_none() {
        return content;
    }
    // #3737: Wayland never tells a window where it is (winit's `inner_position` fails), so egui has
    // no `inner_rect` there, only the monitor's size. A window bigger than the monitor less the
    // taskbar still runs under it wherever it is: at best its content starts at the monitor's top,
    // so at most that much of it can be seen.
    let Some(inner) = inner else {
        let room = vec2(m.x, m.y - TASKBAR);
        return trusted(content, Rect::from_min_size(content.min, room));
    };
    if !inner.min.is_finite() || !inner.max.is_finite() {
        return content;
    }
    // Only the monitor's size is known, not where it is on the desktop: assume monitors tile the
    // desktop by their size (true for one monitor, and for side-by-side ones of the same size).
    let c = inner.center();
    let origin = pos2((c.x / m.x).floor() * m.x, (c.y / m.y).floor() * m.y);
    let screen = Rect::from_min_size(origin, m);
    let usable = Rect::from_min_max(screen.min, pos2(screen.max.x, screen.max.y - TASKBAR));
    // Desktop → window coordinates.
    let shift = content.min - inner.min;
    let local = usable.translate(shift);
    trusted(content, local)
}

/// How the window sits on its monitor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowState {
    Normal,
    Maximized,
    /// Full screen covers the taskbar on purpose: all of it can be seen.
    FullScreen,
}

impl WindowState {
    fn of(v: &egui::ViewportInfo) -> Self {
        if v.fullscreen == Some(true) {
            Self::FullScreen
        } else if v.maximized == Some(true) {
            Self::Maximized
        } else {
            Self::Normal
        }
    }
}

/// `content` cut to `usable`, unless that leaves too little to be a sensible guess.
fn trusted(content: Rect, usable: Rect) -> Rect {
    let visible = content.intersect(usable);
    if !visible.is_positive() || visible.height() < MIN_VISIBLE.min(content.height()) || visible.width() < MIN_VISIBLE.min(content.width()) {
        return content;
    }
    visible
}

/// [`visible_part`] for this frame's viewport.
pub fn visible_rect(ctx: &egui::Context) -> Rect {
    let content = ctx.content_rect();
    ctx.input(|i| {
        let v = i.viewport();
        visible_part(content, v.monitor_size, v.inner_rect, WindowState::of(v))
    })
}

/// Does a window of `outer` size fit on a `monitor` (less the taskbar)?
pub fn fits(outer: Vec2, monitor: Vec2) -> bool {
    outer.x <= monitor.x + 0.5 && outer.y <= monitor.y - TASKBAR + 0.5
}

fn fitted_id() -> egui::Id {
    egui::Id::new("pc-window-fitted")
}

/// Once, as soon as the monitor is known: maximize a window that is bigger than its monitor
/// (it ran under the taskbar and off-screen). Returns true when it asked to.
pub fn fit_window(ctx: &egui::Context) -> bool {
    if ctx.data(|d| d.get_temp::<bool>(fitted_id())).is_some() {
        return false;
    }
    let content = ctx.content_rect();
    let (monitor, outer, maximized, fullscreen) = ctx.input(|i| {
        let v = i.viewport();
        (v.monitor_size, v.outer_rect, v.maximized, v.fullscreen)
    });
    // On Wayland the window's position, and so its outer rect, is never known (#3737): its content
    // size is the least the window takes up.
    let outer = outer.unwrap_or(content);
    let Some(m) = monitor else { return false };
    // macOS keeps new windows inside the screen's visible frame itself.
    if cfg!(target_os = "macos") {
        return false;
    }
    ctx.data_mut(|d| d.insert_temp(fitted_id(), true));
    if maximized == Some(true) || fullscreen == Some(true) || !m.is_finite() || !outer.is_finite() || fits(outer.size(), m) {
        return false;
    }
    ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use WindowState::{FullScreen, Maximized, Normal};
    use egui::{Rect, pos2, vec2};

    const MONITOR: Vec2 = vec2(1366.0, 768.0);

    fn window(x: f32, y: f32, w: f32, h: f32) -> (Rect, Rect) {
        (Rect::from_min_size(pos2(0.0, 0.0), vec2(w, h)), Rect::from_min_size(pos2(x, y), vec2(w, h)))
    }

    #[test]
    fn a_window_taller_than_its_display_is_cut_at_the_taskbar() {
        // 1440 × 900 at the desktop origin on 1366 × 768: right and bottom are off-screen.
        let (content, inner) = window(0.0, 0.0, 1440.0, 900.0);
        assert_eq!(visible_part(content, Some(MONITOR), Some(inner), Normal), Rect::from_min_max(pos2(0.0, 0.0), pos2(1366.0, 768.0 - TASKBAR)));
        // Below a 31 pt title bar, a full-height window reaches under the taskbar.
        let (content, inner) = window(0.0, 31.0, 1366.0, 737.0);
        assert_eq!(visible_part(content, Some(MONITOR), Some(inner), Normal).bottom(), 768.0 - TASKBAR - 31.0);
        // A small window clear of the taskbar is all visible.
        let (content, inner) = window(100.0, 80.0, 800.0, 500.0);
        assert_eq!(visible_part(content, Some(MONITOR), Some(inner), Normal), content);
        // On a second monitor to the right, the same.
        let (content, inner) = window(1366.0 + 100.0, 0.0, 1440.0, 900.0);
        assert_eq!(visible_part(content, Some(MONITOR), Some(inner), Normal).bottom(), 768.0 - TASKBAR);
    }

    #[test]
    fn maximized_unknown_or_nonsense_input_keeps_the_window() {
        let (content, inner) = window(0.0, 0.0, 1440.0, 900.0);
        assert_eq!(visible_part(content, Some(MONITOR), Some(inner), FullScreen), content, "full screen covers the taskbar on purpose");
        assert_eq!(visible_part(content, None, Some(inner), Normal), content);
        assert_eq!(visible_part(content, Some(vec2(f32::NAN, 768.0)), Some(inner), Normal), content);
        assert_eq!(visible_part(content, Some(vec2(0.0, 0.0)), Some(inner), Normal), content);
        assert_eq!(visible_part(content, Some(MONITOR), Some(Rect::from_min_size(pos2(f32::INFINITY, 0.0), vec2(10.0, 10.0))), Normal), content);
        // A guess that would leave almost nothing visible is not trusted.
        let (content, inner) = window(0.0, 600.0, 1000.0, 300.0);
        assert_eq!(visible_part(content, Some(MONITOR), Some(inner), Normal), content);
    }

    #[test]
    fn a_maximized_window_under_a_panel_is_still_cut_at_the_taskbar() {
        // #3737, X11 (Linux Mint): the window manager maximized the window to the whole 1920 × 1080
        // monitor without reserving the bottom panel's space, so the panel covers it.
        for scale in [1.0_f32, 1.5] {
            let monitor = vec2(1920.0, 1080.0) / scale;
            let content = Rect::from_min_size(pos2(0.0, 0.0), monitor);
            let bottom = monitor.y - TASKBAR;
            assert_eq!(visible_part(content, Some(monitor), Some(content), Maximized).bottom(), bottom, "at {scale}×");
            // A window manager that reserves the panel: the maximized window already ends above it.
            let fitted = Rect::from_min_size(pos2(0.0, 0.0), vec2(monitor.x, bottom));
            assert_eq!(visible_part(fitted, Some(monitor), Some(fitted), Maximized), fitted, "at {scale}×");
            // Full screen is meant to cover the panel.
            assert_eq!(visible_part(content, Some(monitor), Some(content), FullScreen), content, "at {scale}×");
        }
    }

    #[test]
    fn an_unknown_window_position_still_keeps_clear_of_the_taskbar() {
        // #3737, Wayland: only the monitor's size is known. A window taller than the monitor less
        // the taskbar can be seen at most down to there, wherever it is.
        let content = Rect::from_min_size(pos2(0.0, 0.0), vec2(1366.0, 740.0));
        assert_eq!(visible_part(content, Some(MONITOR), None, Normal), Rect::from_min_max(pos2(0.0, 0.0), pos2(1366.0, 768.0 - TASKBAR)));
        let big = Rect::from_min_size(pos2(0.0, 0.0), vec2(1440.0, 900.0));
        assert_eq!(visible_part(big, Some(MONITOR), None, Normal), Rect::from_min_max(pos2(0.0, 0.0), pos2(1366.0, 768.0 - TASKBAR)));
        // One that fits, or a maximized one, is all visible.
        let small = Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 600.0));
        assert_eq!(visible_part(small, Some(MONITOR), None, Normal), small);
        assert_eq!(visible_part(content, Some(MONITOR), None, Maximized), content);
        assert_eq!(visible_part(content, Some(MONITOR), None, FullScreen), content);
    }

    fn frame(ctx: &egui::Context, monitor: Option<Vec2>, outer: Option<Rect>, maximized: Option<bool>) -> Vec<egui::ViewportCommand> {
        let mut input = egui::RawInput::default();
        if let Some(v) = input.viewports.get_mut(&egui::ViewportId::ROOT) {
            v.monitor_size = monitor;
            v.outer_rect = outer;
            v.maximized = maximized;
        }
        let mut out = ctx.run_ui(input, |ui| {
            fit_window(ui.ctx());
        });
        out.textures_delta.clear();
        out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.commands.clone()).unwrap_or_default()
    }

    #[test]
    fn a_window_too_big_for_its_display_is_maximized_once() {
        let big = Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(1456.0, 939.0)));
        let ctx = egui::Context::default();
        // The monitor isn't known on the first frames: wait.
        assert!(!frame(&ctx, None, big, Some(false)).contains(&egui::ViewportCommand::Maximized(true)));
        let maximize = frame(&ctx, Some(MONITOR), big, Some(false)).contains(&egui::ViewportCommand::Maximized(true));
        assert_eq!(maximize, !cfg!(target_os = "macos"));
        // Once only: un-maximizing it later is the user's choice.
        assert!(!frame(&ctx, Some(MONITOR), big, Some(false)).contains(&egui::ViewportCommand::Maximized(true)));
        // A window that fits, or one already maximized, is left alone.
        let ctx = egui::Context::default();
        assert!(
            !frame(&ctx, Some(MONITOR), Some(Rect::from_min_size(pos2(40.0, 40.0), vec2(1200.0, 700.0))), Some(false))
                .contains(&egui::ViewportCommand::Maximized(true))
        );
        let ctx = egui::Context::default();
        assert!(!frame(&ctx, Some(MONITOR), big, Some(true)).contains(&egui::ViewportCommand::Maximized(true)));
        // Wayland: no outer rect, but a content area bigger than the monitor is maximized too (#3737).
        let ctx = egui::Context::default();
        let mut input = egui::RawInput { screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(1440.0, 900.0))), ..Default::default() };
        if let Some(v) = input.viewports.get_mut(&egui::ViewportId::ROOT) {
            v.monitor_size = Some(MONITOR);
            v.maximized = Some(false);
        }
        let mut out = ctx.run_ui(input, |ui| {
            fit_window(ui.ctx());
        });
        out.textures_delta.clear();
        let cmds = out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.commands.clone()).unwrap_or_default();
        assert_eq!(cmds.contains(&egui::ViewportCommand::Maximized(true)), !cfg!(target_os = "macos"));
        assert!(fits(vec2(1366.0, 720.0), MONITOR) && !fits(vec2(1366.0, 740.0), MONITOR) && !fits(vec2(1440.0, 700.0), MONITOR));
    }
}
