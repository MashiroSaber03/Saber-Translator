//! Keeps the custom (undecorated) title bar window truly borderless (#2246).
//!
//! winit 0.30 gives an undecorated window the `WS_CAPTION | WS_BORDER` styles and hides the
//! non-client area by spreading the client rect over it (`WM_NCCALCSIZE`). On some systems
//! (reported: Windows 10 LTSB, Intel HD Graphics 4600, 1366 × 768) the desktop compositor still
//! reserves that frame for the window surface: the content is drawn offset by the caption and
//! border, leaving black bars at the window's top-left, while the pointer keeps the un-offset
//! client coordinates, so it no longer points at what is drawn under it.
//!
//! winit 0.27 — before rust-windowing/winit#2419 — simply omitted those styles for undecorated
//! windows, and there was no offset. [`service`] restores that: it strips `WS_CAPTION` and
//! `WS_BORDER` and forces Windows to recalculate the frame. winit puts them back whenever it
//! changes the window flags (maximize, restore, …), so the returned guard checks every frame.
//!
//! Windows sizes a maximize by the frame its styles describe, so a window maximized while its
//! caption is gone is sized without one: its rect covers the whole monitor, taskbar included.
//! The shell reads a caption-less window covering its monitor as a full-screen app and drops the
//! taskbar's backdrop behind it — a black band on stock Windows 11, a glitched unresponsive one
//! under ExplorerPatcher (#3144). The guard therefore also trims a parked maximize's rect back to
//! the work area plus its own frame, clear of the taskbar. winit pins the client area to the work
//! area either way (`WM_NCCALCSIZE`), so the app's pixels do not move.
//!
//! The Windows calls go through safe `winsafe` wrappers; the workspace forbids `unsafe`, and
//! winsafe can't wrap a raw handle without it, so the window is found by its handle instead.

pub use platform::service;

#[cfg(target_os = "windows")]
mod platform {
    use eframe::wgpu::rwh::{HasWindowHandle, RawWindowHandle};
    use photocraft_ui_egui::WindowFrameFn;
    use winsafe::{HWND, HwndPlace, POINT, SIZE, co};

    /// A guard that keeps the window borderless each frame, or `None` when the app uses the
    /// system title bar (the window is decorated) or its handle can't be found.
    pub fn service(cc: &eframe::CreationContext<'_>, custom_titlebar: bool) -> Option<WindowFrameFn> {
        if !custom_titlebar {
            return None;
        }
        let target = match cc.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(h) => h.hwnd.get(),
            _ => return None,
        };
        let hwnd = find_window(target)?;
        reapply(&hwnd);
        Some(Box::new(move || reapply(&hwnd)))
    }

    /// The top-level window with the given raw handle. `EnumWindows` hands out real `HWND`s, so
    /// this matches winit's own window without wrapping a raw handle (which needs `unsafe`).
    fn find_window(target: isize) -> Option<HWND> {
        let mut found = None;
        let _ = winsafe::EnumWindows(|hwnd| {
            if hwnd.ptr() as isize == target {
                found = Some(hwnd);
                false
            } else {
                true
            }
        });
        found
    }

    /// One pass of the per-frame guard: the styles winit put back come off again, and a parked
    /// maximize stays clear of the taskbar.
    fn reapply(hwnd: &HWND) {
        strip(hwnd);
        keep_clear_of_the_taskbar(hwnd);
    }

    /// Removes `WS_CAPTION`/`WS_BORDER` and has Windows recalculate the frame. A no-op once they
    /// are gone, so it is cheap to call every frame.
    fn strip(hwnd: &HWND) {
        let style = hwnd.style();
        let without = borderless(style);
        if without.raw() == style.raw() {
            return;
        }
        hwnd.set_style(without);
        let _ = hwnd.SetWindowPos(
            HwndPlace::None,
            POINT::new(),
            SIZE::new(),
            co::SWP::FRAMECHANGED | co::SWP::NOMOVE | co::SWP::NOSIZE | co::SWP::NOZORDER | co::SWP::NOACTIVATE,
        );
    }

    /// How far a parked maximize reaches past its monitor on the left, top and right: the resize
    /// border (about 8 pt at 100 %, more with DPI). Past that, a maximized window's rect is not
    /// parked but being dragged or snapped, and is left alone.
    const FRAME: i32 = 32;

    /// [`clamped_height`] for this frame's window: trim a maximized window whose rect runs under
    /// the taskbar back to the work area plus its frame (#3144). No-op otherwise.
    fn keep_clear_of_the_taskbar(hwnd: &HWND) {
        if !hwnd.IsZoomed() {
            return;
        }
        let (Ok(r), Ok(mi)) = (hwnd.GetWindowRect(), hwnd.MonitorFromWindow(co::MONITOR::DEFAULTTONEAREST).GetMonitorInfo()) else {
            return;
        };
        let (m, w) = (mi.rcMonitor, mi.rcWork);
        let Some(height) = clamped_height((r.left, r.top, r.right, r.bottom), (m.left, m.top, m.right, m.bottom), (w.left, w.top, w.right, w.bottom)) else {
            return;
        };
        let _ = hwnd.SetWindowPos(
            HwndPlace::None,
            POINT::new(),
            SIZE { cx: r.right - r.left, cy: height },
            co::SWP::NOMOVE | co::SWP::NOZORDER | co::SWP::NOACTIVATE,
        );
    }

    /// The height to trim a maximize to so its rect stops at the taskbar, or `None` when it
    /// already does or `rect` (`left, top, right, bottom`, desktop coordinates) is not parked at
    /// its maximize place. `monitor` is the monitor's full rect and `work` its work area (the
    /// monitor less the taskbar) in the same coordinates. The parked check is against the full
    /// monitor, because that is where Windows puts a maximize; the trim is against the work area.
    fn clamped_height(rect: (i32, i32, i32, i32), monitor: (i32, i32, i32, i32), work: (i32, i32, i32, i32)) -> Option<i32> {
        let (l, t, r, b) = rect;
        let (ml, mt, mr, _) = monitor;
        let (_, _, _, wb) = work;
        let side = ml - l;
        if !(0..=FRAME).contains(&side) || !(0..=FRAME).contains(&(mt - t)) || !(0..=FRAME).contains(&(r - mr)) {
            return None;
        }
        if b <= wb + side {
            return None;
        }
        Some(wb + side - t)
    }

    /// `style` without the OS title bar and frame (`WS_CAPTION`, which already includes
    /// `WS_BORDER`). The sizing border (`WS_THICKFRAME`) stays, so edge resizing and Aero snap
    /// keep working, as they did with winit 0.27's undecorated windows.
    fn borderless(style: co::WS) -> co::WS {
        style & !(co::WS::CAPTION | co::WS::BORDER)
    }

    #[cfg(test)]
    mod tests {
        use super::{borderless, clamped_height};
        use winsafe::co;

        #[test]
        fn the_borderless_style_drops_the_caption_and_border_only() {
            // `WS_CAPTION` is 0x00c0_0000 and already contains `WS_BORDER` (0x0080_0000).
            let decorated = co::WS::CAPTION | co::WS::BORDER | co::WS::SYSMENU | co::WS::THICKFRAME | co::WS::MAXIMIZEBOX | co::WS::MINIMIZEBOX;
            let bare = borderless(decorated);
            assert!(!bare.has(co::WS::CAPTION), "caption gone");
            assert!(!bare.has(co::WS::BORDER), "border gone");
            assert!(bare.has(co::WS::THICKFRAME), "edge resizing kept");
            assert!(bare.has(co::WS::MAXIMIZEBOX) && bare.has(co::WS::SYSMENU), "window buttons kept");
            // Already borderless: unchanged, so the per-frame call does nothing.
            assert_eq!(bare.raw(), borderless(bare).raw());
        }

        #[test]
        fn a_maximize_covering_the_taskbar_is_trimmed_to_the_work_area() {
            // 1920 × 1080 with a 34 pt taskbar: a caption-less maximize covers the whole
            // monitor, taskbar included (#3144). Trimmed so the rect ends at the work area
            // plus the frame the window was parked with.
            let monitor = (0, 0, 1920, 1080);
            let work = (0, 0, 1920, 1046);
            assert_eq!(clamped_height((-7, -7, 1927, 1087), monitor, work), Some(1046 + 7 - (-7)));
        }

        #[test]
        fn a_maximize_already_clear_of_the_taskbar_is_left_alone() {
            // What Windows sizes a framed maximize to: the work area plus the frame.
            assert_eq!(clamped_height((-8, -8, 1928, 1054), (0, 0, 1920, 1080), (0, 0, 1920, 1046)), None);
        }

        #[test]
        fn a_maximize_beside_a_side_taskbar_is_left_alone() {
            // Taskbar on the left: the work area starts further right, but the monitor's bottom
            // is still clear, so the rect reaches nothing.
            assert_eq!(clamped_height((-8, -8, 1928, 1088), (0, 0, 1920, 1080), (48, 0, 1920, 1080)), None);
        }

        #[test]
        fn a_maximize_being_dragged_away_is_left_alone() {
            let monitor = (0, 0, 1920, 1080);
            let work = (0, 0, 1920, 1046);
            assert_eq!(clamped_height((100, 200, 1100, 900), monitor, work), None, "moved");
            assert_eq!(clamped_height((-500, -7, 1427, 1087), monitor, work), None, "resized");
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use photocraft_ui_egui::WindowFrameFn;

    pub fn service(_cc: &eframe::CreationContext<'_>, _custom_titlebar: bool) -> Option<WindowFrameFn> {
        None
    }
}
