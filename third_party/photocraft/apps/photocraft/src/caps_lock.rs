//! Caps Lock state, read from the OS once per frame so the canvas can show the precise crosshair
//! for painting tools whatever the cursor preference (Photoshop's ⇪ behaviour, #1758).
//!
//! winit and egui have no Caps Lock key or modifier — `egui::Key` has no variant, `Modifiers` no
//! field, and egui-winit's `key_from_named_key` drops the `NamedKey::CapsLock` key event before it
//! reaches the app. So the state is polled here instead, from the platform:
//!
//! - **X11 (and XWayland)**: XKB's Lock modifier state, via the pure-Rust x11rb client. The app
//!   already opens on X11 when the `Preferences › Performance › Linux display server` preference
//!   asks for it, and on Wayland through XWayland for pen support, so this is the common path.
//! - **Windows**: `GetKeyState(VK_CAPITAL)` through the safe `winsafe` binding.
//! - **macOS**: `CGEventSourceFlagsState` (`kCGEventFlagAlphaShift`) through `core-graphics2`.
//! - **Native Wayland**: there is no global keyboard-state query (no equivalent of XKB GetState or
//!   VK_CAPITAL); the service is `None` there and Caps Lock leaves the cursor preference alone.
//!
//! No `unsafe`: every OS call goes through a safe binding.

pub use platform::service;

#[cfg(target_os = "macos")]
mod platform {
    use core_graphics2::event_source::CGEventSource;
    use core_graphics2::event_types::{CGEventFlags, CGEventSourceStateID};
    use photocraft_ui_egui::CapsLockFn;

    pub fn service(_cc: &eframe::CreationContext<'_>) -> Option<CapsLockFn> {
        Some(Box::new(|| {
            let flags = CGEventSource::flags_state(CGEventSourceStateID::CombinedSessionState);
            flags.contains(CGEventFlags::MaskAlphaShift)
        }))
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use photocraft_ui_egui::CapsLockFn;

    pub fn service(_cc: &eframe::CreationContext<'_>) -> Option<CapsLockFn> {
        Some(Box::new(|| {
            // The second element of the pair is "key toggled on", which is Caps Lock's state.
            let (_, toggled) = winsafe::GetKeyState(winsafe::co::VK::CAPITAL);
            toggled
        }))
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use eframe::wgpu::rwh::{HasWindowHandle, RawWindowHandle};
    use photocraft_ui_egui::CapsLockFn;
    use x11rb::protocol::xkb::{self, ConnectionExt as _};
    use x11rb::protocol::xproto::ModMask;
    use x11rb::rust_connection::RustConnection;

    pub fn service(cc: &eframe::CreationContext<'_>) -> Option<CapsLockFn> {
        // XKB keyboard state lives on the X server. The window is our window: whether it is X11
        // (or XWayland) is whether there is an X server to ask. Native Wayland has no such
        // connection and degrades to the cursor preference (no service).
        let is_x11 = matches!(cc.window_handle().ok()?.as_raw(), RawWindowHandle::Xlib(_) | RawWindowHandle::Xcb(_));
        if !is_x11 {
            return None;
        }
        // Lazily opened on first use and kept: the Lock-modifier mask is stable per keymap, so it
        // is queried once and only the toggle state is re-read every frame.
        let mut conn: Option<(RustConnection, ModMask)> = None;
        // After a failure, wait before reconnecting (and warn once per failure), so a broken X
        // connection doesn't reconnect and log on every frame.
        let mut retry_at: Option<std::time::Instant> = None;
        Some(Box::new(move || {
            if retry_at.is_some_and(|t| std::time::Instant::now() < t) {
                return false;
            }
            match caps_lock_on(&mut conn) {
                Ok(on) => {
                    retry_at = None;
                    on
                }
                Err(error) => {
                    log::warn!("couldn't read the Caps Lock state: {error}");
                    conn = None;
                    retry_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(5));
                    false
                }
            }
        }))
    }

    /// Caps Lock is on when the Lock modifier is both engaged (`locked_mods`) and present in the
    /// keyboard's indicator map (so a re-mapped lock key is still recognised).
    fn caps_lock_on(conn: &mut Option<(RustConnection, ModMask)>) -> Result<bool, String> {
        if conn.is_none() {
            let (c, _) = RustConnection::connect(None).map_err(|e| e.to_string())?;
            // XKB requests need the extension active first (it is, on every modern server).
            let enabled = c.xkb_use_extension(1, 0).map_err(|e| e.to_string())?.reply().map_err(|e| e.to_string())?;
            if !enabled.supported {
                return Err("the X server has no XKB extension".into());
            }
            let device = u16::from(xkb::ID::USE_CORE_KBD);
            // Indicator 0 is the Caps Lock indicator; its `.mods` is the Lock-modifier mask.
            let map = c.xkb_get_indicator_map(device, 1).map_err(|e| e.to_string())?.reply().map_err(|e| e.to_string())?;
            let lock = map.maps.first().map_or(ModMask::LOCK, |m| m.mods);
            *conn = Some((c, lock));
        }
        let (conn, lock) = match conn.as_ref() {
            Some((conn, lock)) => (conn, *lock),
            None => return Err("no X connection".into()),
        };
        let device = u16::from(xkb::ID::USE_CORE_KBD);
        let state = conn.xkb_get_state(device).map_err(|e| e.to_string())?.reply().map_err(|e| e.to_string())?;
        Ok(u16::from(state.locked_mods) & u16::from(lock) != 0)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod platform {
    use photocraft_ui_egui::CapsLockFn;

    pub fn service(_cc: &eframe::CreationContext<'_>) -> Option<CapsLockFn> {
        None
    }
}
