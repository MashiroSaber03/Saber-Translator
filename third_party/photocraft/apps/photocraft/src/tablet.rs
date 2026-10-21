//! Pen tablet pressure, tilt, rotation and eraser end on macOS and Linux X11 (issue #79).
//!
//! winit 0.30 drops tablet data on both, so `photocraft-tablet` reads it beside winit (an AppKit
//! local event monitor; XInput2 raw events on a second X connection) and this module writes each
//! sample into the UI's [`StylusFeed`], where the canvas reads it exactly like the web runner's
//! Pointer Events and automation's simulated pen. Windows needs nothing here: winit forwards
//! `WM_POINTER` pressure as touch force.
//!
//! Wayland compositors give a pen only to clients that bind the tablet protocol
//! (`zwp_tablet_v2`), which winit 0.30 doesn't (and binding it on winit's connection needs
//! `unsafe`, see the crate docs): there the pen does nothing in the window, not even move the
//! pointer (#639). Xwayland binds it and serves the pen as an XInput2 device, so with a pen
//! attached the app opens its window through Xwayland ([`display_session`]), where the X11 reader
//! gets pressure, tilt and the eraser end.

#[cfg(any(target_os = "linux", test))]
use crate::linux_libs::DisplaySession;
use photocraft_tablet::Sample;
use photocraft_ui_egui::stylus::{PenSample, StylusFeed};

/// A tablet sample as the UI's pen sample.
pub fn pen_sample(s: Sample) -> PenSample {
    PenSample { pressure: s.pressure, tilt_x: s.tilt_x, tilt_y: s.tilt_y, rotation: s.rotation, eraser: s.eraser }.sanitized()
}

/// Writes tablet samples into `feed` (`None` = a mouse).
pub fn sink(feed: StylusFeed) -> impl Fn(Option<Sample>) + Send + 'static {
    move |s| feed.set(s.map(pen_sample))
}

/// Install the AppKit tablet monitor (main thread, once winit created the shared application:
/// from eframe's app creation, not before the event loop). Keep the result until the event loop
/// returns. On failure this logs why: pens then paint like a mouse.
#[cfg(target_os = "macos")]
pub fn install_macos(feed: &StylusFeed) -> Option<photocraft_tablet::macos::Monitor> {
    photocraft_tablet::macos::Monitor::install(sink(feed.clone())).map_err(|e| log::warn!("{e}")).ok()
}

/// The display server to open the window on: the session's own, except Xwayland on Wayland when
/// a pen is attached (see the module docs) and Xwayland can run the window: `$DISPLAY` is set
/// and the X11 libraries are installed. `PHOTOCRAFT_NATIVE_WAYLAND=1` keeps native Wayland.
#[cfg(target_os = "linux")]
pub fn display_session() -> DisplaySession {
    let var = |k: &str| std::env::var(k).ok();
    let session = crate::linux_libs::session_from_env(var);
    let pen = || std::fs::read_to_string("/proc/bus/input/devices").is_ok_and(|d| has_pen(&d));
    let x11_libs = || crate::linux_libs::available(DisplaySession::X11);
    if pen_needs_xwayland(session, var, pen, x11_libs) {
        eprintln!("photocraft: pen found; opening the window through Xwayland, where the pen works (PHOTOCRAFT_NATIVE_WAYLAND=1 keeps Wayland)");
        return DisplaySession::X11;
    }
    session
}

/// [`display_session`]'s decision; the system probes run only when the session asks for them.
#[cfg(any(target_os = "linux", test))]
pub fn pen_needs_xwayland(session: DisplaySession, var: impl Fn(&str) -> Option<String>, pen: impl FnOnce() -> bool, x11_libs: impl FnOnce() -> bool) -> bool {
    let set = |k: &str| var(k).is_some_and(|v| !v.is_empty() && v != "0");
    session == DisplaySession::Wayland && !set("PHOTOCRAFT_NATIVE_WAYLAND") && set("DISPLAY") && pen() && x11_libs()
}

/// `BTN_TOOL_PEN` (linux/input-event-codes.h): the key a pen tablet's stylus, a pen display's or
/// a pen-enabled touchscreen's reports (libinput treats such a device as a tablet tool).
#[cfg(any(target_os = "linux", test))]
const BTN_TOOL_PEN: u32 = 0x140;

/// Is a pen among the input devices `/proc/bus/input/devices` lists? Each device's `B: KEY=` line
/// is its key bitmap as hex words of the kernel's `long`, most significant first.
#[cfg(any(target_os = "linux", test))]
pub fn has_pen(devices: &str) -> bool {
    let bits = usize::BITS;
    let word = usize::try_from(BTN_TOOL_PEN / bits).unwrap_or(usize::MAX);
    devices.lines().filter_map(|l| l.strip_prefix("B: KEY=")).any(|keys| {
        keys.split_whitespace().rev().nth(word).and_then(|w| u64::from_str_radix(w, 16).ok()).is_some_and(|w| (w >> (BTN_TOOL_PEN % bits)) & 1 == 1)
    })
}

/// Start the XInput2 reader when eframe runs on X11 (Xwayland included). On native Wayland,
/// `$DISPLAY` is Xwayland, which sees none of this window's input (and a pen sample from another
/// X app would outlive a mouse stroke here), so nothing starts.
#[cfg(target_os = "linux")]
pub fn spawn_x11(feed: &StylusFeed, display: Option<DisplayKind>) {
    if display != Some(DisplayKind::X11) {
        log::info!("tablet pressure: not available on this display server ({display:?})");
        return;
    }
    if let Err(e) = photocraft_tablet::x11::spawn(None, sink(feed.clone())) {
        log::warn!("{e}");
    }
}

/// The windowing system eframe ended up on.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayKind {
    X11,
    Wayland,
    Other,
}

#[cfg(target_os = "linux")]
impl DisplayKind {
    /// From the creation context's display handle.
    pub fn of(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        use eframe::wgpu::rwh::{HasDisplayHandle, RawDisplayHandle};
        Some(match cc.display_handle().ok()?.as_raw() {
            RawDisplayHandle::Xlib(_) | RawDisplayHandle::Xcb(_) => DisplayKind::X11,
            RawDisplayHandle::Wayland(_) => DisplayKind::Wayland,
            _ => DisplayKind::Other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_tablet::{Update, appkit, xi};
    use photocraft_ui_egui::PhotocraftApp;
    use photocraft_ui_egui::canvas::{ToolEvent, tool_event};
    use photocraft_ui_egui::state::Tool;
    use serde_json::{Value, json};

    #[test]
    fn samples_map_and_sanitize() {
        let p = pen_sample(Sample { pressure: 0.25, tilt_x: 30.0, tilt_y: -10.0, rotation: 90.0, eraser: true });
        assert_eq!(p, PenSample { pressure: 0.25, tilt_x: 30.0, tilt_y: -10.0, rotation: 90.0, eraser: true });
        let p = pen_sample(Sample { pressure: f32::NAN, tilt_x: 500.0, tilt_y: f32::NEG_INFINITY, rotation: -45.0, eraser: false });
        assert_eq!(p, PenSample { pressure: 1.0, tilt_x: 90.0, tilt_y: 0.0, rotation: 315.0, eraser: false });
        let feed = StylusFeed::default();
        let f = sink(feed.clone());
        f(Some(Sample { pressure: 0.5, ..Default::default() }));
        assert_eq!(feed.get().map(|p| p.pressure), Some(0.5));
        f(None);
        assert_eq!(feed.get(), None);
    }

    /// The platform path end to end, minus the OS: raw AppKit / XInput2 values → the crate's
    /// mapping → this module's sink → the stylus feed → the canvas's tool events → `paint.stroke`.
    /// A light-to-heavy pen stroke must paint thin and faint at the light end and wide and opaque
    /// at the heavy end (Size and Opacity on Pen Pressure); a mouse paints full strength.
    fn pressure_stroke(mut next: impl FnMut(f32) -> Option<Sample>) -> (f64, f64, f64, f64) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let feed = app.stylus.feed.clone();
        let sink = sink(feed);
        app.session.execute("file.new", json!({"width": 240, "height": 80, "background": "white"})).unwrap();
        app.session.execute("tools.setColors", json!({"foreground": [0.0, 0.0, 0.0, 1.0]})).unwrap();
        let brush = json!({"size": 30, "hardness": 1.0, "spacing": 0.05, "pressureSize": true, "pressureOpacity": true, "smoothing": {"amount": 0}});
        app.session.execute("tools.setBrush", json!({ "brush": brush })).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;
        let n = 20;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            sink(next(0.1 + 0.9 * t));
            let (x, y) = (20.0 + 200.0 * f64::from(t), 40.0);
            let pressure = app.stylus.pressure();
            let ev = if i == 0 { ToolEvent::Down { x, y, pressure } } else { ToolEvent::Move { x, y, pressure } };
            tool_event(&mut app, ev, m);
        }
        tool_event(&mut app, ToolEvent::Up { x: 220.0, y: 40.0 }, m);
        sink(None);
        // Darkness (1 - luma) along x = 50 (light end) and x = 200 (heavy end), and the painted
        // half-width there.
        let mut dark = |x: i32, y: i32| -> f64 {
            let px: Value = app.session.execute("document.pixel", json!({"x": x, "y": y})).unwrap();
            1.0 - px[0].as_f64().unwrap()
        };
        let (light, heavy) = (dark(50, 40), dark(200, 40));
        let light_w = (0..40).take_while(|d| dark(50, 40 + d) > 0.02).count() as f64;
        let heavy_w = (0..40).take_while(|d| dark(200, 40 + d) > 0.02).count() as f64;
        (light, light_w, heavy, heavy_w)
    }

    #[test]
    fn appkit_tablet_pressure_drives_stroke_width_and_opacity() {
        let mut st = appkit::State::default();
        let (light_dark, light_w, heavy_dark, heavy_w) = pressure_stroke(|p| {
            let e =
                appkit::RawEvent { kind: appkit::event_type::LEFT_MOUSE_DRAGGED, subtype: appkit::subtype::TABLET_POINT, pressure: p, ..Default::default() };
            match st.handle(&e) {
                Update::Set(s) => s,
                Update::Keep => None,
            }
        });
        assert!(heavy_w > light_w * 2.0, "width responds: {light_w} → {heavy_w}");
        assert!(heavy_dark > light_dark + 0.3, "opacity responds: {light_dark} → {heavy_dark}");
        assert!(heavy_dark > 0.9, "full pressure paints (nearly) opaque: {heavy_dark}");

        // A mouse (non-tablet subtype) through the same path: full size and opacity throughout.
        let mut st = appkit::State::default();
        let (m_dark, m_w, _, _) = pressure_stroke(|p| {
            let e = appkit::RawEvent { kind: appkit::event_type::LEFT_MOUSE_DRAGGED, subtype: appkit::subtype::MOUSE, pressure: p, ..Default::default() };
            match st.handle(&e) {
                Update::Set(s) => s,
                Update::Keep => None,
            }
        });
        assert!(m_dark > 0.9 && m_w >= heavy_w - 1.0, "mouse = pressure 1: {m_dark}, {m_w}");
    }

    #[test]
    fn xinput_tablet_pressure_drives_stroke_width_and_opacity() {
        let mut st = xi::State::default();
        st.set_devices([xi::Device::new(9, "Wacom Intuos Pen stylus", [(2, "Abs Pressure".to_string(), 0.0, 65535.0)])]);
        let (light_dark, light_w, heavy_dark, heavy_w) = pressure_stroke(|p| match st.handle(9, [(2, f64::from(p) * 65535.0)]) {
            Update::Set(s) => s,
            Update::Keep => None,
        });
        assert!(heavy_w > light_w * 2.0, "width responds: {light_w} → {heavy_w}");
        assert!(heavy_dark > light_dark + 0.3, "opacity responds: {light_dark} → {heavy_dark}");
    }

    /// `/proc/bus/input/devices` of a laptop with a Wacom Intuos Pro attached (64-bit longs: the
    /// pen's `KEY=1c03 0 0 0 0 0` has bit 0x140, BTN_TOOL_PEN) and of the same laptop without it.
    const WITH_PEN: &str = "I: Bus=0011 Vendor=0001 Product=0001 Version=ab83\n\
        N: Name=\"AT Translated Set 2 keyboard\"\n\
        H: Handlers=sysrq kbd event3 leds\n\
        B: KEY=402000000 3803078f800d001 feffffdfffefffff fffffffffffffffe\n\
        \n\
        I: Bus=0018 Vendor=06cb Product=ce26 Version=0100\n\
        N: Name=\"SYNA2B33:00 06CB:CE26 Touchpad\"\n\
        H: Handlers=mouse1 event7\n\
        B: KEY=e520 10000 0 0 0 0\n\
        B: ABS=2e0800000000003\n\
        \n\
        I: Bus=0003 Vendor=056a Product=0357 Version=0110\n\
        N: Name=\"Wacom Intuos Pro M Pen\"\n\
        H: Handlers=mouse2 event10\n\
        B: KEY=1c03 0 0 0 0 0\n\
        B: ABS=1000f000003\n";

    #[test]
    fn a_pen_is_found_by_its_btn_tool_pen_key() {
        assert!(has_pen(WITH_PEN));
        let without = WITH_PEN.split("\n\n").take(2).collect::<Vec<_>>().join("\n\n");
        assert!(!has_pen(&without), "a touchpad's BTN_TOOL_FINGER (0x145) is not a pen");
        // Short or malformed bitmaps are no pen, never a panic.
        for keys in ["", "B: KEY=", "B: KEY=1", "B: KEY=zz 0 0 0 0 0", "B: KEY=ffffffffffffffffff 0 0 0 0 0", "B: KEY=0 0 0 0 0 0 0 0"] {
            assert!(!has_pen(keys), "{keys:?}");
        }
    }

    #[test]
    fn a_pen_on_wayland_opens_the_window_through_xwayland() {
        use std::collections::HashMap;
        let env = |pairs: &[(&str, &str)]| {
            let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            move |k: &str| m.get(k).cloned()
        };
        let wayland = [("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")];
        assert!(pen_needs_xwayland(DisplaySession::Wayland, env(&wayland), || true, || true));
        // No pen, no Xwayland, no X11 libraries, or the user keeps Wayland: native Wayland.
        assert!(!pen_needs_xwayland(DisplaySession::Wayland, env(&wayland), || false, || true));
        assert!(!pen_needs_xwayland(DisplaySession::Wayland, env(&[("WAYLAND_DISPLAY", "wayland-0")]), || true, || true));
        assert!(!pen_needs_xwayland(DisplaySession::Wayland, env(&wayland), || true, || false));
        let native = [("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0"), ("PHOTOCRAFT_NATIVE_WAYLAND", "1")];
        assert!(!pen_needs_xwayland(DisplaySession::Wayland, env(&native), || true, || true));
        let zero = [("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0"), ("PHOTOCRAFT_NATIVE_WAYLAND", "0")];
        assert!(pen_needs_xwayland(DisplaySession::Wayland, env(&zero), || true, || true));
        // X11 already gets the pen; the probes don't even run off Wayland.
        assert!(!pen_needs_xwayland(DisplaySession::X11, env(&[("DISPLAY", ":0")]), || panic!("probed"), || panic!("probed")));
    }

    #[test]
    fn use_tablet_pressure_off_paints_like_a_mouse() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("prefs.set", json!({"path": "tools.useTabletPressure", "value": false})).unwrap();
        // The canvas copies the preference each frame; do what it does.
        app.stylus.use_pressure = app.session.prefs().tools.use_tablet_pressure;
        sink(app.stylus.feed.clone())(Some(Sample { pressure: 0.2, tilt_x: 40.0, ..Default::default() }));
        assert_eq!(app.stylus.pressure(), 1.0);
        assert_eq!(app.stylus.sample(), None);
    }

    #[test]
    fn eraser_end_switches_to_the_eraser_and_back() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let sink = sink(app.stylus.feed.clone());
        let mut st = appkit::State::default();
        let mut send = |e: appkit::RawEvent| {
            if let Update::Set(s) = st.handle(&e) {
                sink(s);
            }
        };
        let prox = |device, entering| appkit::RawEvent { kind: appkit::event_type::TABLET_PROXIMITY, device, entering, ..Default::default() };
        app.ui.tool = Tool::Brush;
        send(prox(appkit::device::PEN, true));
        assert!(!photocraft_ui_egui::stylus::Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Brush, "the tip changes nothing");
        send(prox(appkit::device::PEN, false));
        send(prox(appkit::device::ERASER, true));
        assert!(photocraft_ui_egui::stylus::Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Eraser);
        // Mouse moves in between change nothing; the tip coming back restores the Brush.
        send(appkit::RawEvent { kind: appkit::event_type::MOUSE_MOVED, ..Default::default() });
        assert!(!photocraft_ui_egui::stylus::Stylus::sync_eraser_tool(&mut app));
        send(prox(appkit::device::PEN, true));
        assert!(photocraft_ui_egui::stylus::Stylus::sync_eraser_tool(&mut app));
        assert_eq!(app.ui.tool, Tool::Brush);
    }
}
