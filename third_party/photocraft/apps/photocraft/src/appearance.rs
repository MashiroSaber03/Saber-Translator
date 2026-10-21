//! Linux desktop colour scheme, including Wayland compositors that do not report a winit theme.

#[cfg(target_os = "linux")]
const PORTAL_NAMESPACE: &str = "org.freedesktop.appearance";
#[cfg(target_os = "linux")]
const PORTAL_KEY: &str = "color-scheme";

#[cfg(target_os = "linux")]
pub(crate) fn portal_proxy() -> Option<zbus::blocking::Proxy<'static>> {
    let conn = zbus::blocking::connection::Builder::session().ok()?.method_timeout(std::time::Duration::from_secs(2)).build().ok()?;
    zbus::blocking::Proxy::new(&conn, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.Settings").ok()
}

#[cfg(target_os = "linux")]
fn portal_theme(proxy: &zbus::blocking::Proxy<'_>) -> Option<egui::Theme> {
    let value: zbus::zvariant::OwnedValue = proxy.call("Read", &(PORTAL_NAMESPACE, PORTAL_KEY)).ok()?;
    portal_code(value).and_then(decode).or_else(gtk_theme)
}

#[cfg(target_os = "linux")]
fn portal_code(value: zbus::zvariant::OwnedValue) -> Option<u32> {
    let mut value: zbus::zvariant::Value<'_> = value.into();
    // The Settings.Read result is a variant; some portals nest another variant inside it.
    for _ in 0..4 {
        match value {
            zbus::zvariant::Value::Value(inner) => value = *inner,
            other => return u32::try_from(other).ok(),
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn decode(code: u32) -> Option<egui::Theme> {
    match code {
        1 => Some(egui::Theme::Dark),
        2 => Some(egui::Theme::Light),
        _ => None,
    }
}

/// GNOME fallback when the portal has no preference. Read only on the appearance worker,
/// using an absolute path rather than the shell's PATH.
#[cfg(target_os = "linux")]
fn gtk_theme() -> Option<egui::Theme> {
    let exe = ["/usr/bin/gsettings", "/bin/gsettings", "/usr/local/bin/gsettings"].into_iter().find(|p| std::path::Path::new(p).is_file())?;
    let output = std::process::Command::new(exe).args(["get", "org.gnome.desktop.interface", "color-scheme"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    parse_gtk_scheme(std::str::from_utf8(&output.stdout).ok()?)
}

#[cfg(target_os = "linux")]
fn parse_gtk_scheme(s: &str) -> Option<egui::Theme> {
    match s.trim().trim_matches('\'') {
        "prefer-dark" => Some(egui::Theme::Dark),
        "prefer-light" | "default" => Some(egui::Theme::Light),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn code_of(theme: Option<egui::Theme>) -> u8 {
    match theme {
        Some(egui::Theme::Dark) => 1,
        Some(egui::Theme::Light) => 2,
        None => 0,
    }
}

#[cfg(target_os = "linux")]
fn portal_startup<T>(subscribe: impl FnOnce(), read: impl FnOnce() -> T) -> T {
    subscribe();
    read()
}

/// Reads the colour scheme once, then follows the portal's `SettingChanged` signal. There is no
/// polling: the worker sleeps in the signal iterator and wakes the UI only when the value changes.
pub fn service() -> Option<photocraft_ui_egui::SystemThemeFn> {
    #[cfg(target_os = "linux")]
    {
        start(|publisher| {
            let proxy = portal_proxy();
            // Subscribe before the first read, so a change during start-up isn't lost.
            let mut signals = None;
            let initial = portal_startup(
                || signals = proxy.as_ref().and_then(|proxy| proxy.receive_signal("SettingChanged").ok()),
                || proxy.as_ref().and_then(portal_theme).or_else(gtk_theme),
            );
            publisher.publish(initial);
            let Some(signals) = signals else { return };
            for message in signals {
                let Ok((namespace, key, changed)) = message.body().deserialize::<(String, String, zbus::zvariant::OwnedValue)>() else {
                    continue;
                };
                if namespace != PORTAL_NAMESPACE || key != PORTAL_KEY {
                    continue;
                }
                publisher.publish(portal_code(changed).and_then(decode).or_else(gtk_theme));
            }
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The worker's handle on the shared value. Every change wakes the registered UI, including the
/// first read when it lands after the start-up wait below.
#[cfg(target_os = "linux")]
struct Publisher {
    value: std::sync::Arc<std::sync::atomic::AtomicU8>,
    wake: std::sync::Arc<std::sync::OnceLock<egui::Context>>,
    ready: std::sync::mpsc::Sender<()>,
}

#[cfg(target_os = "linux")]
impl Publisher {
    fn publish(&self, theme: Option<egui::Theme>) {
        let code = code_of(theme);
        if self.value.swap(code, std::sync::atomic::Ordering::Relaxed) != code
            && let Some(ctx) = self.wake.get()
        {
            ctx.request_repaint();
        }
        let _ = self.ready.send(());
    }
}

/// Runs `worker` on its own thread and waits up to 250 ms for its first value, so a fast desktop
/// answers before the first frame.
#[cfg(target_os = "linux")]
fn start(worker: impl FnOnce(&Publisher) + Send + 'static) -> Option<photocraft_ui_egui::SystemThemeFn> {
    use std::sync::{Arc, OnceLock, atomic::AtomicU8};
    let value = Arc::new(AtomicU8::new(0));
    let wake: Arc<OnceLock<egui::Context>> = Arc::new(OnceLock::new());
    let (ready, wait) = std::sync::mpsc::channel();
    let publisher = Publisher { value: Arc::clone(&value), wake: Arc::clone(&wake), ready };
    std::thread::Builder::new().name("appearance-portal".into()).spawn(move || worker(&publisher)).ok()?;
    let _ = wait.recv_timeout(std::time::Duration::from_millis(250));
    Some(Box::new(move |ctx| {
        let _ = wake.set(ctx.clone());
        decode(u32::from(value.load(std::sync::atomic::Ordering::Relaxed)))
    }))
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    #[test]
    fn startup_keeps_changes_emitted_during_initial_read() {
        use std::cell::Cell;

        for (initial_value, changed_value) in [(2, 1), (1, 2), (2, 2)] {
            let subscribed = Cell::new(false);
            let mut portal_value = initial_value;
            let mut pending_signal = None;
            let initial = super::portal_startup(
                || subscribed.set(true),
                || {
                    let initial = portal_value;
                    portal_value = changed_value;
                    if subscribed.get() && portal_value != initial {
                        pending_signal = Some(portal_value);
                    }
                    initial
                },
            );
            let cached = pending_signal.unwrap_or(initial);
            assert_eq!(cached, changed_value, "portal changed during startup but cached {cached}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn portal_scheme_values() {
        assert_eq!(super::decode(0), None);
        assert_eq!(super::decode(1), Some(egui::Theme::Dark));
        assert_eq!(super::decode(2), Some(egui::Theme::Light));
        assert_eq!(super::decode(9), None);
        let nested = zbus::zvariant::Value::Value(Box::new(zbus::zvariant::Value::Value(Box::new(zbus::zvariant::Value::U32(2)))));
        assert_eq!(super::portal_code(zbus::zvariant::OwnedValue::try_from(nested).unwrap()), Some(2));
        assert_eq!(super::parse_gtk_scheme("'prefer-light'\n"), Some(egui::Theme::Light));
        assert_eq!(super::parse_gtk_scheme("'prefer-dark'\n"), Some(egui::Theme::Dark));
        // GNOME's "default" appearance is light, even though the portal reports no preference.
        assert_eq!(super::parse_gtk_scheme("'default'\n"), Some(egui::Theme::Light));
        assert_eq!(super::parse_gtk_scheme("default"), Some(egui::Theme::Light));
        assert_eq!(super::parse_gtk_scheme("'not-a-scheme'"), None);
    }

    /// #2798: a first read that misses the start-up wait still wakes the idle UI.
    #[cfg(target_os = "linux")]
    #[test]
    fn late_initial_read_requests_repaint() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let reader = super::start(move |publisher| {
            let _ = gate.recv();
            publisher.publish(Some(egui::Theme::Light));
            let _ = done.send(());
        })
        .expect("worker thread");
        let ctx = egui::Context::default();
        let repaints = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&repaints);
        ctx.set_request_repaint_callback(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(reader(&ctx), None, "start-up wait elapsed without an answer");
        release.send(()).unwrap();
        finished.recv().unwrap();
        assert_eq!(repaints.load(Ordering::Relaxed), 1);
        assert_eq!(reader(&ctx), Some(egui::Theme::Light));
    }
}
