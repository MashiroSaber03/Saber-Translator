//! Why a File › Open / Save dialog may never appear on Linux (#3354).
//!
//! `rfd` talks to the XDG file-chooser portal through `libdbus-1`, which it loads at runtime, and
//! falls back to `zenity`. When neither works (a minimal window-manager session without a portal
//! backend) it logs the failure and answers "nothing picked", which the app took for Cancel: the
//! menu item did nothing and said nothing. [`start`] looks for the pieces once, in the background,
//! and [`known_problem`] returns what to install when a dialog came back empty and none of them
//! is there.
//!
//! The probe only ever explains an empty answer; it never stops a dialog from opening. Anything
//! it can't tell counts as present, so a working system never sees the message.

use std::sync::OnceLock;

/// What the probe found. `None` means it couldn't tell, which counts as present.
// Only Linux probes; the types and helpers stay testable on every platform.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Findings {
    /// `libdbus-1.so.3` is in the linker cache (`rfd` loads it for the portal).
    pub libdbus: Option<bool>,
    /// The session bus has the portal's `FileChooser` interface (a portal and a backend for it).
    pub file_chooser_portal: Option<bool>,
    /// `zenity` is on `PATH` (`rfd`'s fallback).
    pub zenity: bool,
}

/// The message for `findings`, or `None` when some file dialog should work.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn problem(findings: Findings) -> Option<String> {
    let portal = findings.libdbus != Some(false) && findings.file_chooser_portal != Some(false);
    if portal || findings.zenity {
        return None;
    }
    let cause = if findings.libdbus == Some(false) { "libdbus is missing" } else { "no desktop portal offers a file chooser" };
    Some(format!(
        "No file dialog can open: {cause}. Install xdg-desktop-portal with a backend (xdg-desktop-portal-gtk, -kde or -gnome) and libdbus, or zenity. Meanwhile, drop a file onto the window or paste a copied image."
    ))
}

static RESULT: OnceLock<Option<String>> = OnceLock::new();

/// The problem found by [`start`], once it has finished.
pub fn known_problem() -> Option<String> {
    RESULT.get().cloned().flatten()
}

/// Whether `program` is an executable file in one of the `path` directories.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn on_path(program: &str, path: &std::ffi::OsStr) -> bool {
    std::env::split_paths(path).any(|dir| dir.join(program).is_file())
}

/// Look for the file dialog backends on a helper thread (the portal call can take a moment).
#[cfg(target_os = "linux")]
pub fn start() {
    let spawned = std::thread::Builder::new().name("file dialog probe".into()).spawn(|| {
        let findings = Findings {
            libdbus: crate::linux_libs::cached("libdbus-1.so.3"),
            file_chooser_portal: file_chooser_portal(),
            zenity: std::env::var_os("PATH").is_some_and(|path| on_path("zenity", &path)),
        };
        log::info!("file dialog backends: {findings:?}");
        let _ = RESULT.set(problem(findings));
    });
    if let Err(e) = spawned {
        log::warn!("couldn't start the file dialog probe: {e}");
    }
}

/// Whether the portal's `FileChooser` interface is on the session bus: `Some(false)` only when the
/// bus or the interface is definitely absent, `None` for anything inconclusive (a slow reply).
#[cfg(target_os = "linux")]
fn file_chooser_portal() -> Option<bool> {
    let Ok(conn) = zbus::blocking::connection::Builder::session().and_then(|b| b.method_timeout(std::time::Duration::from_secs(5)).build()) else {
        return Some(false);
    };
    let proxy = zbus::blocking::proxy::Builder::<zbus::blocking::Proxy<'_>>::new(&conn)
        .destination("org.freedesktop.portal.Desktop")
        .and_then(|b| b.path("/org/freedesktop/portal/desktop"))
        .and_then(|b| b.interface("org.freedesktop.portal.FileChooser"))
        .and_then(|b| b.cache_properties(zbus::proxy::CacheProperties::No).build())
        .ok()?;
    match proxy.get_property::<u32>("version") {
        Ok(_) => Some(true),
        Err(zbus::Error::MethodError(name, _, _)) => {
            let name = name.as_str();
            let absent = ["ServiceUnknown", "UnknownObject", "UnknownInterface", "UnknownProperty"].iter().any(|n| name.ends_with(n));
            absent.then_some(false)
        }
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(libdbus: Option<bool>, file_chooser_portal: Option<bool>, zenity: bool) -> Findings {
        Findings { libdbus, file_chooser_portal, zenity }
    }

    #[test]
    fn a_working_backend_is_not_reported() {
        assert_eq!(problem(found(Some(true), Some(true), false)), None, "portal");
        assert_eq!(problem(found(Some(false), Some(false), true)), None, "zenity alone");
        assert_eq!(problem(found(None, None, false)), None, "unknown counts as present");
        assert_eq!(problem(found(Some(true), None, false)), None, "slow portal reply");
    }

    #[test]
    fn nothing_to_show_a_dialog_with_is_explained() {
        let no_portal = problem(found(Some(true), Some(false), false)).unwrap();
        assert!(no_portal.contains("no desktop portal offers a file chooser") && no_portal.contains("xdg-desktop-portal"), "{no_portal}");
        let no_dbus = problem(found(Some(false), Some(true), false)).unwrap();
        assert!(no_dbus.contains("libdbus is missing"), "{no_dbus}");
    }

    #[test]
    fn programs_are_found_on_the_given_path() {
        let dir = std::env::temp_dir().join(format!("photocraft-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("zenity"), b"").unwrap();
        let path = std::env::join_paths([std::path::Path::new("/nonexistent"), dir.as_path()]).unwrap();
        assert!(on_path("zenity", &path));
        assert!(!on_path("kdialog", &path));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
