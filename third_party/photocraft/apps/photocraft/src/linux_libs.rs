//! Linux start-up check for the shared libraries winit and wgpu load at runtime (issue #201).
//!
//! The windowing (X11/Wayland, xkbcommon) and GPU (Vulkan, EGL) libraries aren't ELF `NEEDED`
//! entries: winit and wgpu `dlopen` them when the window opens, and some of those crates panic
//! when a library is missing (`xkbcommon-dl`: "libxkbcommon-x11.so could not be loaded"). So,
//! before eframe starts, we look for the libraries the current session type needs and, if one is
//! missing, print the package to install and exit non-zero instead of crashing (AGENTS.md, Never
//! crash).
//!
//! No `dlopen` here (that needs `unsafe`): we read the dynamic linker's cache (`ldconfig -p`) and
//! look in `LD_LIBRARY_PATH` and the usual library directories. Only the cache is conclusive: if
//! it can't be read (musl, NixOS, some containers) and a library isn't found in the directories,
//! that proves nothing, so we start silently. `PHOTOCRAFT_SKIP_LIB_CHECK=1` skips the check.
//!
//! The parsing and decision logic is pure and tested on every platform; only [`preflight`]
//! and [`available`] touch the system, and they are only called on Linux.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::path::{Path, PathBuf};

/// The display server winit will connect to, chosen the way winit 0.30 does: Wayland when
/// `WAYLAND_DISPLAY` or `WAYLAND_SOCKET` is set (non-empty), otherwise X11 when `DISPLAY` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplaySession {
    Wayland,
    X11,
    /// Neither is set: winit reports that itself (an error, not a panic).
    None,
}

pub fn session_from_env(var: impl Fn(&str) -> Option<String>) -> DisplaySession {
    let set = |name: &str| var(name).is_some_and(|v| !v.is_empty());
    if set("WAYLAND_DISPLAY") || set("WAYLAND_SOCKET") {
        DisplaySession::Wayland
    } else if set("DISPLAY") {
        DisplaySession::X11
    } else {
        DisplaySession::None
    }
}

/// A runtime-loaded library and the package that provides it on each distro family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lib {
    pub soname: &'static str,
    pub debian: &'static str,
    pub fedora: &'static str,
    pub arch: &'static str,
    pub suse: &'static str,
}

const fn lib(soname: &'static str, debian: &'static str, fedora: &'static str, arch: &'static str, suse: &'static str) -> Lib {
    Lib { soname, debian, fedora, arch, suse }
}

const XKBCOMMON: Lib = lib("libxkbcommon.so.0", "libxkbcommon0", "libxkbcommon", "libxkbcommon", "libxkbcommon0");
const XKBCOMMON_X11: Lib = lib("libxkbcommon-x11.so.0", "libxkbcommon-x11-0", "libxkbcommon-x11", "libxkbcommon-x11", "libxkbcommon-x11-0");
const X11: Lib = lib("libX11.so.6", "libx11-6", "libX11", "libx11", "libX11-6");
const X11_XCB: Lib = lib("libX11-xcb.so.1", "libx11-xcb1", "libX11-xcb", "libx11", "libX11-xcb1");
const XCB: Lib = lib("libxcb.so.1", "libxcb1", "libxcb", "libxcb", "libxcb1");
const XCURSOR: Lib = lib("libXcursor.so.1", "libxcursor1", "libXcursor", "libxcursor", "libXcursor1");
const XI: Lib = lib("libXi.so.6", "libxi6", "libXi", "libxi", "libXi6");
const WAYLAND_CLIENT: Lib = lib("libwayland-client.so.0", "libwayland-client0", "libwayland-client", "wayland", "libwayland-client0");
const VULKAN: Lib = lib("libvulkan.so.1", "libvulkan1", "vulkan-loader", "vulkan-icd-loader", "libvulkan1");
const EGL: Lib = lib("libEGL.so.1", "libegl1", "libglvnd-egl", "libglvnd", "libEGL1");

/// One requirement: at least one of these libraries must be present.
pub type AnyOf = &'static [Lib];

/// What winit 0.30 and wgpu `dlopen` for a session. X11: x11-dl opens Xlib, Xcursor, Xlib-xcb
/// and XInput2, x11rb opens libxcb, and xkbcommon-dl opens xkbcommon and xkbcommon-x11 (RandR
/// goes through x11rb's protocol code, so libXrandr isn't needed). Wayland: wayland-sys opens
/// libwayland-client (wayland-cursor is pure Rust) plus xkbcommon. wgpu needs Vulkan or EGL.
pub fn requirements(session: DisplaySession) -> Vec<AnyOf> {
    const GPU: AnyOf = &[VULKAN, EGL];
    match session {
        DisplaySession::Wayland => vec![&[WAYLAND_CLIENT], &[XKBCOMMON], GPU],
        DisplaySession::X11 => vec![&[X11], &[X11_XCB], &[XCB], &[XCURSOR], &[XI], &[XKBCOMMON], &[XKBCOMMON_X11], GPU],
        DisplaySession::None => Vec::new(),
    }
}

/// The `ldconfig -p` architecture tag for this build, when it's one we can tell apart from a
/// 32-bit library of the same name (`libc6,x86-64` vs plain `libc6` for i386).
fn ldconfig_arch_tag() -> Option<&'static str> {
    if cfg!(target_arch = "x86_64") {
        Some("x86-64")
    } else if cfg!(target_arch = "aarch64") {
        Some("AArch64")
    } else {
        None
    }
}

/// Does `ldconfig -p` output list `soname` for our architecture? Lines look like
/// `\tlibxkbcommon-x11.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libxkbcommon-x11.so.0`.
pub fn ldconfig_has(output: &str, soname: &str, arch_tag: Option<&str>) -> bool {
    output.lines().any(|line| {
        let line = line.trim_start();
        let Some((name, rest)) = line.split_once(' ') else { return false };
        if name != soname {
            return false;
        }
        match arch_tag {
            None => true,
            Some(tag) => {
                let flags = rest.split_once('(').and_then(|(_, r)| r.split_once(')')).map_or("", |(f, _)| f);
                flags.split(',').any(|f| f.trim() == tag)
            }
        }
    })
}

/// A real `ldconfig -p` listing starts with "N libs found in cache" (translated on a system whose
/// locale `LC_ALL=C` doesn't reach): a count, then the cache's name.
fn ldconfig_output_is_valid(output: &str) -> bool {
    output.lines().next().is_some_and(|l| l.split_whitespace().next().is_some_and(|n| n.parse::<u32>().is_ok()))
}

/// Where we look for libraries, and what we found there.
pub struct Probe {
    /// `ldconfig -p` output, when it ran and looked like a real cache listing.
    pub ldconfig: Option<String>,
    /// Directories to check for the soname (`LD_LIBRARY_PATH` first).
    pub dirs: Vec<PathBuf>,
}

impl Probe {
    fn has(&self, soname: &str, exists: &impl Fn(&Path) -> bool) -> bool {
        self.ldconfig.as_deref().is_some_and(|out| ldconfig_has(out, soname, ldconfig_arch_tag())) || self.dirs.iter().any(|d| exists(&d.join(soname)))
    }
}

/// The outcome of the check.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Ok,
    /// Requirements with none of their alternatives found. `conclusive` is true when the linker
    /// cache was read, so the libraries really are absent (and winit would fail or panic).
    Missing {
        missing: Vec<AnyOf>,
        conclusive: bool,
    },
}

pub fn check(session: DisplaySession, probe: &Probe, exists: impl Fn(&Path) -> bool) -> Verdict {
    let missing: Vec<AnyOf> = requirements(session).into_iter().filter(|any| !any.iter().any(|l| probe.has(l.soname, &exists))).collect();
    if missing.is_empty() { Verdict::Ok } else { Verdict::Missing { missing, conclusive: probe.ldconfig.is_some() } }
}

/// Distro family, from `/etc/os-release` `ID` and `ID_LIKE`, to name the right packages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distro {
    Debian,
    Fedora,
    Arch,
    Suse,
    Unknown,
}

pub fn distro_from_os_release(text: &str) -> Distro {
    let mut ids = Vec::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        if key.trim() == "ID" || key.trim() == "ID_LIKE" {
            ids.extend(value.trim().trim_matches(|c| c == '"' || c == '\'').split_whitespace().map(str::to_ascii_lowercase));
        }
    }
    let any = |names: &[&str]| ids.iter().any(|id| names.contains(&id.as_str()));
    if any(&["debian", "ubuntu"]) {
        Distro::Debian
    } else if any(&["fedora", "rhel", "centos"]) {
        Distro::Fedora
    } else if any(&["arch"]) {
        Distro::Arch
    } else if any(&["suse", "opensuse"]) || ids.iter().any(|id| id.starts_with("opensuse")) {
        Distro::Suse
    } else {
        Distro::Unknown
    }
}

fn package(lib: &Lib, distro: Distro) -> &'static str {
    match distro {
        Distro::Debian | Distro::Unknown => lib.debian,
        Distro::Fedora => lib.fedora,
        Distro::Arch => lib.arch,
        Distro::Suse => lib.suse,
    }
}

fn install_command(packages: &[&str], distro: Distro) -> String {
    let list = packages.join(" ");
    match distro {
        Distro::Debian => format!("sudo apt install {list}"),
        Distro::Fedora => format!("sudo dnf install {list}"),
        Distro::Arch => format!("sudo pacman -S {list}"),
        Distro::Suse => format!("sudo zypper install {list}"),
        Distro::Unknown => format!("Debian/Ubuntu: sudo apt install {list}"),
    }
}

/// The message for missing libraries: what's missing, and the command that installs it.
pub fn missing_message(session: DisplaySession, missing: &[AnyOf], distro: Distro) -> String {
    let session_name = match session {
        DisplaySession::Wayland => "Wayland",
        DisplaySession::X11 => "X11",
        DisplaySession::None => "display",
    };
    let names: Vec<String> = missing.iter().map(|any| any.iter().map(|l| l.soname).collect::<Vec<_>>().join(" or ")).collect();
    // For an either/or requirement, suggest its first alternative.
    let mut packages: Vec<&str> = Vec::new();
    for p in missing.iter().filter_map(|any| any.first()).map(|l| package(l, distro)) {
        if !packages.contains(&p) {
            packages.push(p);
        }
    }
    let mut msg = format!("photocraft: cannot start: the {session_name} session needs libraries that are not installed: {}.\n", names.join(", "));
    msg.push_str(&format!("Install them with:\n  {}\n", install_command(&packages, distro)));
    if distro == Distro::Unknown {
        let fedora: Vec<&str> = missing.iter().filter_map(|any| any.first()).map(|l| l.fedora).collect();
        msg.push_str(&format!("  Fedora: sudo dnf install {}\n", fedora.join(" ")));
    }
    msg.push_str("(Set PHOTOCRAFT_SKIP_LIB_CHECK=1 to start anyway.)\n");
    msg
}

/// Library directories to search: `LD_LIBRARY_PATH`, then the usual system and Flatpak paths.
fn search_dirs(ld_library_path: Option<&str>) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = ld_library_path.unwrap_or("").split(':').filter(|d| !d.is_empty()).map(PathBuf::from).collect();
    let triplet = if cfg!(target_arch = "aarch64") {
        "aarch64-linux-gnu"
    } else if cfg!(target_arch = "riscv64") {
        "riscv64-linux-gnu"
    } else {
        "x86_64-linux-gnu"
    };
    for d in ["/app/lib", "/usr/local/lib", "/usr/lib64", "/lib64", "/usr/lib", "/lib"] {
        dirs.push(PathBuf::from(d));
    }
    dirs.push(Path::new("/usr/lib").join(triplet));
    dirs.push(Path::new("/lib").join(triplet));
    dirs
}

/// `ldconfig -p`, from the usual locations (`/sbin` isn't on a normal user's PATH on Debian).
fn read_ldconfig() -> Option<String> {
    for cmd in ["/sbin/ldconfig", "/usr/sbin/ldconfig", "ldconfig"] {
        // The listing is parsed, so it must be the untranslated one: a German system prints
        // "5522 Bibliotheken im Cache", which `ldconfig_output_is_valid` took for no cache at all.
        if let Ok(out) = std::process::Command::new(cmd).arg("-p").env("LC_ALL", "C").output()
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout).into_owned();
            if ldconfig_output_is_valid(&text) {
                return Some(text);
            }
        }
    }
    None
}

/// This system's linker cache and library directories.
fn system_probe() -> Probe {
    Probe { ldconfig: read_ldconfig(), dirs: search_dirs(std::env::var("LD_LIBRARY_PATH").ok().as_deref()) }
}

/// Whether the linker cache lists `soname`; `None` when the cache can't be read, which proves
/// nothing (musl, NixOS, some containers).
pub fn cached(soname: &str) -> Option<bool> {
    read_ldconfig().map(|text| ldconfig_has(&text, soname, ldconfig_arch_tag()))
}

/// Are all the libraries `session` needs found on this system? (For choosing a display server:
/// libraries the check can't find count as missing.)
pub fn available(session: DisplaySession) -> bool {
    check(session, &system_probe(), |p| p.exists()) == Verdict::Ok
}

/// Run the check for the display server the window opens on against this system. `Err` carries
/// the message to print before exiting with a non-zero status.
pub fn preflight(session: DisplaySession) -> Result<(), String> {
    if std::env::var_os("PHOTOCRAFT_SKIP_LIB_CHECK").is_some_and(|v| !v.is_empty() && v != "0") {
        return Ok(());
    }
    if session == DisplaySession::None {
        return Ok(());
    }
    let verdict = check(session, &system_probe(), |p| p.exists());
    decide(session, verdict, || std::fs::read_to_string("/etc/os-release").map(|t| distro_from_os_release(&t)).unwrap_or(Distro::Unknown))
}

/// `Err` with the message to print only when the linker cache was read and the libraries are
/// really absent. Without the cache (musl has no `ldconfig -p`; NixOS, AppImages that ship their
/// own libraries) not finding a file proves nothing, so we start without a misleading warning.
fn decide(session: DisplaySession, verdict: Verdict, distro: impl FnOnce() -> Distro) -> Result<(), String> {
    match verdict {
        Verdict::Missing { missing, conclusive: true } => Err(missing_message(session, &missing, distro())),
        Verdict::Missing { conclusive: false, .. } | Verdict::Ok => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn session_follows_winit_order() {
        assert_eq!(session_from_env(env(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")])), DisplaySession::Wayland);
        assert_eq!(session_from_env(env(&[("WAYLAND_SOCKET", "3")])), DisplaySession::Wayland);
        assert_eq!(session_from_env(env(&[("WAYLAND_DISPLAY", ""), ("DISPLAY", ":99")])), DisplaySession::X11);
        assert_eq!(session_from_env(env(&[("DISPLAY", "")])), DisplaySession::None);
        assert_eq!(session_from_env(env(&[])), DisplaySession::None);
    }

    const CACHE: &str = "1234 libs found in cache `/etc/ld.so.cache'\n\
        \tlibxkbcommon.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libxkbcommon.so.0\n\
        \tlibxkbcommon-x11.so.0 (libc6) => /lib/i386-linux-gnu/libxkbcommon-x11.so.0\n\
        \tlibX11.so.6 (libc6,x86-64) => /lib/x86_64-linux-gnu/libX11.so.6\n\
        \tlibX11-xcb.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libX11-xcb.so.1\n\
        \tlibxcb.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libxcb.so.1\n\
        \tlibXcursor.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libXcursor.so.1\n\
        \tlibXi.so.6 (libc6,x86-64) => /lib/x86_64-linux-gnu/libXi.so.6\n\
        \tlibEGL.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libEGL.so.1\n\
        \tlibvulkan.so.1 (libc6,AArch64) => /lib/aarch64-linux-gnu/libvulkan.so.1\n";

    #[test]
    fn ldconfig_parsing_matches_name_and_architecture() {
        assert!(ldconfig_output_is_valid(CACHE));
        assert!(!ldconfig_output_is_valid("ldconfig: command not found"));
        // A translated listing is still a listing (it used to be rejected, so the check was
        // silently skipped on non-English systems).
        assert!(ldconfig_output_is_valid("5522 Bibliotheken im Cache »/etc/ld.so.cache« gefunden\n\tlibz.so.1 (libc6,x86-64) => /usr/lib/libz.so.1\n"));
        assert!(!ldconfig_output_is_valid("Error: kein Cache\n"));
        assert!(ldconfig_has(CACHE, "libxkbcommon.so.0", Some("x86-64")));
        // Only the 32-bit build is installed: doesn't count for a 64-bit app.
        assert!(!ldconfig_has(CACHE, "libxkbcommon-x11.so.0", Some("x86-64")));
        assert!(ldconfig_has(CACHE, "libxkbcommon-x11.so.0", None));
        assert!(ldconfig_has(CACHE, "libvulkan.so.1", Some("AArch64")));
        assert!(!ldconfig_has(CACHE, "libvulkan.so.1", Some("x86-64")));
        // Prefix of another name, or a missing name, is not a match.
        assert!(!ldconfig_has(CACHE, "libX11.so", None));
        assert!(!ldconfig_has(CACHE, "libwayland-client.so.0", None));
        assert!(!ldconfig_has("", "libX11.so.6", None));
        assert!(!ldconfig_has("garbage\n\t(\n)) =>", "libX11.so.6", Some("x86-64")));
    }

    fn cache_for_this_arch(names: &[&str]) -> String {
        let tag = ldconfig_arch_tag().map(|t| format!("libc6,{t}")).unwrap_or_else(|| "libc6".into());
        let mut s = format!("{} libs found in cache `/etc/ld.so.cache'\n", names.len());
        for n in names {
            s.push_str(&format!("\t{n} ({tag}) => /usr/lib/{n}\n"));
        }
        s
    }

    const ALL_X11: &[&str] =
        &["libX11.so.6", "libX11-xcb.so.1", "libxcb.so.1", "libXcursor.so.1", "libXi.so.6", "libxkbcommon.so.0", "libxkbcommon-x11.so.0", "libEGL.so.1"];

    #[test]
    fn x11_without_xkbcommon_x11_is_reported() {
        // Issue #201: Ubuntu server + Xvfb with only libxkbcommon0 installed.
        let names: Vec<&str> = ALL_X11.iter().copied().filter(|n| *n != "libxkbcommon-x11.so.0").collect();
        let probe = Probe { ldconfig: Some(cache_for_this_arch(&names)), dirs: vec![] };
        let v = check(DisplaySession::X11, &probe, |_| false);
        let Verdict::Missing { missing, conclusive } = v else { panic!("expected missing, got {v:?}") };
        assert!(conclusive);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0][0].soname, "libxkbcommon-x11.so.0");
        let msg = missing_message(DisplaySession::X11, &missing, Distro::Debian);
        assert!(msg.contains("libxkbcommon-x11.so.0"), "{msg}");
        assert!(msg.contains("sudo apt install libxkbcommon-x11-0"), "{msg}");
        assert!(msg.contains("PHOTOCRAFT_SKIP_LIB_CHECK"), "{msg}");
    }

    #[test]
    fn complete_systems_pass() {
        let probe = Probe { ldconfig: Some(cache_for_this_arch(ALL_X11)), dirs: vec![] };
        assert_eq!(check(DisplaySession::X11, &probe, |_| false), Verdict::Ok);
        let wayland = cache_for_this_arch(&["libwayland-client.so.0", "libxkbcommon.so.0", "libvulkan.so.1"]);
        let probe = Probe { ldconfig: Some(wayland), dirs: vec![] };
        assert_eq!(check(DisplaySession::Wayland, &probe, |_| false), Verdict::Ok);
        // Wayland doesn't need the X11 libraries; no session needs nothing.
        assert_eq!(check(DisplaySession::None, &Probe { ldconfig: None, dirs: vec![] }, |_| false), Verdict::Ok);
    }

    #[test]
    fn gpu_needs_vulkan_or_egl() {
        let probe = Probe { ldconfig: Some(cache_for_this_arch(&["libwayland-client.so.0", "libxkbcommon.so.0"])), dirs: vec![] };
        let Verdict::Missing { missing, .. } = check(DisplaySession::Wayland, &probe, |_| false) else { panic!("expected missing") };
        assert_eq!(missing.len(), 1);
        let msg = missing_message(DisplaySession::Wayland, &missing, Distro::Fedora);
        assert!(msg.contains("libvulkan.so.1 or libEGL.so.1"), "{msg}");
        assert!(msg.contains("sudo dnf install vulkan-loader"), "{msg}");
    }

    #[test]
    fn library_dirs_count_and_unread_cache_is_inconclusive() {
        let dirs = search_dirs(Some("/opt/x/lib::/nix/store/abc/lib"));
        assert_eq!(dirs[0], PathBuf::from("/opt/x/lib"));
        assert_eq!(dirs[1], PathBuf::from("/nix/store/abc/lib"));
        // Found by file in a directory (no ldconfig, as on NixOS).
        let probe = Probe { ldconfig: None, dirs: vec![PathBuf::from("/nix/lib")] };
        let present = ["/nix/lib/libwayland-client.so.0", "/nix/lib/libxkbcommon.so.0", "/nix/lib/libEGL.so.1"];
        assert_eq!(check(DisplaySession::Wayland, &probe, |p| present.iter().any(|q| Path::new(q) == p)), Verdict::Ok);
        // Not found and no cache to confirm it: inconclusive (the app starts, see #1144).
        let v = check(DisplaySession::Wayland, &probe, |_| false);
        let Verdict::Missing { missing, conclusive } = v else { panic!("expected missing") };
        assert!(!conclusive);
        let msg = missing_message(DisplaySession::Wayland, &missing, Distro::Unknown);
        assert!(msg.contains("apt install libwayland-client0 libxkbcommon0 libvulkan1"), "{msg}");
        assert!(msg.contains("dnf install libwayland-client libxkbcommon vulkan-loader"), "{msg}");
    }

    #[test]
    fn unreadable_cache_never_warns_or_fails() {
        // Issue #1144: Alpine (musl) has no `ldconfig -p`, so a library that exists was reported missing.
        let probe = Probe { ldconfig: None, dirs: vec![] };
        let verdict = check(DisplaySession::X11, &probe, |_| false);
        assert_eq!(decide(DisplaySession::X11, verdict, || Distro::Debian), Ok(()));
        // A readable cache that lacks a library is still an error.
        let names: Vec<&str> = ALL_X11.iter().copied().filter(|n| *n != "libxkbcommon-x11.so.0").collect();
        let probe = Probe { ldconfig: Some(cache_for_this_arch(&names)), dirs: vec![] };
        let verdict = check(DisplaySession::X11, &probe, |_| false);
        let Err(msg) = decide(DisplaySession::X11, verdict, || Distro::Debian) else { panic!("expected an error") };
        assert!(msg.contains("libxkbcommon-x11.so.0"), "{msg}");
    }

    #[test]
    fn distro_detection() {
        assert_eq!(distro_from_os_release("NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian\n"), Distro::Debian);
        assert_eq!(distro_from_os_release("ID=linuxmint\nID_LIKE=\"ubuntu debian\"\n"), Distro::Debian);
        assert_eq!(distro_from_os_release("ID=fedora\n"), Distro::Fedora);
        assert_eq!(distro_from_os_release("ID=\"rocky\"\nID_LIKE=\"rhel centos fedora\"\n"), Distro::Fedora);
        assert_eq!(distro_from_os_release("ID=endeavouros\nID_LIKE=arch\n"), Distro::Arch);
        assert_eq!(distro_from_os_release("ID=\"opensuse-tumbleweed\"\nID_LIKE=\"opensuse suse\"\n"), Distro::Suse);
        assert_eq!(distro_from_os_release("ID=nixos\n"), Distro::Unknown);
        assert_eq!(distro_from_os_release(""), Distro::Unknown);
        assert_eq!(distro_from_os_release("=\n==\nID\n"), Distro::Unknown);
    }
}
