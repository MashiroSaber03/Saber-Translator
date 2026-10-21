//! The displays and their ICC profiles, for the colour-managed canvas (Edit › Color Settings ›
//! Monitor Profile = `auto`; #569).
//!
//! macOS: every `NSScreen` with its `CGDirectDisplayID`, name, frame and
//! `colorSpace.ICCProfileData`, documented AppKit APIs read through `osascript` (AppKit via
//! AppleScriptObjC) so the app needs no `unsafe` FFI. A read takes about 0.4 s on a background
//! thread; the shell starts one at launch and again when the displays may have changed
//! (`photocraft_ui_egui::monitor_status`), and each window then uses the profile of the display
//! it is on. Linux X11: RandR output names and CRTC frames are matched to colord display devices
//! by the exact `XRANDR_name` metadata; cloned outputs get a profile only when every output has
//! the same ICC bytes. Native Wayland, including Xwayland windows opened from a Wayland session,
//! reports no display profiles: eframe/winit has no safe way to set the app surface's Wayland
//! image description, so applying the output ICC in PhotoCraft could double-transform it.
//! Other platforms report no displays (sRGB, or the profile chosen in Color Settings).

use std::sync::mpsc::Receiver;

use photocraft_engine::display_color::Display;

/// What the platform reader returns: the displays, or why there are none.
pub type Detection = Result<Vec<Display>, String>;

/// Starts reading the displays in the background; `None` when the platform has no reader.
#[cfg(not(target_os = "linux"))]
pub fn detect_async() -> Option<Receiver<Detection>> {
    #[cfg(target_os = "macos")]
    {
        Some(spawn_reader(detect))
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Linux display profile reads are supported only for an X11 session. In particular, an app
/// window opened through Xwayland still belongs to the Wayland desktop and must not apply a
/// compositor-managed output profile a second time.
#[cfg(target_os = "linux")]
pub fn detect_async_for_session(session: crate::linux_libs::DisplaySession) -> Option<Receiver<Detection>> {
    (session == crate::linux_libs::DisplaySession::X11).then(|| spawn_reader(detect))
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn spawn_reader(detect: fn() -> Detection) -> Receiver<Detection> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(detect());
    });
    rx
}

/// One line per screen, primary (menu-bar) screen first: id, frame (x, y, width, height in
/// points, Cocoa's bottom-left origin), the profile as base64, the profile's name in System
/// Settings and the display's name, tab-separated. Numbers as integers (text conversion of
/// reals follows the user's locale, e.g. "0,0"). The id goes through NSNumber's `stringValue`:
/// a display id above AppleScript's integer range (2^29) is otherwise coerced to a real and
/// printed as "2.077748985E+9" (#1371).
#[cfg(target_os = "macos")]
const SCRIPT: &str = r#"use framework "AppKit"
set out to ""
repeat with s in (current application's NSScreen's screens() as list)
  set f to s's frame()
  set n to (((s's deviceDescription()'s objectForKey:"NSScreenNumber")'s stringValue()) as text)
  set icc to ""
  set cs to s's colorSpace()
  set csName to ""
  if cs is not missing value then
    set csName to ((cs's localizedName()) as text)
    set d to cs's ICCProfileData()
    if d is not missing value then set icc to ((d's base64EncodedStringWithOptions:0) as text)
  end if
  set out to out & n & tab & ((item 1 of item 1 of f) as integer) & tab & ((item 2 of item 1 of f) as integer) & tab & ((item 1 of item 2 of f) as integer) & tab & ((item 2 of item 2 of f) as integer) & tab & icc & tab & csName & tab & ((s's localizedName()) as text) & linefeed
end repeat
return out"#;

#[cfg(target_os = "macos")]
fn detect() -> Detection {
    let mut cmd = std::process::Command::new("/usr/bin/osascript");
    cmd.args(["-e", SCRIPT]);
    parse_reply(&run_helper(&mut cmd, HELPER_TIMEOUT)?)
}

/// How long a reading may take before the helper is stopped (it normally takes about 0.4 s).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const HELPER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Run the helper and return its standard output; a helper that fails, or runs longer than
/// `timeout`, is an error (and is killed and reaped), so a hung reading can't block later ones.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn run_helper(cmd: &mut std::process::Command, timeout: std::time::Duration) -> Result<String, String> {
    use std::io::Read;
    use std::process::Stdio;
    let mut child =
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("couldn't run the display profile reader: {e}"))?;
    // Drain both pipes while waiting: a reply larger than the pipe buffer would otherwise block
    // the helper until the deadline.
    let drain = |p: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = p {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let err = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if std::time::Instant::now() < deadline => std::thread::sleep(std::time::Duration::from_millis(20)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("reading the display profiles took longer than {} s and was stopped", timeout.as_secs()));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("waiting for the display profile reader failed: {e}"));
            }
        }
    };
    let out = out.join().unwrap_or_default();
    let err = err.join().unwrap_or_default();
    if !status.success() {
        let err = String::from_utf8_lossy(&err);
        let first = err.lines().next().unwrap_or("").trim();
        return Err(format!("reading the display profiles failed ({status}){}", if first.is_empty() { String::new() } else { format!(": {first}") }));
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

#[cfg(target_os = "linux")]
fn detect() -> Detection {
    linux::detect()
}

#[cfg(target_os = "linux")]
mod linux {
    use std::collections::{BTreeMap, HashMap};
    use std::io::Read as _;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    use photocraft_engine::display_color::Display;
    use x11rb::connection::Connection as _;
    use x11rb::protocol::randr;
    use zbus::blocking::{connection::Builder, proxy::Proxy};
    use zbus::zvariant::OwnedObjectPath;

    use super::Detection;

    const MAX_ICC_BYTES: usize = 64 * 1024 * 1024;
    const COLORD: &str = "org.freedesktop.ColorManager";
    const COLORD_ROOT: &str = "/org/freedesktop/ColorManager";
    const COLORD_MANAGER: &str = "org.freedesktop.ColorManager";
    const COLORD_DEVICE: &str = "org.freedesktop.ColorManager.Device";
    const COLORD_PROFILE: &str = "org.freedesktop.ColorManager.Profile";

    struct IccProfile {
        name: String,
        bytes: Arc<Vec<u8>>,
    }

    pub(super) fn detect() -> Detection {
        let (connection, screen) = x11rb::connect(None).map_err(|e| format!("couldn't connect to the X11 display: {e}"))?;
        let root = connection.setup().roots.get(screen).ok_or_else(|| "the X11 server didn't provide a screen".to_string())?.root;
        let resources = randr::get_screen_resources_current(&connection, root)
            .map_err(|e| format!("couldn't request XRandR outputs: {e}"))?
            .reply()
            .map_err(|e| format!("couldn't read XRandR outputs: {e}"))?;

        let mut output_names = HashMap::new();
        let mut active_crtcs = BTreeMap::new();
        for output in resources.outputs {
            let info = match randr::get_output_info(&connection, output, resources.config_timestamp) {
                Ok(cookie) => match cookie.reply() {
                    Ok(info) => info,
                    Err(e) => {
                        log::debug!("couldn't read one XRandR output: {e}");
                        continue;
                    }
                },
                Err(e) => {
                    log::debug!("couldn't request one XRandR output: {e}");
                    continue;
                }
            };
            if info.connection != randr::Connection::CONNECTED || info.crtc == 0 {
                continue;
            }
            let Ok(name) = String::from_utf8(info.name) else {
                log::debug!("skipping an XRandR output with a non-UTF-8 name");
                continue;
            };
            if !name.is_empty() {
                output_names.insert(output, name);
                active_crtcs.insert(info.crtc, ());
            }
        }

        let profiles = match colord_profiles() {
            Ok(profiles) => profiles,
            Err(e) => {
                log::warn!("couldn't read X11 display profiles from colord: {e}");
                HashMap::new()
            }
        };
        let mut displays = Vec::with_capacity(active_crtcs.len());
        for crtc in active_crtcs.into_keys() {
            let crtc_info = match randr::get_crtc_info(&connection, crtc, resources.config_timestamp) {
                Ok(cookie) => match cookie.reply() {
                    Ok(info) if info.status == randr::SetConfig::SUCCESS && info.width > 0 && info.height > 0 => info,
                    Ok(_) => continue,
                    Err(e) => {
                        log::debug!("couldn't read XRandR CRTC {crtc}: {e}");
                        continue;
                    }
                },
                Err(e) => {
                    log::debug!("couldn't request XRandR CRTC {crtc}: {e}");
                    continue;
                }
            };
            let Some(output_names) = crtc_info.outputs.iter().map(|output| output_names.get(output).cloned()).collect::<Option<Vec<_>>>() else { continue };
            let Some(primary_name) = output_names.first() else { continue };
            let profile = unambiguous_clone_profile(&output_names, &profiles);
            displays.push(Display {
                id: crtc,
                name: primary_name.clone(),
                frame: [f64::from(crtc_info.x), f64::from(crtc_info.y), f64::from(crtc_info.width), f64::from(crtc_info.height)],
                profile_name: profile.as_ref().map(|p| p.name.clone()),
                icc: profile.map(|p| p.bytes),
            });
        }
        if displays.is_empty() {
            return Err("X11 reported no active RandR outputs".into());
        }
        Ok(displays)
    }

    fn colord_profiles() -> Result<HashMap<String, Vec<IccProfile>>, String> {
        let connection = Builder::system()
            .map_err(|e| format!("couldn't connect to the system bus: {e}"))?
            .method_timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("couldn't connect to the system bus: {e}"))?;
        let manager = Proxy::new(&connection, COLORD, COLORD_ROOT, COLORD_MANAGER).map_err(|e| format!("couldn't access colord: {e}"))?;
        let devices: Vec<OwnedObjectPath> =
            manager.call("GetDevicesByKind", &("display",)).map_err(|e| format!("colord couldn't list display devices: {e}"))?;
        let mut profiles: HashMap<String, Vec<IccProfile>> = HashMap::new();
        for device_path in devices {
            let device = Proxy::new(&connection, COLORD, &device_path, COLORD_DEVICE).map_err(|e| format!("couldn't access a colord display device: {e}"))?;
            let metadata: HashMap<String, String> = device.get_property("Metadata").map_err(|e| format!("couldn't read colord display metadata: {e}"))?;
            let Some(output_name) = metadata.get("XRANDR_name").filter(|name| !name.is_empty()) else { continue };
            // `*` is colord's default profile qualifier. A display without one is kept, with no
            // ICC, so the engine reports the normal sRGB fallback.
            let profile_path: OwnedObjectPath = match device.call("GetProfileForQualifiers", &(vec!["*"],)) {
                Ok(path) => path,
                Err(_) => continue,
            };
            let profile = Proxy::new(&connection, COLORD, &profile_path, COLORD_PROFILE).map_err(|e| format!("couldn't access a colord ICC profile: {e}"))?;
            let filename: String = match profile.get_property::<String>("Filename") {
                Ok(filename) if !filename.is_empty() => filename,
                _ => continue,
            };
            let Some(profile) = read_icc(Path::new(&filename)) else {
                log::warn!("couldn't read a valid ICC profile assigned to X11 output {output_name}");
                continue;
            };
            profiles.entry(output_name.clone()).or_default().push(profile);
        }
        Ok(profiles)
    }

    fn read_icc(path: &Path) -> Option<IccProfile> {
        let mut bytes = Vec::new();
        std::fs::File::open(path).ok()?.take((MAX_ICC_BYTES + 1) as u64).read_to_end(&mut bytes).ok()?;
        let bytes = checked_icc_bytes(bytes)?;
        let name = path.file_name()?.to_str()?.to_owned();
        Some(IccProfile { name, bytes: Arc::new(bytes) })
    }

    fn checked_icc_bytes(mut bytes: Vec<u8>) -> Option<Vec<u8>> {
        if bytes.len() > MAX_ICC_BYTES || bytes.len() < 132 || bytes.get(36..40) != Some(b"acsp") {
            return None;
        }
        let declared = u32::from_be_bytes(bytes.get(0..4)?.try_into().ok()?) as usize;
        if declared < 132 || declared > bytes.len() {
            return None;
        }
        bytes.truncate(declared);
        let profile = photocraft_cms::Profile::parse(&bytes).ok()?;
        (profile.color_space == photocraft_cms::ColorSpace::Rgb).then_some(bytes)
    }

    fn unambiguous_clone_profile(output_names: &[String], profiles: &HashMap<String, Vec<IccProfile>>) -> Option<IccProfile> {
        let mut selected: Option<&IccProfile> = None;
        for output in output_names {
            let candidates = profiles.get(output)?;
            let first = candidates.first()?;
            if candidates.iter().any(|candidate| candidate.bytes != first.bytes) {
                return None;
            }
            if selected.is_some_and(|profile| profile.bytes != first.bytes) {
                return None;
            }
            selected = Some(first);
        }
        selected.map(|profile| IccProfile { name: profile.name.clone(), bytes: profile.bytes.clone() })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn checked_icc_bytes_require_a_bounded_parseable_rgb_profile() {
            let rgb = photocraft_cms::Builtin::Srgb.profile().to_bytes();
            let bytes = rgb.as_ref().clone();
            assert_eq!(checked_icc_bytes(bytes.clone()), Some(bytes));

            let cmyk = photocraft_cms::Builtin::CoatedCmyk.profile().to_bytes();
            assert!(checked_icc_bytes(cmyk.as_ref().clone()).is_none());
            assert!(checked_icc_bytes(vec![0; 132]).is_none());
            assert!(checked_icc_bytes(vec![0; MAX_ICC_BYTES + 1]).is_none());
        }

        #[test]
        fn cloned_outputs_use_only_one_unambiguous_profile() {
            let srgb = photocraft_cms::Builtin::Srgb.profile().to_bytes();
            let p3 = photocraft_cms::Builtin::DisplayP3.profile().to_bytes();
            let profile = |name: &str, bytes: std::sync::Arc<Vec<u8>>| IccProfile { name: name.into(), bytes };
            let outputs = ["DP-1".to_owned(), "HDMI-1".to_owned()];

            let matching =
                HashMap::from([("DP-1".to_owned(), vec![profile("sRGB", srgb.clone())]), ("HDMI-1".to_owned(), vec![profile("sRGB", srgb.clone())])]);
            assert_eq!(unambiguous_clone_profile(&outputs, &matching).map(|p| p.name), Some("sRGB".into()));

            let differing =
                HashMap::from([("DP-1".to_owned(), vec![profile("sRGB", srgb.clone())]), ("HDMI-1".to_owned(), vec![profile("Display P3", p3.clone())])]);
            assert!(unambiguous_clone_profile(&outputs, &differing).is_none());

            let missing = HashMap::from([("DP-1".to_owned(), vec![profile("sRGB", srgb.clone())])]);
            assert!(unambiguous_clone_profile(&outputs, &missing).is_none());

            let conflicting = HashMap::from([("DP-1".to_owned(), vec![profile("sRGB", srgb), profile("Display P3", p3)])]);
            assert!(unambiguous_clone_profile(&["DP-1".to_owned()], &conflicting).is_none());
        }
    }
}

/// The helper's reply ([`SCRIPT`]) → displays, frames flipped to a top-left origin with y down
/// (the coordinates window positions use). Lines that don't parse are skipped.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_reply(out: &str) -> Detection {
    let mut displays = Vec::new();
    let mut primary_height = None;
    for line in out.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.splitn(8, '\t').collect();
        let [id, x, y, w, h, b64, profile_name, name] = f.as_slice() else { continue };
        let num = |s: &str| s.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        let (Some(id), Some(x), Some(y), Some(w), Some(h)) = (display_id(id), num(x), num(y), num(w), num(h)) else { continue };
        // Cocoa frames are relative to the primary screen's bottom-left corner, y up.
        let top = *primary_height.get_or_insert(h) - (y + h);
        let name = name.trim();
        let name = if name.is_empty() { format!("Display {id}") } else { name.to_string() };
        let b64 = b64.trim();
        // An ICC profile starts with its size and carries `acsp` at offset 36; anything else is
        // treated as no profile (the engine then says the display has none).
        let icc = (!b64.is_empty()).then(|| base64_decode(b64)).flatten().filter(|b| b.len() >= 132 && b.get(36..40) == Some(b"acsp"));
        let profile_name = Some(profile_name.trim().to_string()).filter(|n| !n.is_empty());
        displays.push(Display { id, name, frame: [x, top, w, h], profile_name, icc: icc.map(std::sync::Arc::new) });
    }
    if displays.is_empty() {
        // Only the start of the first field: a display line carries a whole base64 ICC profile,
        // which must not end up in a notice (#1371).
        let first = out.lines().next().unwrap_or("").split('\t').next().unwrap_or("").trim();
        let mut shown: String = first.chars().take(80).collect();
        if shown.len() < first.len() {
            shown.push('…');
        }
        return Err(if shown.is_empty() { "the display reader returned nothing".into() } else { format!("couldn't read the display list: {shown}") });
    }
    Ok(displays)
}

/// A display id: a plain integer, or the scientific notation AppleScript printed for ids above
/// its integer range before the helper asked for `stringValue` (#1371), when it is exact.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn display_id(s: &str) -> Option<u32> {
    let s = s.trim();
    s.parse::<u32>().ok().or_else(|| {
        let v = s.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0 && *v <= f64::from(u32::MAX))?;
        Some(v as u32)
    })
}

/// Standard base64 (RFC 4648, with padding) → bytes; `None` on any invalid character.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let s = s.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    for chunk in s.chunks(4) {
        let mut acc = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            acc |= val(*c)? << (18 - 6 * i);
        }
        let n = match chunk.len() {
            4 => 3,
            3 => 2,
            2 => 1,
            _ => return None,
        };
        out.extend_from_slice(&acc.to_be_bytes()[1..1 + n]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64() {
        assert_eq!(base64_decode("aGVsbG8=").as_deref(), Some(&b"hello"[..]));
        assert_eq!(base64_decode("aGVsbG8h").as_deref(), Some(&b"hello!"[..]));
        assert_eq!(base64_decode("aGk=").as_deref(), Some(&b"hi"[..]));
        assert_eq!(base64_decode("").as_deref(), Some(&b""[..]));
        assert!(base64_decode("a").is_none());
        assert!(base64_decode("a$==").is_none());
    }

    #[test]
    fn helper_replies() {
        // Only the ICC signature is checked here; parsing is the engine's job.
        let mut icc = vec![0u8; 132];
        icc[36..40].copy_from_slice(b"acsp");
        let b64 = base64_encode(&icc);
        // The #569 setup: built-in display primary, the external one right of it and higher up.
        let reply = format!(
            "1\t0\t0\t1728\t1117\t{b64}\tColor LCD\tBuilt-in Retina Display\n4\t1728\t92\t3008\t1692\t{b64}\tApple_Display_26-04-13.icc\tROG PG32UQX\n"
        );
        let d = parse_reply(&reply).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!((d[0].id, d[0].name.as_str(), d[0].frame), (1, "Built-in Retina Display", [0.0, 0.0, 1728.0, 1117.0]));
        assert_eq!((d[1].id, d[1].frame), (4, [1728.0, -667.0, 3008.0, 1692.0]));
        assert_eq!(d[1].profile_name.as_deref(), Some("Apple_Display_26-04-13.icc"));
        assert_eq!(d[1].icc.as_deref(), Some(&icc));
        // No profile, a bad profile, no name: kept, without a profile or with a fallback name.
        let d = parse_reply("2\t0\t0\t800\t600\t\t\t\n3\t800\t0\t800\t600\taGVsbG8=\tX\tY\n").unwrap();
        assert_eq!((d[0].name.as_str(), d[0].icc.is_none(), d[0].profile_name.is_none()), ("Display 2", true, true));
        assert!(d[1].icc.is_none());
        // Garbage lines are skipped; nothing usable is an error.
        assert_eq!(parse_reply("junk\n5\t0\t0\t10\t10\t\t\tZ").unwrap().len(), 1);
        assert!(parse_reply("").is_err());
        assert!(parse_reply("execution error: -1728").unwrap_err().contains("-1728"));
    }

    #[test]
    fn large_display_ids_and_short_errors() {
        // #1371: a CGDirectDisplayID above AppleScript's integer range came back as a real.
        assert_eq!(display_id("2077748985"), Some(2_077_748_985));
        assert_eq!(display_id(" 2.077748985E+9 "), Some(2_077_748_985));
        assert_eq!(display_id("4.294967295E+9"), Some(u32::MAX));
        for bad in ["4.3E+9", "-1", "1.5", "NaN", "inf", "", "x"] {
            assert_eq!(display_id(bad), None, "{bad}");
        }
        let d = parse_reply("2.077748985E+9\t0\t0\t1512\t982\t\tColor LCD\tBuilt-in Retina Display\n").unwrap();
        assert_eq!(d[0].id, 2_077_748_985);
        // A line that can't be read never puts its profile into the error.
        let profile = "A".repeat(4000);
        let e = parse_reply(&format!("bad id\t0\t0\t1\t1\t{profile}\tColor LCD\tX\n")).unwrap_err();
        assert!(e.contains("bad id") && !e.contains("AAAA") && e.len() < 120, "{e}");
        let e = parse_reply(&"Z".repeat(500)).unwrap_err();
        assert!(e.chars().count() < 120 && e.ends_with('…'), "{e}");
    }

    // Drives `sleep` (found on PATH: NixOS and the Nix build sandbox have no /bin/sleep) and
    // /bin/sh; the helper itself only runs on macOS.
    #[cfg(unix)]
    #[test]
    fn a_hung_helper_is_stopped_and_big_replies_dont_block() {
        let t0 = std::time::Instant::now();
        let e = run_helper(std::process::Command::new("sleep").arg("30"), std::time::Duration::from_millis(300)).unwrap_err();
        assert!(e.contains("took longer than"), "{e}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "killed at the deadline");
        // 1 MB on stdout (far beyond a pipe buffer) arrives whole.
        let big = run_helper(std::process::Command::new("/bin/sh").args(["-c", "head -c 1048576 /dev/zero | tr '\\0' a"]), std::time::Duration::from_secs(20))
            .unwrap();
        assert_eq!(big.len(), 1 << 20);
        let e = run_helper(std::process::Command::new("/bin/sh").args(["-c", "echo boom >&2; exit 3"]), std::time::Duration::from_secs(5)).unwrap_err();
        assert!(e.contains("boom"), "{e}");
        assert!(run_helper(&mut std::process::Command::new("/nonexistent/helper"), std::time::Duration::from_secs(1)).is_err());
    }

    fn base64_encode(b: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in b.chunks(3) {
            let n = c.iter().enumerate().fold(0u32, |n, (i, v)| n | u32::from(*v) << (16 - 8 * i));
            for i in 0..=c.len() {
                s.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
            }
        }
        while !s.len().is_multiple_of(4) {
            s.push('=');
        }
        s
    }

    /// On a Mac, every display is read, and the profiles parse as RGB profiles.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn detected_displays_parse() {
        if let Ok(displays) = detect() {
            for d in displays {
                let Some(icc) = d.icc else { continue };
                let p = photocraft_engine::color_cmds::profile_from_bytes(&icc).expect("parses");
                assert_eq!(format!("{:?}", p.color_space), "Rgb", "{}: {}", d.name, p.description);
            }
        }
    }
}
