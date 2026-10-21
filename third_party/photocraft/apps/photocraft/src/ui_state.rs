//! The saved window geometry and egui layout (`ui.ron`, written by eframe's persistence).
//!
//! eframe restores this state while it creates the window, before PhotoCraft's code runs, and
//! some values it does not check abort startup: a zoom factor of zero, NaN or 1e30 fails a
//! `dpi` assertion, and a NaN or huge window size fails surface creation. A truncated or
//! hand-edited file would then crash every launch. [`sanitize`] runs first and removes a file
//! holding such values (the window and panels open at their defaults; nothing else is lost).
//! A file that isn't RON at all is left alone: eframe already ignores it.

use std::path::Path;

/// Window positions, sizes and panel rects beyond this many points or pixels are corrupt.
const MAX_COORD: f64 = 1.0e6;
/// egui's zoom factor is the UI scale over the display scale (75% on a 2× display is 0.375,
/// 300% on a 1× display is 3).
const ZOOM: std::ops::RangeInclusive<f64> = 0.1..=10.0;

/// Remove `path` if it holds state that would crash startup. Never fails: errors are logged.
pub fn sanitize(path: Option<&Path>) {
    let Some(path) = path else { return };
    let Ok(text) = std::fs::read_to_string(path) else { return };
    if let Err(why) = check(&text) {
        log::warn!("ignoring saved window layout {}: {why}", path.display());
        if let Err(e) = std::fs::remove_file(path) {
            log::warn!("could not remove {}: {e}", path.display());
        }
    }
}

/// Whether `path` holds a window position from an earlier session, which eframe restores. Then
/// `NativeOptions::centered` must be off: eframe applies it after restoring the saved geometry,
/// so it moved the window back to the centre of the main monitor on every launch (#4826).
pub fn has_saved_position(path: Option<&Path>) -> bool {
    let Some(text) = path.and_then(|p| std::fs::read_to_string(p).ok()) else { return false };
    saved_position(&text)
}

fn saved_position(text: &str) -> bool {
    let Ok(kv) = ron::from_str::<std::collections::HashMap<String, String>>(text) else { return false };
    let Some(window) = kv.get("window").and_then(|w| ron::from_str::<ron::Value>(w).ok()) else { return false };
    // eframe restores the inner position on macOS and the outer position elsewhere
    // (`egui_winit::WindowSettings::initialize_viewport_builder`).
    let key = if cfg!(target_os = "macos") { "inner_position_pixels" } else { "outer_position_pixels" };
    matches!(field(&window, key), Some(ron::Value::Option(Some(_))))
}

/// `Err` names the first value that would crash startup.
fn check(text: &str) -> Result<(), String> {
    // eframe stores a map of key → RON string; a file that doesn't parse is ignored by eframe.
    let Ok(kv) = ron::from_str::<std::collections::HashMap<String, String>>(text) else { return Ok(()) };
    for (key, value) in &kv {
        // eframe also skips an entry that doesn't parse.
        let Ok(v) = ron::from_str::<ron::Value>(value) else { continue };
        floats_sane(&v, 0).map_err(|e| format!("{key}: {e}"))?;
        let zoom = (key == "egui").then(|| field(&v, "options").and_then(|o| field(o, "zoom_factor")).and_then(as_f64)).flatten();
        if let Some(z) = zoom.filter(|z| !ZOOM.contains(z)) {
            return Err(format!("egui: zoom factor {z}"));
        }
    }
    Ok(())
}

/// Every float finite and within [`MAX_COORD`]; depth-limited (the file is untrusted).
fn floats_sane(v: &ron::Value, depth: usize) -> Result<(), String> {
    if depth > 64 {
        return Err("nested too deeply".into());
    }
    match v {
        ron::Value::Number(_) => match as_f64(v) {
            Some(f) if !f.is_finite() || f.abs() > MAX_COORD => Err(format!("value {f}")),
            _ => Ok(()),
        },
        ron::Value::Option(Some(inner)) => floats_sane(inner, depth + 1),
        ron::Value::Seq(items) => items.iter().try_for_each(|i| floats_sane(i, depth + 1)),
        ron::Value::Map(m) => m.iter().try_for_each(|(k, v)| floats_sane(k, depth + 1).and_then(|()| floats_sane(v, depth + 1))),
        _ => Ok(()),
    }
}

/// A float (integers such as egui's hashed ids are not coordinates).
fn as_f64(v: &ron::Value) -> Option<f64> {
    match v {
        ron::Value::Number(ron::value::Number::F32(f)) => Some(f64::from(f.get())),
        ron::Value::Number(ron::value::Number::F64(f)) => Some(f.get()),
        _ => None,
    }
}

fn field<'a>(v: &'a ron::Value, name: &str) -> Option<&'a ron::Value> {
    match v {
        ron::Value::Map(m) => m.get(&ron::Value::String(name.into())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{check, saved_position};

    fn file(window: &str, zoom: &str) -> String {
        let egui = format!("(options:(zoom_factor:{zoom},max_passes:2),data:([(18446744073709551615,(ron:\"(x:1.0)\"))]))");
        format!("{{\"window\":{window:?},\"egui\":{egui:?}}}")
    }

    const WINDOW: &str = "(inner_position_pixels:Some((x:616.0,y:430.0)),fullscreen:false,inner_size_points:Some((x:1100.0,y:760.0)))";

    #[test]
    fn a_saved_layout_is_kept() {
        assert_eq!(check(&file(WINDOW, "1.0")), Ok(()));
        assert_eq!(check(&file(WINDOW, "0.375")), Ok(()));
    }

    #[test]
    fn values_that_abort_startup_are_rejected() {
        // Each of these aborted the app at launch before the check (dpi / wgpu assertions).
        for zoom in ["0.0", "NaN", "1e30", "-1.0"] {
            assert!(check(&file(WINDOW, zoom)).is_err(), "zoom {zoom}");
        }
        for size in ["(x:NaN,y:760.0)", "(x:1100.0,y:1e30)", "(x:inf,y:1.0)"] {
            let w = WINDOW.replace("(x:1100.0,y:760.0)", size);
            assert!(check(&file(&w, "1.0")).is_err(), "size {size}");
        }
    }

    #[test]
    fn a_saved_window_position_is_detected() {
        // #4826: the position from the report (Linux Mint, window moved to the third monitor).
        let window = "(inner_position_pixels:Some((x:2048.0,y:1265.0)),outer_position_pixels:Some((x:2048.0,y:1228.0)),fullscreen:false,maximized:false,inner_size_points:Some((x:880.0,y:480.0)))";
        assert!(saved_position(&file(window, "1.0")));
        let no_position = "(inner_position_pixels:None,outer_position_pixels:None,fullscreen:false,maximized:false,inner_size_points:Some((x:880.0,y:480.0)))";
        assert!(!saved_position(&file(no_position, "1.0")));
        assert!(!saved_position(&format!("{{\"egui\":{:?}}}", "()")), "no window entry");
        for text in ["", "not ron {{{", "{\"window\":\"(((\"}"] {
            assert!(!saved_position(text), "{text:?}");
        }
        assert!(!super::has_saved_position(None));
        assert!(!super::has_saved_position(Some(&std::env::temp_dir().join("photocraft-missing-ui.ron"))));
    }

    #[test]
    fn unreadable_files_are_left_to_eframe() {
        // eframe ignores files and entries it can't parse, so they can't crash it.
        for text in ["", "not ron {{{", "{\"window\":\"(((\"}", "[1,2,3]"] {
            assert_eq!(check(text), Ok(()), "{text:?}");
        }
        let deep = format!("{{\"egui\":{:?}}}", "[".repeat(500) + &"]".repeat(500));
        let _ = check(&deep);
    }

    #[test]
    fn sanitize_removes_only_a_bad_file() {
        let dir = std::env::temp_dir().join(format!("photocraft-ui-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.ron");
        let bad = dir.join("bad.ron");
        std::fs::write(&good, file(WINDOW, "1.0")).unwrap();
        std::fs::write(&bad, file(WINDOW, "0.0")).unwrap();
        super::sanitize(Some(&good));
        super::sanitize(Some(&bad));
        super::sanitize(Some(&dir.join("missing.ron")));
        super::sanitize(None);
        assert!(good.exists() && !bad.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
