//! Photocraft desktop app.
//!
//! Usage: `photocraft [--control <port>] [--control-token <64-hex> |
//! --control-token-file <path>] [--automation-read-root <dir>]
//! [--automation-write-root <dir>] [--safe-gpu] [--in-window-menus] [files…]`
//!
//! `--safe-gpu` starts with the CPU renderer (no GPU canvas; a software adapter for the window
//! where the platform has one) for this launch, e.g. after a graphics driver crash. A start that
//! crashes inside the driver also falls back by itself next time (see `gpu_startup`).
//!
//! `--in-window-menus` (or `PHOTOCRAFT_IN_WINDOW_MENUS=1`) keeps the menus inside the window on
//! macOS instead of the macOS menu bar (`mac_menu`).
//!
//! `--control <port>` (or `PHOTOCRAFT_CONTROL_PORT`) starts a localhost JSON-lines control server.
//! The first line must authenticate; subsequent request lines get reply lines.
//! `{"id":1,"ok":true,"result":…}`. See `photocraft_ui_egui::control` for the methods.

// Release builds on Windows are GUI-subsystem apps, so launching from the Start Menu or Explorer
// doesn't open a console window. (`--version` output then only shows when redirected.)
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod app_dirs;
mod app_icon;
mod appearance;
#[cfg(target_os = "macos")]
mod apple_events;
mod caps_lock;
mod caption_style;
mod control_server;
mod crash_guard;
mod cursor;
mod file_dialog_probe;
mod gpu_startup;
#[cfg(target_os = "linux")]
mod icon_theme;
#[cfg(target_os = "macos")]
mod mac_menu;
#[cfg(target_os = "macos")]
mod mac_window;
// Pure logic is tested on every platform; only Linux runs the check.
#[cfg(any(target_os = "linux", test))]
mod linux_libs;
mod logging;
mod monitor_profile;
mod photoshop_settings;
mod screen_color;
mod services;
// Windows gets pen pressure from winit (WM_POINTER); the web runner has its own listener.
#[cfg(any(target_os = "macos", target_os = "linux", test))]
mod tablet;
mod ui_state;
mod window_frame;

use photocraft_engine::Session;
use photocraft_ui_egui::PhotocraftApp;

/// Matches the `.desktop` file and hicolor icon name, so Wayland docks pick up the icon.
const APP_ID: &str = "ai.storyteller.photocraft";

/// Windows and Linux: no OS title bar; the app's top bar is the title bar, with its own caption
/// buttons and edge resizing (`photocraft_ui_egui::titlebar`), as Photoshop does on Windows. macOS
/// keeps its traffic lights over the integrated title strip.
const CUSTOM_TITLEBAR: bool = !cfg!(target_os = "macos");

/// Whether this start draws its own title bar: Windows and Linux do, unless Preferences ›
/// Interface › System Title Bar asks for the system's (#1271, #1316). Read from the saved
/// preferences before the window opens; a missing or unreadable file keeps the default.
fn custom_titlebar(prefs_file: Option<&std::path::Path>) -> bool {
    let prefs: photocraft_engine::prefs::Preferences =
        prefs_file.and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    CUSTOM_TITLEBAR && !prefs.interface.system_title_bar
}

/// The main window: 1440 × 900 (shrunk to fit the monitor, and maximized on the first frame
/// when it still doesn't fit, `work_area::fit_window`), centred on the main monitor. Without
/// `centered`, Windows cascades each new window from the top-left corner, so it opened at a
/// different offset every launch (#419). A later launch restores the saved position instead
/// (`main` turns `centered` off, #4826). Wayland compositors place windows themselves.
fn native_options(custom_titlebar: bool) -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_icon(app_icon::window_icon())
            .with_app_id(APP_ID)
            .with_title("PhotoCraft")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([760.0, 480.0])
            .with_drag_and_drop(true)
            .with_decorations(!custom_titlebar)
            .with_fullsize_content_view(true)
            .with_titlebar_shown(false)
            .with_title_shown(false),
        centered: true,
        // eframe saves native window geometry and egui panel/window sizes on exit.
        // Keep that state beside preferences, including config overrides and portable mode.
        persistence_path: services::config_dir().map(|dir| dir.join("ui.ron")),
        window_builder: Some(Box::new(restored_window)),
        ..Default::default()
    }
}

/// The window eframe restores from `ui.ron`, corrected before it is created: at least the
/// minimum size, and never full screen. PhotoCraft always starts in Standard Screen Mode, but
/// eframe also saves and restores full screen, and a window closed in a full-screen mode is
/// saved full screen and maximized at once: the overlap `view_cmds::screen_mode_commands` keeps
/// apart at run time. [OBSERVED 2026-10-12, Windows 11] Restored that way, View › Screen Mode ›
/// Full Screen did nothing until `F` had cycled through Standard.
fn restored_window(builder: egui::ViewportBuilder) -> egui::ViewportBuilder {
    at_least_min_size(builder).with_fullscreen(false)
}

/// Grow a restored window that is smaller than the minimum size, keeping its centre. On
/// Windows, eframe saves a minimized window's size (160 pt wide, 64 pt tall once eframe has
/// clamped it), and the minimum size isn't applied to a restored one: PhotoCraft then reopened
/// as a 160 × 64 strip on every launch.
fn at_least_min_size(mut builder: egui::ViewportBuilder) -> egui::ViewportBuilder {
    let (Some(size), Some(min)) = (builder.inner_size, builder.min_inner_size) else { return builder };
    let grown = size.max(min);
    if grown != size {
        builder.position = builder.position.map(|pos| pos - (grown - size) / 2.0);
        builder.inner_size = Some(grown);
    }
    builder
}

/// Parse a `--control` / `PHOTOCRAFT_CONTROL_PORT` value. An unparseable port is an error that
/// names the value (issue #701): silently running with no control server leaves a launcher or
/// agent unable to tell a typo from a successful grant.
fn parse_control_port(value: &str, source: &str) -> Result<u16, String> {
    value.trim().parse().map_err(|_| format!("{source}: `{value}` is not a valid port (expected a number from 0 to 65535)"))
}

/// Process exit status for the collected `--control` / `PHOTOCRAFT_CONTROL_PORT` errors: `None`
/// when there are none, else 2 (a command-line usage error), never 0.
fn control_args_exit_code(errors: &[String]) -> Option<i32> {
    if errors.is_empty() { None } else { Some(2) }
}

#[cfg(test)]
mod control_port_tests {
    use super::{control_args_exit_code, parse_control_port};

    #[test]
    fn control_port_errors_exit_non_zero() {
        assert_eq!(control_args_exit_code(&[]), None);
        let errors = vec![parse_control_port("nope", "--control").unwrap_err()];
        assert_eq!(control_args_exit_code(&errors), Some(2));
    }

    #[test]
    fn control_port_accepts_valid_numbers() {
        assert_eq!(parse_control_port("50494", "--control"), Ok(50494));
        assert_eq!(parse_control_port(" 8080 ", "--control"), Ok(8080));
        assert_eq!(parse_control_port("0", "PHOTOCRAFT_CONTROL_PORT"), Ok(0));
    }

    #[test]
    fn control_port_rejects_unparseable_values_naming_them() {
        // Out of range, a following flag eaten as the value, and an empty value all fail,
        // naming the offending value in the message.
        let err = parse_control_port("78787", "--control").unwrap_err();
        assert!(err.contains("78787"), "{err}");
        assert!(parse_control_port("--safe-gpu", "--control").is_err());
        assert!(parse_control_port("", "--control").is_err());
    }
}

fn main() -> eframe::Result {
    // First, so the panic hook and every start-up warning are recorded (`logging`).
    let logger = logging::install();
    crash_guard::install_hook();
    // The system font scan can take seconds (minutes on some Macs): run it off the UI thread.
    photocraft_ui_egui::type_tool::warm_fonts();
    let mut control_port: Option<u16> = None;
    let mut control_arg_errors: Vec<String> = Vec::new();
    if let Some(value) = std::env::var("PHOTOCRAFT_CONTROL_PORT").ok().filter(|v| !v.trim().is_empty()) {
        match parse_control_port(&value, "PHOTOCRAFT_CONTROL_PORT") {
            Ok(port) => control_port = Some(port),
            Err(error) => control_arg_errors.push(error),
        }
    }
    let mut control_token = None;
    let mut control_token_file = None;
    let mut automation_read_root = std::env::var_os("PHOTOCRAFT_AUTOMATION_READ_ROOT").map(std::path::PathBuf::from);
    let mut automation_write_root = std::env::var_os("PHOTOCRAFT_AUTOMATION_WRITE_ROOT").map(std::path::PathBuf::from);
    let mut files = Vec::new();
    // Arguments that are not valid Unicode (issue #1108): a file manager can hand over a Latin-1
    // file name on Linux, and `std::env::args()` would panic before any window opened. Paths are
    // reported in a notice once the app is up; a control token is a usage error like a bad port.
    let mut unreadable_paths: Vec<String> = Vec::new();
    let mut safe_gpu = false;
    let mut in_window_menus = std::env::var_os("PHOTOCRAFT_IN_WINDOW_MENUS").is_some_and(|v| !v.is_empty() && v != "0");
    let mut args = std::env::args_os().skip(1);
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("--control") => match args.next() {
                Some(value) => match parse_control_port(&value.to_string_lossy(), "--control") {
                    Ok(port) => control_port = Some(port),
                    Err(error) => control_arg_errors.push(error),
                },
                None => control_arg_errors.push("--control: missing port value (expected `--control <port>`)".to_string()),
            },
            Some("--control-token") => match args.next().map(std::ffi::OsString::into_string) {
                Some(Ok(token)) => control_token = Some(token),
                Some(Err(raw)) => control_arg_errors.push(format!("--control-token: value is not valid Unicode (`{}`)", raw.to_string_lossy())),
                None => control_token = None,
            },
            Some("--control-token-file") => control_token_file = args.next().map(std::path::PathBuf::from),
            Some("--automation-read-root") => automation_read_root = args.next().map(std::path::PathBuf::from),
            Some("--automation-write-root") => automation_write_root = args.next().map(std::path::PathBuf::from),
            Some("--safe-gpu") => safe_gpu = true,
            Some("--in-window-menus") => in_window_menus = true,
            Some("--version") => {
                println!("photocraft {}", photocraft_engine::build_info::long_version());
                return Ok(());
            }
            // Old macOS passes a process serial number when launched from Finder.
            Some(s) if s.starts_with("-psn_") => {}
            Some(s) => files.push(s.to_owned()),
            None => unreadable_paths.push(a.to_string_lossy().into_owned()),
        }
    }

    // A malformed control port must not silently drop the control server (issue #701): name the
    // bad value and fail the launch, like `photocraft-cli serve --port` does for the same typo.
    // Exit status 2 is the usual command-line usage error, so a launcher sees the failure.
    if let Some(code) = control_args_exit_code(&control_arg_errors) {
        for error in &control_arg_errors {
            eprintln!("photocraft: {error}");
        }
        std::process::exit(code);
    }

    // The log file lives under the settings directory; opened after the arguments, so `--version`
    // and usage errors leave no file behind. Records logged until now are written to it first.
    if let (Some(logger), Some(dir)) = (logger, services::config_dir()) {
        match logger.attach_dir(&dir.join("logs")) {
            Ok(path) => log::info!("PhotoCraft {}, log file {}", photocraft_engine::build_info::long_version(), path.display()),
            // Standard error only by now (`attach_dir` gave up on the file); unlike `eprintln!`, never panics.
            Err(e) => log::warn!("no log file: {e}"),
        }
    }

    // The display server to open the window on: Xwayland for a pen on Wayland, which gives this
    // app no pen input (#639).
    #[cfg(target_os = "linux")]
    let native_session = linux_libs::session_from_env(|name| std::env::var(name).ok());
    #[cfg(target_os = "linux")]
    let session = tablet::display_session();
    // Preferences › Performance › Linux display server = X11: Xwayland on a Wayland session too,
    // where native file drops work (winit 0.30 has none on Wayland, #386). Only when Xwayland can
    // run the window ($DISPLAY set, X11 libraries installed); otherwise the session's own.
    #[cfg(target_os = "linux")]
    let session = if session == linux_libs::DisplaySession::Wayland
        && gpu_startup::read_display_server(services::prefs_file().as_deref()) == photocraft_engine::prefs::LinuxDisplayServer::X11
        && std::env::var_os("DISPLAY").is_some_and(|d| !d.is_empty())
        && linux_libs::available(linux_libs::DisplaySession::X11)
    {
        eprintln!("photocraft: opening the window through Xwayland (Preferences › Performance › Linux display server)");
        linux_libs::DisplaySession::X11
    } else {
        session
    };

    // winit and wgpu dlopen the windowing and GPU libraries, and some of those crates panic when
    // one is missing (issue #201). Name the package to install and exit instead.
    #[cfg(target_os = "linux")]
    if let Err(message) = linux_libs::preflight(session) {
        eprint!("{message}");
        std::process::exit(1);
    }

    let control = if let Some(port) = control_port {
        let (supplied, token_file) = photocraft_automation::security::token_inputs(control_token, control_token_file);
        let token = match photocraft_automation::security::server_token(supplied.as_deref(), token_file.as_deref()) {
            Ok(token) => token,
            Err(e) => {
                eprintln!("photocraft: cannot configure control authentication: {e}");
                std::process::exit(2);
            }
        };
        if let Some(path) = token_file {
            eprintln!("photocraft: control token file: {}", path.display());
        } else if supplied.is_none() {
            eprintln!("photocraft: control token: {token}");
        } else {
            eprintln!("photocraft: using supplied control token");
        }
        let workspace = match photocraft_automation::AuthorizedWorkspace::new(automation_read_root.as_deref(), automation_write_root.as_deref()) {
            Ok(workspace) => workspace,
            Err(error) => {
                eprintln!("photocraft: cannot configure automation workspace: {error}");
                std::process::exit(2);
            }
        };
        Some((port, token, workspace))
    } else {
        None
    };
    // Finder / Dock / Open With deliver files as Apple events, not arguments; catch the one that
    // launched us as well as later ones. Lives until the event loop returns.
    #[cfg(target_os = "macos")]
    let apple_events = apple_events::AppleEvents::install();
    #[cfg(target_os = "macos")]
    let apple_events = &apple_events;

    // The macOS pen tablet monitor (installed once eframe created the app, see below); it lives
    // until the event loop returns.
    #[cfg(target_os = "macos")]
    let tablet_monitor = std::cell::OnceCell::new();
    #[cfg(target_os = "macos")]
    let tablet_monitor = &tablet_monitor;

    // Read the displays' ICC profiles while the window opens (colour-managed canvas; `None`
    // where the platform has no reader).
    #[cfg(target_os = "linux")]
    let monitor = monitor_profile::detect_async_for_session(native_session);
    #[cfg(not(target_os = "linux"))]
    let monitor = monitor_profile::detect_async();
    // Brush presets load in the background; the app attaches them when they arrive.
    let presets = services::presets_dir().map(photocraft_engine::preset_store::open_dir_async);
    let custom_titlebar = custom_titlebar(services::prefs_file().as_deref());
    let mut options = native_options(custom_titlebar);
    // winit picks Wayland whenever `WAYLAND_DISPLAY` is set; open on X11 when that was chosen.
    #[cfg(target_os = "linux")]
    if session == linux_libs::DisplaySession::X11 {
        use winit::platform::x11::EventLoopBuilderExtX11 as _;
        options.event_loop_builder = Some(Box::new(|builder| {
            builder.with_x11();
        }));
    }
    // eframe restores the saved window layout before our code runs; drop values that would crash it.
    ui_state::sanitize(options.persistence_path.as_deref());
    // Centre only the first window: eframe centres after restoring the saved position, so a
    // window closed on another monitor reopened in the middle of the main one (#4826).
    options.centered = !ui_state::has_saved_position(options.persistence_path.as_deref());
    // Crash-safe GPU startup (#4): pick the backend (a marker left by a start that died in the
    // driver moves to a safer one), and lock this start's marker until the first frames render.
    let t_sentinel = std::time::Instant::now();
    let os = gpu_startup::Os::current();
    let (pref, mode) = gpu_startup::read_rendering_prefs(services::prefs_file().as_deref());
    let (previous, sentinel) = match services::config_dir() {
        Some(dir) => gpu_startup::Sentinel::begin(&dir),
        None => (gpu_startup::Previous::Clean, None),
    };
    let env_backend = std::env::var("WGPU_BACKEND").ok();
    let plan = gpu_startup::plan_with_mode(pref, mode, previous.crashed(), env_backend.as_deref(), safe_gpu, os);
    if let Some(m) = previous.crashed() {
        log::warn!("the previous start didn't finish (GPU backend {}, adapter {:?}); {}", m.backend, m.adapter, plan.reason.as_deref().unwrap_or(""));
    }
    let sentinel: gpu_startup::SharedSentinel = std::sync::Arc::new(std::sync::Mutex::new(sentinel));
    if let Some(s) = sentinel.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_mut() {
        let backend = match &plan.env {
            Some(v) => format!("env:{v}"),
            None => plan.backend.name().to_string(),
        };
        let marker = gpu_startup::Marker { backend, version: photocraft_engine::build_info::long_version().to_string(), ..Default::default() };
        if let Err(e) = s.write(marker) {
            log::warn!("GPU startup marker: {e}");
        }
    }
    let gpu_note: std::sync::Arc<std::sync::Mutex<Option<String>>> = Default::default();
    // The adapter's real texture limits (egui asks for 8192 px), so big documents stay on the GPU.
    gpu_startup::configure(&mut options.wgpu_options.wgpu_setup, &plan, os, sentinel.clone(), gpu_note.clone());
    let sentinel_ms = t_sentinel.elapsed().as_secs_f64() * 1000.0;
    log::info!("GPU startup: {:?} ({sentinel_ms:.2} ms)", plan);
    let retry_cpu = !safe_gpu && plan.backend != photocraft_engine::prefs::GpuBackend::Cpu;
    let keep_marker = gpu_startup::keep_marker_after_error(&plan, os);
    let app_created = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let created_in_callback = app_created.clone();
    let started_sentinel = sentinel.clone();
    let result = eframe::run_native(
        "PhotoCraft",
        options,
        Box::new(move |cc| {
            created_in_callback.store(true, std::sync::atomic::Ordering::Relaxed);
            let automation = control.as_ref().map(|(_, _, workspace)| workspace.clone());
            let mut services = services::native(automation);
            services.preset_store = presets;
            #[cfg(target_os = "linux")]
            let display = tablet::DisplayKind::of(cc);
            #[cfg(target_os = "linux")]
            {
                services.is_wayland = display == Some(tablet::DisplayKind::Wayland);
                file_dialog_probe::start();
                if services.is_wayland {
                    let var = |name| std::env::var(name).ok();
                    services.xwayland_command =
                        services::xwayland_command(var("FLATPAK_ID").as_deref(), var("APPIMAGE").as_deref(), std::env::var_os("DISPLAY").is_some());
                }
            }
            let mut app = PhotocraftApp::new(Session::new(), services);
            app.integrated_titlebar = cfg!(target_os = "macos");
            app.custom_titlebar = custom_titlebar;
            // Only the title bar's free gap drags the window, never the menus (mac_window.rs).
            #[cfg(target_os = "macos")]
            mac_window::disable_native_title_drag();
            // Long commands and file opens run as background jobs with progress and Cancel (#210).
            app.background_jobs = std::env::var_os("PHOTOCRAFT_INLINE_JOBS").is_none();
            // Displays and their profiles (#569): wait briefly so the first frames already use
            // the right profile; a slower reading is applied when it arrives, and the shell reads
            // again when the displays may have changed.
            if let Some(rx) = monitor {
                #[cfg(target_os = "linux")]
                {
                    app.services.read_displays = Some(std::sync::Arc::new(move || monitor_profile::detect_async_for_session(native_session)));
                }
                #[cfg(not(target_os = "linux"))]
                {
                    app.services.read_displays = Some(std::sync::Arc::new(monitor_profile::detect_async));
                }
                match rx.recv_timeout(std::time::Duration::from_secs(2)) {
                    Ok(r) => photocraft_ui_egui::monitor_status::apply(&mut app, r),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => photocraft_ui_egui::monitor_status::pending(&mut app, rx),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        photocraft_ui_egui::monitor_status::apply(&mut app, Err("the display profile reader stopped without an answer".into()));
                    }
                }
            }
            // Preferences › Performance › Use Graphics Processor (and the GPU backend: `cpu`
            // composites on the CPU).
            let info = &mut app.perf.gpu_info;
            info.preference = pref.name().to_string();
            info.selected = if plan.env.is_some() { "env".into() } else { plan.backend.name().to_string() };
            info.fallback = match (plan.reason.clone(), gpu_note.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()) {
                (Some(reason), Some(note)) => Some(format!("{reason}. {note}")),
                (reason, note) => reason.or(note),
            };
            info.canvas = "cpu".into();
            if let Some(rs) = cc.wgpu_render_state.clone() {
                app.perf.gpu_info.set_adapter(&rs.adapter.get_info());
                let software_window = rs.adapter.get_info().device_type == eframe::wgpu::DeviceType::Cpu;
                if software_window && mode != photocraft_engine::prefs::RenderingMode::Cpu {
                    app.perf.gpu_info.fallback = Some("No compatible hardware graphics adapter; using software graphics.".into());
                }
                if !software_window
                    && plan.backend != photocraft_engine::prefs::GpuBackend::Cpu
                    && std::env::var_os("PHOTOCRAFT_CPU_CANVAS").is_none()
                    && app.session.prefs().performance.effective_rendering_mode() != photocraft_engine::prefs::RenderingMode::Cpu
                {
                    app.set_wgpu(rs);
                } else {
                    // The window still draws with wgpu: record its errors instead of panicking.
                    let _ = photocraft_ui_egui::gpu_canvas::DeviceHealth::watch(&rs.device);
                }
            }
            let fallback_reason = std::env::var("PHOTOCRAFT_GPU_STARTUP_FAILURE").ok().or_else(|| {
                (app.perf.gpu_info.canvas == "cpu" && mode != photocraft_engine::prefs::RenderingMode::Cpu)
                    .then(|| app.perf.gpu_info.fallback.clone())
                    .flatten()
            });
            if let Some(reason) = fallback_reason {
                photocraft_ui_egui::gpu_status::queue_fallback_notice(&mut app, &reason);
            }
            app.perf.span("gpuSentinel", sentinel_ms);
            // CPU rendering after a failed GPU start is temporary. Only the
            // recovery-dialog choice may persist it (#3483).
            let remember = gpu_startup::backend_to_remember(&plan);
            app.on_started(move |app| {
                if let Some(s) = started_sentinel.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
                    s.finish();
                }
                if let Some(b) = remember
                    && app.session.prefs().performance.gpu_backend != b
                    && let Err(e) = app.run("prefs.set", serde_json::json!({"path": "performance.gpuBackend", "value": b.name()}))
                {
                    log::warn!("couldn't remember GPU backend {}: {e}", b.name());
                }
            });
            if let Some((port, token, _)) = control {
                let rx = control_server::start(port, token, cc.egui_ctx.clone());
                app = app.with_control(rx);
            }
            #[cfg(target_os = "macos")]
            {
                app.services.os_events = Some(apple_events.connect(&cc.egui_ctx));
                // Quit through AppKit, not by closing the window (Touch Bar Macs, #1575, #1458).
                app.services.quit = Some(Box::new(mac_window::terminate_later));
                // The macOS menu bar, installed now so winit's default menu doesn't stay up.
                if !in_window_menus {
                    app.services.native_menu = mac_menu::install(&cc.egui_ctx, &app);
                }
            }
            #[cfg(not(target_os = "macos"))]
            let _ = in_window_menus;
            // Where file drags and drops are (winit 0.30 doesn't say).
            app.services.cursor_pos = cursor::service(cc);
            // The custom title bar's undecorated window must stay borderless, or its caption and
            // border offset the content and the pointer on some Windows systems (#2246).
            app.services.window_frame = window_frame::service(cc, custom_titlebar);
            // Linux: the desktop's caption buttons (GNOME's, KDE's) on the sides its layout puts them.
            if custom_titlebar {
                app.services.caption_style = caption_style::service();
            }
            // Caps Lock state (X11/Windows/macOS; `None` on native Wayland): the canvas shows the
            // precise crosshair for painting tools while it is toggled on (#1758).
            app.services.caps_lock = caps_lock::service(cc);
            app.services.screen_pick = screen_color::service(cc, app.services.is_wayland);
            // Tablet pressure/tilt/eraser (winit drops them): the macOS monitor and the X11 reader
            // write into the stylus feed. The monitor goes in here, not before the event loop:
            // AppKit's shared application only exists once winit created it (#759).
            #[cfg(target_os = "macos")]
            if let Some(monitor) = tablet::install_macos(&app.stylus.feed) {
                let _ = tablet_monitor.set(monitor);
            }
            #[cfg(target_os = "linux")]
            tablet::spawn_x11(&app.stylus.feed, display);
            // Paths on the command line (Linux/Windows file associations, `photocraft a.psd`).
            app.open_paths(&files);
            if !unreadable_paths.is_empty() {
                use photocraft_ui_egui::i18n;
                let line = i18n::t("{path}: the path is not valid Unicode; rename the file and open it again.");
                let lines = unreadable_paths.iter().map(|p| i18n::fmt(line, &[("path", p.as_str())])).collect();
                photocraft_ui_egui::notices::post(&mut app, i18n::t("Could not open"), lines, true, None);
            }
            // Portable marker found but its data folder isn't writable (#228): say where settings went.
            if let Some(w) = &app_dirs::current().warning {
                photocraft_ui_egui::notices::post(&mut app, "Portable mode is off", vec![w.clone()], false, None);
            }
            Ok(Box::new(app))
        }),
    );
    // Closed or failed outside graphics initialization: not a driver crash. A renderer error
    // keeps the marker, so the next start tries a safer backend, unless there is no safer one:
    // a failed CPU start would otherwise pin every later start to CPU.
    let renderer_error = gpu_startup::keep_marker_after_run(&result);
    if !(renderer_error && keep_marker)
        && let Some(mut s) = sentinel.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    {
        if result.is_err()
            && !renderer_error
            && let Some(marker) = previous.crashed()
        {
            // This failure supplies no new graphics-crash evidence: retain the previous
            // marker, rather than recording this attempt's fallback backend.
            if let Err(error) = s.write(marker.clone()) {
                log::warn!("couldn't restore previous GPU startup marker: {error}");
            }
        } else {
            s.finish();
        }
    }
    // Retry in a fresh process: winit event loops cannot be recreated reliably in-process.
    // Only renderer initialization failures qualify; never restart after editing has begun.
    if retry_cpu && !app_created.load(std::sync::atomic::Ordering::Relaxed) && matches!(&result, Err(eframe::Error::Wgpu(_))) {
        let reason = result.as_ref().err().map(ToString::to_string).unwrap_or_default();
        if let Some(s) = sentinel.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            s.finish();
        }
        if let Ok(exe) = std::env::current_exe() {
            let launched = std::process::Command::new(exe)
                .args(std::env::args_os().skip(1))
                .arg("--safe-gpu")
                .env_remove("WGPU_BACKEND")
                .env("PHOTOCRAFT_GPU_STARTUP_FAILURE", &reason)
                .spawn();
            match launched {
                Ok(_) => return Ok(()),
                Err(error) => log::error!("could not start CPU compatibility mode: {error}"),
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn window_and_panel_geometry_survive_a_restart() {
        let options = super::native_options(super::CUSTOM_TITLEBAR);
        assert!(options.persist_window);
        assert_eq!(options.persistence_path, super::services::config_dir().map(|dir| dir.join("ui.ron")));

        #[derive(Default)]
        struct Storage(std::collections::BTreeMap<String, String>);
        impl eframe::Storage for Storage {
            fn get_string(&self, key: &str) -> Option<String> {
                self.0.get(key).cloned()
            }
            fn set_string(&mut self, key: &str, value: String) {
                self.0.insert(key.into(), value);
            }
            fn remove_string(&mut self, key: &str) {
                self.0.remove(key);
            }
            fn flush(&mut self) {}
        }
        let ctx = egui::Context::default();
        let input = || egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0))), ..Default::default() };
        let mut output = ctx.run_ui(input(), |ui| {
            egui::Panel::right("dock").exact_size(410.0).show(ui, |ui| ui.set_min_width(ui.available_width()));
        });
        output.textures_delta.clear();
        let mut storage = Storage::default();
        ctx.memory(|memory| eframe::set_value(&mut storage, "egui", memory));
        let restored = egui::Context::default();
        restored.memory_mut(|memory| *memory = eframe::get_value(&storage, "egui").unwrap());
        let mut output = restored.run_ui(input(), |ui| {
            egui::Panel::right("dock").default_size(290.0).show(ui, |ui| ui.set_min_width(ui.available_width()));
        });
        output.textures_delta.clear();
        let panel = egui::containers::panel::PanelState::load(&restored, egui::Id::new("dock")).unwrap();
        assert_eq!(panel.size().x, 410.0);
    }

    #[test]
    fn the_window_opens_centred_at_its_default_size() {
        let o = super::native_options(super::CUSTOM_TITLEBAR);
        assert!(o.centered, "#419: centred, not cascaded from the top-left corner");
        assert_eq!(o.viewport.inner_size, Some(egui::vec2(1440.0, 900.0)));
        // eframe shrinks the start size to the monitor, so the centred position is on-screen.
        assert_ne!(o.viewport.clamp_size_to_monitor_size, Some(false));
    }

    #[test]
    fn a_window_saved_while_minimized_reopens_at_the_minimum_size() {
        let hook = super::native_options(super::CUSTOM_TITLEBAR).window_builder.expect("a window builder hook");
        // Saved minimized on Windows, then centred on a 1920 × 1080 monitor by eframe.
        let saved = super::native_options(super::CUSTOM_TITLEBAR).viewport.with_inner_size([160.0, 64.0]).with_position([880.0, 508.0]);
        let opened = hook(saved);
        assert_eq!(opened.inner_size, Some(egui::vec2(760.0, 480.0)));
        assert_eq!(opened.position, Some(egui::pos2(580.0, 300.0)), "still centred");
        // Too narrow only: only the width grows.
        let opened = super::at_least_min_size(super::native_options(false).viewport.with_inner_size([500.0, 900.0]));
        assert_eq!(opened.inner_size, Some(egui::vec2(760.0, 900.0)));
        // A window at or above the minimum, or with no size, is left alone.
        for size in [[760.0, 480.0], [1100.0, 760.0]] {
            let builder = super::native_options(false).viewport.with_inner_size(size).with_position([10.0, 20.0]);
            let opened = super::at_least_min_size(builder);
            assert_eq!((opened.inner_size, opened.position), (Some(egui::Vec2::from(size)), Some(egui::pos2(10.0, 20.0))));
        }
        let mut builder = super::native_options(false).viewport;
        builder.inner_size = None;
        assert_eq!(super::at_least_min_size(builder).inner_size, None);
    }

    #[test]
    fn a_window_saved_in_full_screen_reopens_maximized_not_full_screen() {
        let hook = super::native_options(super::CUSTOM_TITLEBAR).window_builder.expect("a window builder hook");
        // What eframe restores from a ui.ron saved in a full-screen mode on Windows.
        let saved = super::native_options(super::CUSTOM_TITLEBAR)
            .viewport
            .with_inner_size([2560.0, 1392.0])
            .with_position([0.0, 0.0])
            .with_fullscreen(true)
            .with_maximized(true);
        let opened = hook(saved);
        assert_eq!(opened.fullscreen, Some(false), "Standard Screen Mode is where PhotoCraft starts");
        assert_eq!(opened.maximized, Some(true), "the rest of the saved geometry is kept");
        assert_eq!(opened.inner_size, Some(egui::vec2(2560.0, 1392.0)));
    }

    #[test]
    fn the_system_title_bar_preference_keeps_the_window_decorations() {
        // #1271, #1316: Preferences › Interface › System Title Bar gives the window back its
        // system decorations on Windows and Linux; macOS always has them.
        let dir = std::env::temp_dir().join(format!("pc-titlebar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("prefs.json");
        assert_eq!(super::custom_titlebar(None), super::CUSTOM_TITLEBAR, "no preferences file: the default");
        std::fs::write(&file, "not json").unwrap();
        assert_eq!(super::custom_titlebar(Some(&file)), super::CUSTOM_TITLEBAR, "an unreadable file: the default");
        std::fs::write(&file, r#"{"interface":{"systemTitleBar":true}}"#).unwrap();
        assert!(!super::custom_titlebar(Some(&file)));
        assert_eq!(super::native_options(false).viewport.decorations, Some(true));
        std::fs::write(&file, r#"{"interface":{"systemTitleBar":false}}"#).unwrap();
        assert_eq!(super::custom_titlebar(Some(&file)), super::CUSTOM_TITLEBAR);
        if super::CUSTOM_TITLEBAR {
            assert_eq!(super::native_options(true).viewport.decorations, Some(false));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// X11 startup notification and docks match a window to its launcher by `StartupWMClass`,
    /// so it has to be the class the window carries: the app id, on Wayland and X11 alike.
    #[test]
    fn the_desktop_entry_names_the_window_class() {
        let entry = include_str!("../../../packaging/linux/ai.storyteller.photocraft.desktop");
        let class = entry.lines().find_map(|line| line.strip_prefix("StartupWMClass="));
        assert_eq!(class, Some(super::APP_ID));
    }
}
