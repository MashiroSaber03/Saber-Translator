//! Desktop colour capture starts only on an explicit UI request. No snapshots are written to
//! disk, transmitted or retained after a pick. Wayland delegates to its permission portal.
use photocraft_ui_egui::screen_picker::{Capture, MAX_PIXELS, Pending, ScreenImage, Service};

pub fn service(cc: &eframe::CreationContext<'_>, wayland: bool) -> Option<Service> {
    if !cfg!(any(target_os = "linux", target_os = "macos", target_os = "windows")) {
        return None;
    }
    let window = cc.winit_window()?.clone();
    Some(Box::new(move |ctx| {
        let (tx, rx) = std::sync::mpsc::channel();
        let fallback = tx.clone();
        let placements: Vec<Placement> = window
            .available_monitors()
            .enumerate()
            .map(|(index, m)| {
                let p = m.position();
                let size = m.size();
                Placement { index, x: p.x, y: p.y, width: size.width, height: size.height }
            })
            .collect();
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = cancelled.clone();
        let scale = ctx.native_pixels_per_point().unwrap_or(1.0);
        let wake = ctx.clone();
        if let Err(e) = std::thread::Builder::new().name("screen-color-picker".into()).spawn(move || {
            let result = capture(wayland, scale, &stop, &placements);
            let _ = tx.send(result);
            wake.request_repaint();
        }) {
            let _ = fallback.send(Err(format!("Could not start screen color picker: {e}")));
        }
        Pending { receiver: rx, cancelled }
    }))
}
#[derive(Clone, Copy)]
struct Placement {
    index: usize,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}
fn find_monitor(placements: &[Placement], x: i32, y: i32, w: u32, h: u32) -> Option<usize> {
    placements.iter().find(|p| p.x == x && p.y == y && p.width == w && p.height == h).map(|p| p.index)
}
fn count(width: usize, height: usize) -> Result<usize, String> {
    width.checked_mul(height).filter(|&n| n > 0 && n <= MAX_PIXELS).ok_or_else(|| "Screen capture exceeds the 64 MP memory budget".into())
}
#[cfg(target_os = "linux")]
const MAX_ICC_PROFILE_BYTES: usize = 1024 * 1024;
#[cfg(target_os = "linux")]
const ICC_PROFILE_PROPERTY_LONG_LENGTH: u32 = 256 * 1024;
#[cfg(target_os = "linux")]
fn icc_profile(bytes: Vec<u8>) -> Option<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > MAX_ICC_PROFILE_BYTES {
        return None;
    }
    let profile = match photocraft_cms::Profile::parse(&bytes) {
        Ok(profile) => profile,
        Err(_) => return None,
    };
    (profile.color_space == photocraft_cms::ColorSpace::Rgb).then_some(bytes)
}
#[cfg(target_os = "linux")]
fn icc_profile_atom<C: x11rb::connection::RequestConnection>(conn: &C, name: &str) -> Option<u32> {
    use x11rb::protocol::xproto::ConnectionExt as _;
    let cookie = match conn.intern_atom(true, name.as_bytes()) {
        Ok(cookie) => cookie,
        Err(_) => return None,
    };
    let atom = match cookie.reply() {
        Ok(reply) => reply.atom,
        Err(_) => return None,
    };
    (atom != 0).then_some(atom)
}
#[cfg(target_os = "linux")]
fn output_icc_profile<C: x11rb::connection::RequestConnection>(conn: &C, output: u32, atom: u32) -> Option<Vec<u8>> {
    use x11rb::protocol::{randr::ConnectionExt as _, xproto::AtomEnum};
    let cookie = match conn.randr_get_output_property(output, atom, AtomEnum::CARDINAL, 0, ICC_PROFILE_PROPERTY_LONG_LENGTH, false, false) {
        Ok(cookie) => cookie,
        Err(_) => return None,
    };
    let reply = match cookie.reply() {
        Ok(reply) => reply,
        Err(_) => return None,
    };
    if reply.type_ != u32::from(AtomEnum::CARDINAL) || reply.format != 8 || reply.bytes_after != 0 {
        return None;
    }
    icc_profile(reply.data)
}
#[cfg(target_os = "linux")]
fn root_icc_profile<C: x11rb::connection::RequestConnection>(conn: &C, root: u32, atom: u32) -> Option<Vec<u8>> {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
    let cookie = match conn.get_property(false, root, atom, AtomEnum::ANY, 0, ICC_PROFILE_PROPERTY_LONG_LENGTH) {
        Ok(cookie) => cookie,
        Err(_) => return None,
    };
    let reply = match cookie.reply() {
        Ok(reply) => reply,
        Err(_) => return None,
    };
    if reply.format != 8 || reply.bytes_after != 0 {
        return None;
    }
    icc_profile(reply.value)
}
#[cfg(target_os = "linux")]
fn capture(wayland: bool, scale: f32, stop: &std::sync::atomic::AtomicBool, placements: &[Placement]) -> Result<Capture, String> {
    if wayland {
        return portal(stop);
    }
    use x11rb::{
        connection::Connection,
        image::{Image, PixelLayout},
        protocol::{randr::ConnectionExt as _, xproto::VisualClass},
    };
    let (conn, index) = x11rb::connect(None).map_err(|e| e.to_string())?;
    let screen = conn.setup().roots.get(index).ok_or("X11 screen is missing")?;
    let (w, h) = (screen.width_in_pixels, screen.height_in_pixels);
    count(usize::from(w), usize::from(h))?;
    let mut monitors = conn.randr_get_monitors(screen.root, true).ok().and_then(|c| c.reply().ok()).map(|r| r.monitors).unwrap_or_default();
    monitors.sort_by_key(|m| !m.primary);
    if !monitors.iter().any(|m| m.primary)
        && let Some(first) = monitors.first_mut()
    {
        first.primary = true;
    }
    let regions: Vec<(i16, i16, u16, u16, bool, Vec<x11rb::protocol::randr::Output>)> = if monitors.is_empty() {
        vec![(0, 0, w, h, true, Vec::new())]
    } else {
        monitors.iter().map(|m| (m.x, m.y, m.width, m.height, m.primary, m.outputs.clone())).collect()
    };
    if regions.len() > 16 {
        return Err("Too many screens for the color picker".into());
    }
    let mut total = 0usize;
    for &(x, y, mw, mh, _, _) in &regions {
        if x < 0 || y < 0 || u32::from(x as u16) + u32::from(mw) > u32::from(w) || u32::from(y as u16) + u32::from(mh) > u32::from(h) {
            return Err("Invalid X11 display geometry".into());
        }
        total = total
            .checked_add(count(usize::from(mw), usize::from(mh))?)
            .filter(|&n| n <= MAX_PIXELS)
            .ok_or("Combined displays exceed the screen picker memory budget")?;
    }
    let (image, visual_id) = Image::get(&conn, screen.root, 0, 0, w, h).map_err(|e| format!("Screen capture failed: {e}"))?;
    let visual = screen.allowed_depths.iter().flat_map(|d| &d.visuals).find(|v| v.visual_id == visual_id).ok_or("Unsupported X11 screen visual")?;
    if visual.class != VisualClass::TRUE_COLOR {
        return Err("Screen picking requires an X11 TrueColor visual".into());
    }
    let layout = PixelLayout::from_visual_type(*visual).map_err(|e| e.to_string())?;
    let output_profile_atom = icc_profile_atom(&conn, "_ICC_PROFILE");
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let byte = |v: u16| ((u32::from(v) + 128) / 257) as u8;
    let mut images = Vec::new();
    for (x, y, mw, mh, primary, outputs) in regions {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(Capture::Color(None));
        }
        let profile = output_profile_atom.and_then(|atom| outputs.iter().find_map(|output| output_icc_profile(&conn, *output, atom))).or_else(|| {
            if !primary {
                return None;
            }
            icc_profile_atom(&conn, "_ICC_PROFILE").and_then(|atom| root_icc_profile(&conn, screen.root, atom))
        });
        let mut rgba = Vec::with_capacity(count(usize::from(mw), usize::from(mh))?.saturating_mul(4));
        for row in 0..mh {
            for col in 0..mw {
                let (r, g, b) = layout.decode(image.get_pixel((x as u16) + col, (y as u16) + row));
                rgba.extend_from_slice(&[byte(r), byte(g), byte(b), 255]);
            }
        }
        images.push(ScreenImage {
            monitor: find_monitor(placements, i32::from(x), i32::from(y), u32::from(mw), u32::from(mh)),
            position: egui::pos2(f32::from(x), f32::from(y)) / scale,
            size: egui::vec2(f32::from(mw), f32::from(mh)) / scale,
            width: usize::from(mw),
            height: usize::from(mh),
            rgba,
            profile,
        });
    }
    Ok(Capture::Images(images))
}
#[cfg(target_os = "linux")]
fn portal(stop: &std::sync::atomic::AtomicBool) -> Result<Capture, String> {
    use std::collections::HashMap;
    use zbus::{
        blocking::Proxy,
        zvariant::{OwnedObjectPath, OwnedValue, Value},
    };
    let conn = zbus::blocking::connection::Builder::session()
        .map_err(|e| e.to_string())?
        .method_timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let unique = conn.unique_name().ok_or("Portal connection has no unique name")?.as_str().trim_start_matches(':').replace('.', "_");
    let token = format!("photocraft_{}", std::process::id());
    let path = format!("/org/freedesktop/portal/desktop/request/{unique}/{token}");
    let request = Proxy::new(&conn, "org.freedesktop.portal.Desktop", path.as_str(), "org.freedesktop.portal.Request").map_err(|e| e.to_string())?;
    use futures_lite::StreamExt;
    let mut replies = futures_lite::future::block_on(request.inner().receive_signal("Response")).map_err(|e| e.to_string())?;
    let screenshot = Proxy::new(&conn, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.Screenshot")
        .map_err(|e| e.to_string())?;
    let options: HashMap<&str, Value<'_>> = HashMap::from([("handle_token", Value::from(token.as_str()))]);
    let _handle: OwnedObjectPath =
        screenshot.call("PickColor", &("", options)).map_err(|e| format!("The desktop portal does not support color picking: {e}"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let reply = loop {
        if stop.load(std::sync::atomic::Ordering::Relaxed) || std::time::Instant::now() >= deadline {
            let _: Result<(), _> = request.call("Close", &());
            return Ok(Capture::Color(None));
        }
        let next = futures_lite::future::block_on(futures_lite::future::race(async { Some(replies.next().await) }, async {
            async_io::Timer::after(std::time::Duration::from_millis(50)).await;
            None
        }));
        if let Some(message) = next {
            break message.ok_or("The desktop color picker closed without a result")?;
        }
    };
    let (code, values): (u32, HashMap<String, OwnedValue>) = reply.body().deserialize().map_err(|e| e.to_string())?;
    if code == 1 {
        return Ok(Capture::Color(None));
    }
    if code != 0 {
        return Err("The desktop color picker could not sample this screen".into());
    }
    let value = values.get("color").ok_or("Desktop picker returned no color")?;
    let (r, g, b): (f64, f64, f64) = value.try_clone().map_err(|e| e.to_string())?.try_into().map_err(|e: zbus::zvariant::Error| e.to_string())?;
    if [r, g, b].iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
        return Err("Desktop picker returned an invalid color".into());
    }
    Ok(Capture::Color(Some([r as f32, g as f32, b as f32])))
}
#[cfg(target_os = "windows")]
fn capture(_wayland: bool, _scale: f32, stop: &std::sync::atomic::AtomicBool, placements: &[Placement]) -> Result<Capture, String> {
    let monitors = xcap::Monitor::all().map_err(|e| e.to_string())?;
    if monitors.is_empty() || monitors.len() > 16 {
        return Err("No supported screen is available".into());
    }
    let mut budget = 0usize;
    for m in &monitors {
        budget = budget
            .checked_add(count(m.width().map_err(|e| e.to_string())? as usize, m.height().map_err(|e| e.to_string())? as usize)?)
            .filter(|&n| n <= MAX_PIXELS)
            .ok_or("Combined displays exceed the screen picker memory budget")?;
    }
    let mut total = 0usize;
    let mut images = Vec::new();
    for m in monitors {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(Capture::Color(None));
        }
        let (w, h) = (m.width().map_err(|e| e.to_string())?, m.height().map_err(|e| e.to_string())?);
        let scale = m.scale_factor().map_err(|e| e.to_string())?;
        if !scale.is_finite() || scale <= 0.0 || scale > 16.0 {
            return Err("Invalid display scale".into());
        }
        let captured = m.capture_image().map_err(|e| format!("Screen capture failed: {e}"))?;
        let (width, height) = (captured.width() as usize, captured.height() as usize);
        total = total.checked_add(count(width, height)?).filter(|&n| n <= MAX_PIXELS).ok_or("Combined displays exceed the screen picker memory budget")?;
        let (x, y) = (m.x().map_err(|e| e.to_string())?, m.y().map_err(|e| e.to_string())?);
        images.push(ScreenImage {
            monitor: find_monitor(placements, x, y, w, h),
            position: egui::pos2(x as f32, y as f32) / scale,
            size: egui::vec2(w as f32, h as f32) / scale,
            width,
            height,
            rgba: captured.into_raw(),
            profile: None,
        });
    }
    Ok(Capture::Images(images))
}
#[cfg(target_os = "macos")]
fn capture(_wayland: bool, _scale: f32, stop: &std::sync::atomic::AtomicBool, placements: &[Placement]) -> Result<Capture, String> {
    use core_graphics2::{display::get_active_display_list, image::CGImageAlphaInfo as Alpha, window};
    if !window::preflight_screen_capture_access() && !window::request_screen_capture_access() {
        return Err("Allow PhotoCraft in System Settings > Privacy & Security > Screen Recording, then reopen PhotoCraft".into());
    }
    let displays = get_active_display_list(17).ok_or("Could not enumerate displays")?;
    if displays.is_empty() || displays.len() > 16 {
        return Err("No supported screen is available".into());
    }
    let mut budget = 0usize;
    for display in &displays {
        let mode = display.copy_display_mode().ok_or("Display mode is unavailable")?;
        budget = budget
            .checked_add(count(mode.pixel_width(), mode.pixel_height())?)
            .filter(|&n| n <= MAX_PIXELS)
            .ok_or("Combined displays exceed the screen picker memory budget")?;
    }
    let mut images = Vec::new();
    let mut total = 0usize;
    for display in displays {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(Capture::Color(None));
        }
        // Capture a composed monitor at native resolution through safe bindings. Keeping the
        // CoreGraphics seam preserves PhotoCraft's existing macOS 11 deployment target.
        let bounds = display.bounds();
        let image = window::new_image(
            bounds,
            window::CGWindowListOption::OnScreenOnly,
            window::kCGNullWindowID,
            window::CGWindowImageOption::BestResolution | window::CGWindowImageOption::ShouldBeOpaque,
        )
        .ok_or("Screen capture failed; check Screen Recording permission")?;
        let (width, height) = (image.width(), image.height());
        total = total.checked_add(count(width, height)?).filter(|&n| n <= MAX_PIXELS).ok_or("Combined displays exceed the screen picker memory budget")?;
        let profile = image.color_space().and_then(|space| space.copy_icc_data());
        if profile.as_ref().is_some_and(|p| p.len() > 1024 * 1024) {
            return Err("Screen profile exceeds the memory budget".into());
        }
        let info = image.bitmap_info().bits();
        if image.bits_per_component() != 8 || image.bits_per_pixel() != 32 || info & (1 << 8) != 0 {
            return Err("Unsupported screen capture pixel format".into());
        }
        let (first, premultiplied, opaque) = match image.alpha_info() {
            Alpha::AlphaPremultipliedFirst => (true, true, false),
            Alpha::AlphaPremultipliedLast => (false, true, false),
            Alpha::AlphaFirst => (true, false, false),
            Alpha::AlphaLast => (false, false, false),
            Alpha::AlphaNoneSkipFirst => (true, false, true),
            Alpha::AlphaNoneSkipLast => (false, false, true),
            _ => return Err("Unsupported screen capture alpha layout".into()),
        };
        let little = match info & 0x7000 {
            0 => cfg!(target_endian = "little"),
            0x2000 => true,
            0x4000 => false,
            _ => return Err("Unsupported screen capture byte order".into()),
        };
        let order = match (first, little) {
            (true, true) => [2, 1, 0, 3],
            (true, false) => [1, 2, 3, 0],
            (false, true) => [3, 2, 1, 0],
            (false, false) => [0, 1, 2, 3],
        };
        let stride = image.bytes_per_row();
        checked_stride(width, height, stride)?;
        let bytes = image.data_provider().and_then(|p| p.copy_data()).ok_or("Screen image has no pixels")?;
        let rgba = unpack_rgba(bytes.bytes(), width, height, stride, order, premultiplied, opaque)?;
        let (w, h) = (bounds.size.width as f32, bounds.size.height as f32);
        let (x, y) = (bounds.origin.x as f32, bounds.origin.y as f32);
        if ![w, h, x, y].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
            return Err("Invalid display geometry".into());
        }
        let scale = width as f32 / w;
        images.push(ScreenImage {
            monitor: find_monitor(placements, (x * scale).round() as i32, (y * scale).round() as i32, width as u32, height as u32),
            position: egui::pos2(x, y),
            size: egui::vec2(w, h),
            width,
            height,
            rgba,
            profile: profile.map(|p| p.bytes().to_vec()),
        });
    }
    Ok(Capture::Images(images))
}
// CoreGraphics rows can be padded and their byte order/alpha vary. Kept platform independent
// so Linux CI also tests the macOS decoding path against synthetic pixels.
#[cfg(any(target_os = "macos", test))]
fn checked_stride(width: usize, height: usize, stride: usize) -> Result<usize, String> {
    count(width, height)?;
    let row = width.checked_mul(4).ok_or("Invalid screen row")?;
    if stride < row || stride > row.saturating_add(4096) {
        return Err("Invalid screen row stride".into());
    }
    stride.checked_mul(height).filter(|&n| n <= MAX_PIXELS.saturating_mul(4)).ok_or_else(|| "Screen buffer exceeds the memory budget".into())
}
#[cfg(any(target_os = "macos", test))]
fn unpack_rgba(bytes: &[u8], width: usize, height: usize, stride: usize, order: [usize; 4], premultiplied: bool, opaque: bool) -> Result<Vec<u8>, String> {
    let length = checked_stride(width, height, stride)?;
    if bytes.len() < length || order.iter().any(|&i| i > 3) {
        return Err("Invalid screen pixel buffer".into());
    }
    let mut out = Vec::with_capacity(count(width, height)?.saturating_mul(4));
    for row in bytes.chunks_exact(stride).take(height) {
        for p in row[..width * 4].as_chunks::<4>().0 {
            let a = if opaque { 255 } else { p[order[3]] };
            for &i in &order[..3] {
                let v = u32::from(p[i]);
                out.push(if premultiplied && a != 255 {
                    if a == 0 { 0 } else { ((v * 255 + u32::from(a) / 2) / u32::from(a)).min(255) as u8 }
                } else {
                    v as u8
                });
            }
            out.push(255);
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn screen_capture_decodes_padded_bgra_and_premultiplied_argb() {
        assert_eq!(
            unpack_rgba(&[10, 20, 30, 0, 99, 99, 99, 99, 40, 50, 60, 0, 99, 99, 99, 99], 1, 2, 8, [2, 1, 0, 3], false, true).unwrap(),
            [30, 20, 10, 255, 60, 50, 40, 255]
        );
        assert_eq!(unpack_rgba(&[128, 64, 32, 16], 1, 1, 4, [1, 2, 3, 0], true, false).unwrap(), [128, 64, 32, 255]);
        assert_eq!(unpack_rgba(&[0, 0, 0, 0], 1, 1, 4, [0, 1, 2, 3], true, false).unwrap(), [0, 0, 0, 255]);
    }
    #[test]
    fn screen_capture_rejects_unbounded_or_truncated_buffers() {
        assert!(count(usize::MAX, 2).is_err());
        assert!(checked_stride(1, 1, usize::MAX).is_err());
        assert!(checked_stride(1, MAX_PIXELS, 4096).is_err());
        assert!(unpack_rgba(&[0; 3], 1, 1, 4, [0, 1, 2, 3], false, true).is_err());
        assert!(unpack_rgba(&[0; 4], 1, 1, 4, [4, 1, 2, 3], false, true).is_err());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn x11_screen_capture_accepts_only_bounded_rgb_icc_profiles() {
        let rgb = photocraft_cms::Builtin::Srgb.profile().to_bytes();
        let bytes = rgb.as_ref().clone();
        assert_eq!(icc_profile(bytes.clone()), Some(bytes));

        assert!(icc_profile(Vec::new()).is_none());
        assert!(icc_profile(vec![0; MAX_ICC_PROFILE_BYTES + 1]).is_none());
        let cmyk = photocraft_cms::Builtin::CoatedCmyk.profile().to_bytes();
        assert!(icc_profile(cmyk.as_ref().clone()).is_none());
    }
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires a live X11 display; draws only a tiny synthetic fixture"]
    fn screen_capture_reads_a_known_x11_fixture_pixel() {
        use x11rb::{
            COPY_DEPTH_FROM_PARENT,
            connection::Connection,
            protocol::xproto::{ConfigureWindowAux, ConnectionExt, CreateWindowAux, StackMode, WindowClass},
        };
        let (conn, index) = x11rb::connect(None).unwrap();
        let screen = &conn.setup().roots[index];
        let window = conn.generate_id().unwrap();
        conn.create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            screen.root,
            200,
            200,
            8,
            8,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().override_redirect(1).background_pixel(0xff8040),
        )
        .unwrap()
        .check()
        .unwrap();
        conn.map_window(window).unwrap().check().unwrap();
        conn.configure_window(window, &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE)).unwrap().check().unwrap();
        conn.flush().unwrap();
        // X11 request completion precedes the compositor presenting the mapped fixture.
        std::thread::sleep(std::time::Duration::from_millis(150));
        let images = capture(false, 1.0, &std::sync::atomic::AtomicBool::new(false), &[]);
        conn.destroy_window(window).unwrap().check().unwrap();
        conn.flush().unwrap();
        let Capture::Images(images) = images.unwrap() else { panic!("not an image capture") };
        let first = images.iter().find(|m| m.position.x <= 200.0 && m.position.y <= 200.0).unwrap();
        let x = (200.0 - first.position.x) as usize;
        let y = (200.0 - first.position.y) as usize;
        let offset = (y * first.width + x) * 4;
        assert_eq!(&first.rgba[offset..offset + 4], &[255, 128, 64, 255]);
    }
    #[test]
    fn screen_capture_matches_negative_and_hidpi_monitor_geometry() {
        let m = [Placement { index: 2, x: -3840, y: 0, width: 3840, height: 2160 }];
        assert_eq!(find_monitor(&m, -3840, 0, 3840, 2160), Some(2));
        assert_eq!(find_monitor(&m, 0, 0, 1920, 1080), None);
    }
}
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn capture(_wayland: bool, _scale: f32, _stop: &std::sync::atomic::AtomicBool, _placements: &[Placement]) -> Result<Capture, String> {
    Err("Screen color picking is unavailable on this platform".into())
}
