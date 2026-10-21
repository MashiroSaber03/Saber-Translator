//! The app icon, everywhere Windows, macOS and Linux show one (#248).
//!
//! - **Executable:** `build.rs` embeds `assets/app-icon/photocraft.ico` (16–256 px) as icon
//!   resource 1, which Explorer, Alt+Tab and an unpinned taskbar button fall back to.
//! - **Window:** [`window_icon`] sets the title-bar and taskbar icon at runtime (`WM_SETICON` on
//!   Windows, `_NET_WM_ICON` on X11, the Dock when running unbundled on macOS).
//! - **Start Menu shortcut (MSI):** the taskbar matches a running window to the shortcut that
//!   launches its executable and shows that shortcut's icon. Windows Installer caches an advertised
//!   shortcut's icon under its `<Icon Id>`, so that id must carry the target's `.exe` extension
//!   (ICE50); an extensionless id showed a blank page on the taskbar. The tests below check it.
//!
//! The process keeps Windows' implicit AppUserModelID (derived from the executable path), which
//! is what makes that matching work for both the MSI shortcut and the portable exe. An explicit
//! ID would need a Win32 call (`unsafe`, which the workspace forbids) and would then also have to
//! be stamped on the shortcut, or the taskbar would stop matching the two.

/// Window, taskbar and (when running unbundled) Dock icon. macOS gets the padded 1024 px render
/// on Apple's icon grid; elsewhere the tighter 256 px hicolor render reads better at small sizes.
pub fn window_icon() -> egui::IconData {
    match eframe::icon_data::from_png_bytes(PNG) {
        Ok(icon) => icon,
        Err(e) => {
            log::warn!("app icon: {e}");
            egui::IconData::default()
        }
    }
}

#[cfg(target_os = "macos")]
const PNG: &[u8] = include_bytes!("../../../assets/app-icon/photocraft-1024.png");
#[cfg(not(target_os = "macos"))]
const PNG: &[u8] = include_bytes!("../../../assets/app-icon/hicolor/256x256/apps/ai.storyteller.photocraft.png");

#[cfg(test)]
mod tests {
    use super::*;

    const ICO: &[u8] = include_bytes!("../../../assets/app-icon/photocraft.ico");
    const WXS: &str = include_str!("../../../packaging/windows/photocraft.wxs");

    fn u16_at(b: &[u8], i: usize) -> usize {
        u16::from_le_bytes([b[i], b[i + 1]]) as usize
    }
    fn u32_at(b: &[u8], i: usize) -> usize {
        u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
    }

    #[test]
    fn window_icon_decodes() {
        let icon = window_icon();
        assert!(icon.width >= 256 && icon.width == icon.height, "{}x{}", icon.width, icon.height);
        assert_eq!(icon.rgba.len(), icon.width as usize * icon.height as usize * 4);
        assert!(icon.rgba.chunks(4).any(|p| p[3] > 0), "the icon isn't blank");
    }

    /// The `.ico` build.rs embeds: every size Windows asks for, each a valid PNG or BMP image.
    #[test]
    fn exe_icon_has_every_shell_size() {
        assert_eq!((u16_at(ICO, 0), u16_at(ICO, 2)), (0, 1), "ICONDIR header, type 1 (icon)");
        let n = u16_at(ICO, 4);
        let mut sizes = Vec::new();
        for i in 0..n {
            let e = 6 + i * 16;
            let w = if ICO[e] == 0 { 256 } else { ICO[e] as usize };
            let h = if ICO[e + 1] == 0 { 256 } else { ICO[e + 1] as usize };
            assert_eq!(w, h);
            let (len, off) = (u32_at(ICO, e + 8), u32_at(ICO, e + 12));
            let data = ICO.get(off..off + len).expect("image data inside the file");
            let png = data.starts_with(b"\x89PNG\r\n\x1a\n");
            let bmp = data.len() >= 40 && u32_at(data, 0) == 40;
            assert!(png || bmp, "{w}px entry is neither PNG nor BMP");
            if png {
                // IHDR width/height match the directory entry.
                let pw = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
                assert_eq!(pw, w, "{w}px entry holds a {pw}px PNG");
            }
            sizes.push(w);
        }
        for want in [16, 24, 32, 48, 64, 256] {
            assert!(sizes.contains(&want), "photocraft.ico lacks {want}px (has {sizes:?})");
        }
    }

    /// The value of `attr="…"` inside `element`.
    fn attr<'a>(element: &'a str, attr: &str) -> Option<&'a str> {
        let key = format!(" {attr}=\"");
        let start = element.find(&key)? + key.len();
        element[start..].split('"').next()
    }

    /// Each `<tag …>` element's text up to its closing `>`.
    fn elements<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
        let open = format!("<{tag} ");
        xml.match_indices(&open).map(|(i, _)| &xml[i..i + xml[i..].find('>').unwrap()]).collect()
    }

    /// ICE50 (#248): advertised shortcut icons are cached under their `<Icon Id>`; without the
    /// target's extension the shell shows a blank page on the Start Menu and the taskbar.
    #[test]
    fn msi_shortcut_icons_keep_the_exe_extension() {
        let icons: Vec<&str> = elements(WXS, "Icon").iter().filter_map(|e| attr(e, "Id")).collect();
        assert!(!icons.is_empty());
        for id in &icons {
            assert!(id.ends_with(".exe") || id.ends_with(".ico"), "<Icon Id=\"{id}\"> needs an .exe or .ico extension");
        }
        let shortcuts = elements(WXS, "Shortcut");
        assert!(!shortcuts.is_empty());
        for s in shortcuts {
            let icon = attr(s, "Icon").expect("shortcuts name their icon");
            assert!(icons.contains(&icon), "shortcut icon {icon} isn't an <Icon>");
            if attr(s, "Advertise") == Some("yes") {
                // Every shortcut in the MSI targets photocraft.exe.
                assert!(icon.ends_with(".exe"), "advertised shortcut icon {icon} must end in .exe like its target (ICE50)");
            }
        }
        let arp = WXS.split("Id=\"ARPPRODUCTICON\" Value=\"").nth(1).and_then(|r| r.split('"').next()).expect("ARPPRODUCTICON");
        assert!(icons.contains(&arp), "ARPPRODUCTICON {arp} isn't an <Icon>");
        // The Start Menu shortcut must not set its own AppUserModelID: the process keeps the
        // implicit one, and a mismatch would split the taskbar button from the shortcut.
        assert!(!WXS.contains("System.AppUserModel.ID"));
    }
}
