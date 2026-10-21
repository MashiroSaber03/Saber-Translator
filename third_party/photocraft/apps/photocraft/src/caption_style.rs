//! Linux: which desktop the custom title bar's caption buttons look like, and which buttons go on
//! which side (`photocraft_ui_egui::CaptionStyle`). GNOME gets libadwaita's buttons in the order of
//! its `button-layout` setting (read through the settings portal, and followed as it changes); KDE
//! gets Breeze's in KWin's order (`kwinrc`); other desktops get plain round buttons, or Adwaita's
//! or Breeze's when their GTK theme is one of those, in GTK's `gtk-decoration-layout` order.
//!
//! `PHOTOCRAFT_CAPTION_STYLE` (`windows`, `adwaita`, `breeze`, `generic`) and
//! `PHOTOCRAFT_CAPTION_LAYOUT` (a GTK layout such as `close,minimize:`) override what is found.

#[cfg(target_os = "linux")]
use photocraft_ui_egui::{CaptionFamily, CaptionIcon, CaptionIcons, CaptionStyle};

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Desktop {
    Gnome,
    Kde,
    Other,
}

/// The desktop, from the first of its variables that is set: `ubuntu:GNOME`, `Budgie:GNOME` and
/// the like count as GNOME.
#[cfg(target_os = "linux")]
fn desktop(var: &dyn Fn(&str) -> Option<String>) -> Desktop {
    for key in ["XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION"] {
        let Some(v) = var(key).map(|v| v.to_ascii_lowercase()).filter(|v| !v.trim().is_empty()) else { continue };
        return if v.contains("gnome") {
            Desktop::Gnome
        } else if v.contains("kde") || v.contains("plasma") {
            Desktop::Kde
        } else {
            Desktop::Other
        };
    }
    if var("KDE_FULL_SESSION").is_some() {
        Desktop::Kde
    } else if var("GNOME_DESKTOP_SESSION_ID").is_some() {
        Desktop::Gnome
    } else {
        Desktop::Other
    }
}

/// GNOME's apps are libadwaita's whatever the GTK theme, and KWin draws Breeze; elsewhere the GTK
/// theme says (Adwaita, adw-gtk3, Yaru, Breeze), and any other theme gets the plain buttons.
#[cfg(target_os = "linux")]
fn family(desktop: Desktop, gtk_theme: Option<&str>) -> CaptionFamily {
    match desktop {
        Desktop::Gnome => CaptionFamily::Adwaita,
        Desktop::Kde => CaptionFamily::Breeze,
        Desktop::Other => {
            let theme = gtk_theme.unwrap_or_default().trim().to_ascii_lowercase();
            if ["adwaita", "adw-gtk3", "yaru"].iter().any(|p| theme.starts_with(p)) {
                CaptionFamily::Adwaita
            } else if theme.starts_with("breeze") {
                CaptionFamily::Breeze
            } else {
                CaptionFamily::Generic
            }
        }
    }
}

/// The value of `key` in `[section]` of an INI-style file (GTK's `settings.ini`, KDE's `kwinrc`).
#[cfg(target_os = "linux")]
fn ini_value(text: &str, section: &str, key: &str) -> Option<String> {
    let mut inside = false;
    for line in text.lines().map(str::trim) {
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            inside = name == section;
        } else if inside
            && let Some((k, v)) = line.split_once('=')
            && k.trim() == key
        {
            return Some(v.trim().to_string());
        }
    }
    None
}

/// KWin's button order (`[org.kde.kdecoration2]` `ButtonsOnLeft` / `ButtonsOnRight`, one letter a
/// button: `I` minimize, `A` maximize, `X` close), as a GTK layout. Missing keys keep KWin's
/// defaults: the window menu on the left, minimize, maximize and close on the right.
#[cfg(target_os = "linux")]
fn kwin_layout(kwinrc: &str) -> String {
    let side = |key: &str, default: &str| {
        let letters = ini_value(kwinrc, "org.kde.kdecoration2", key).unwrap_or_else(|| default.to_string());
        letters
            .chars()
            .filter_map(|c| match c {
                'I' => Some("minimize"),
                'A' => Some("maximize"),
                'X' => Some("close"),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    format!("{}:{}", side("ButtonsOnLeft", "MS"), side("ButtonsOnRight", "HIAX"))
}

/// What the style is worked out from.
#[cfg(target_os = "linux")]
#[derive(Clone, Default)]
struct Found {
    gtk_theme: Option<String>,
    /// A GTK decoration layout.
    layout: Option<String>,
    icon_theme: Option<String>,
}

/// The icon theme the glyphs come from: `PHOTOCRAFT_CAPTION_ICONS` (`none` draws PhotoCraft's own),
/// the desktop's setting, or its default theme.
#[cfg(target_os = "linux")]
fn icon_theme_name(desktop: Desktop, found: &Found, var: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    match var("PHOTOCRAFT_CAPTION_ICONS") {
        Some(name) if name.trim().eq_ignore_ascii_case("none") => None,
        Some(name) => Some(name.trim().to_string()),
        None => Some(found.icon_theme.clone().unwrap_or_else(|| if desktop == Desktop::Kde { "breeze" } else { "Adwaita" }.into())),
    }
}

/// The style from what was found; `icons` loads a theme's glyphs.
#[cfg(target_os = "linux")]
fn resolve(desktop: Desktop, found: &Found, var: &dyn Fn(&str) -> Option<String>, icons: &dyn Fn(&str) -> Option<CaptionIcons>) -> CaptionStyle {
    let family = var("PHOTOCRAFT_CAPTION_STYLE").and_then(|s| CaptionFamily::parse(&s)).unwrap_or_else(|| family(desktop, found.gtk_theme.as_deref()));
    // GNOME's own default shows Close only; GTK's shows all three.
    let default = if desktop == Desktop::Gnome { "appmenu:close" } else { "menu:minimize,maximize,close" };
    let layout = var("PHOTOCRAFT_CAPTION_LAYOUT").or_else(|| found.layout.clone()).unwrap_or_else(|| default.to_string());
    let mut style = CaptionStyle::from_gtk_layout(family, &layout);
    style.icons = icon_theme_name(desktop, found, var).and_then(|name| icons(&name));
    style
}

#[cfg(target_os = "linux")]
fn read_text(path: &std::path::Path) -> Option<String> {
    crate::icon_theme::read_small(path).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// The themes and layout from the desktop's files: `kwinrc` and `kdeglobals` on KDE, GTK 3's
/// `settings.ini` elsewhere (where GNOME's portal answers, read later, replace them).
#[cfg(target_os = "linux")]
fn from_files(desktop: Desktop, var: &dyn Fn(&str) -> Option<String>) -> Found {
    let config = var("XDG_CONFIG_HOME")
        .filter(|d| std::path::Path::new(d).is_absolute())
        .map(std::path::PathBuf::from)
        .or_else(|| var("HOME").map(|h| std::path::Path::new(&h).join(".config")));
    let Some(config) = config else { return Found::default() };
    let gtk = read_text(&config.join("gtk-3.0").join("settings.ini")).unwrap_or_default();
    let gtk_theme = ini_value(&gtk, "Settings", "gtk-theme-name");
    if desktop == Desktop::Kde {
        let layout = Some(kwin_layout(&read_text(&config.join("kwinrc")).unwrap_or_default()));
        let icon_theme = ini_value(&read_text(&config.join("kdeglobals")).unwrap_or_default(), "Icons", "Theme");
        Found { gtk_theme, layout, icon_theme }
    } else {
        let layout = ini_value(&gtk, "Settings", "gtk-decoration-layout");
        Found { gtk_theme, layout, icon_theme: ini_value(&gtk, "Settings", "gtk-icon-theme-name") }
    }
}

/// The caption glyphs of icon theme `theme` (`window-*-symbolic.svg`), read now and rendered at
/// whatever size the bar asks for. `None` when the theme (with its parents) has none of them.
#[cfg(target_os = "linux")]
fn load_icons(theme: &str) -> Option<CaptionIcons> {
    use crate::icon_theme::{IconTheme, base_dirs, rasterize_mask, read_small};
    let found = IconTheme::load(theme, &base_dirs(&|k: &str| std::env::var(k).ok()))?;
    let svg = |name: &str| found.find_svg(name).and_then(|p| read_small(&p)).map(std::sync::Arc::<[u8]>::from);
    let files = [svg("window-minimize-symbolic"), svg("window-maximize-symbolic"), svg("window-restore-symbolic"), svg("window-close-symbolic")];
    if files.iter().all(Option::is_none) {
        return None;
    }
    let render = move |icon: CaptionIcon, px: u32| {
        let i = match icon {
            CaptionIcon::Minimize => 0,
            CaptionIcon::Maximize => 1,
            CaptionIcon::Restore => 2,
            CaptionIcon::Close => 3,
        };
        files.get(i)?.as_deref().and_then(|svg| rasterize_mask(svg, px))
    };
    Some(CaptionIcons { theme: theme.to_string(), render: std::sync::Arc::new(render) })
}

/// [`load_icons`] for the last theme asked for, kept, so a layout change doesn't reread it.
#[cfg(target_os = "linux")]
#[derive(Default)]
struct IconCache(std::sync::Mutex<Option<(String, Option<CaptionIcons>)>>);

#[cfg(target_os = "linux")]
impl IconCache {
    fn get(&self, theme: &str) -> Option<CaptionIcons> {
        let mut last = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((name, icons)) = last.as_ref()
            && name == theme
        {
            return icons.clone();
        }
        let icons = load_icons(theme);
        *last = Some((theme.to_string(), icons.clone()));
        icons
    }
}

#[cfg(target_os = "linux")]
const WM_NAMESPACE: &str = "org.gnome.desktop.wm.preferences";
#[cfg(target_os = "linux")]
const LAYOUT_KEY: &str = "button-layout";
#[cfg(target_os = "linux")]
const INTERFACE_NAMESPACE: &str = "org.gnome.desktop.interface";
#[cfg(target_os = "linux")]
const ICONS_KEY: &str = "icon-theme";

/// A string setting from the portal, unwrapping the variants some portals nest it in.
#[cfg(target_os = "linux")]
fn portal_string(value: zbus::zvariant::OwnedValue) -> Option<String> {
    let mut value: zbus::zvariant::Value<'_> = value.into();
    for _ in 0..4 {
        match value {
            zbus::zvariant::Value::Value(inner) => value = *inner,
            zbus::zvariant::Value::Str(s) => return Some(s.to_string()),
            _ => return None,
        }
    }
    None
}

/// The caption style now, then with the icon theme's glyphs and (GNOME) the portal's settings,
/// following the button layout and icon theme as they change. A worker does the file and D-Bus
/// work and wakes the UI only when the style changes; the first answer usually lands before the
/// first frame.
pub fn service() -> Option<photocraft_ui_egui::CaptionStyleFn> {
    #[cfg(target_os = "linux")]
    {
        use std::sync::{Arc, Mutex, OnceLock, PoisonError};
        let var = |k: &str| std::env::var(k).ok();
        let desktop = desktop(&var);
        let mut found = from_files(desktop, &var);
        // Icons need the theme read, which takes tens of milliseconds: the worker brings them.
        let current = Arc::new(Mutex::new(resolve(desktop, &found, &var, &|_: &str| None)));
        let wake: Arc<OnceLock<egui::Context>> = Arc::new(OnceLock::new());
        let (shared, waker) = (Arc::clone(&current), Arc::clone(&wake));
        let cache = IconCache::default();
        let publish = move |found: &Found| {
            let style = resolve(desktop, found, &|k: &str| std::env::var(k).ok(), &|t: &str| cache.get(t));
            let mut cur = shared.lock().unwrap_or_else(PoisonError::into_inner);
            if *cur != style {
                *cur = style;
                if let Some(ctx) = waker.get() {
                    ctx.request_repaint();
                }
            }
        };
        let (ready, wait) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("caption-style".into()).spawn(move || {
            let proxy = if desktop == Desktop::Gnome { crate::appearance::portal_proxy() } else { None };
            let Some(proxy) = proxy else {
                publish(&found);
                let _ = ready.send(());
                return;
            };
            // Subscribe before the first read, so a change during start-up isn't lost.
            let signals = proxy.receive_signal("SettingChanged").ok();
            let read = |namespace: &str, key: &str| {
                let value: Option<zbus::zvariant::OwnedValue> = proxy.call("Read", &(namespace, key)).ok();
                value.and_then(portal_string)
            };
            found.layout = read(WM_NAMESPACE, LAYOUT_KEY).or(found.layout.take());
            found.icon_theme = read(INTERFACE_NAMESPACE, ICONS_KEY).or(found.icon_theme.take());
            publish(&found);
            let _ = ready.send(());
            for message in signals.into_iter().flatten() {
                let Ok((namespace, key, value)) = message.body().deserialize::<(String, String, zbus::zvariant::OwnedValue)>() else { continue };
                match (namespace.as_str(), key.as_str()) {
                    (WM_NAMESPACE, LAYOUT_KEY) => found.layout = portal_string(value),
                    (INTERFACE_NAMESPACE, ICONS_KEY) => found.icon_theme = portal_string(value),
                    _ => continue,
                }
                publish(&found);
            }
        });
        // A fast desktop answers before the first frame; a slow one updates the bar later.
        if spawned.is_ok() {
            let _ = wait.recv_timeout(std::time::Duration::from_millis(150));
        }
        Some(Box::new(move |ctx| {
            let _ = wake.set(ctx.clone());
            current.lock().unwrap_or_else(PoisonError::into_inner).clone()
        }))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use photocraft_ui_egui::Caption::{Close, Maximize, Minimize};

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let pairs: Vec<(String, String)> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| pairs.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone())
    }

    #[test]
    fn the_desktop_comes_from_the_first_variable_set() {
        assert_eq!(desktop(&env(&[("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")])), Desktop::Gnome);
        assert_eq!(desktop(&env(&[("XDG_CURRENT_DESKTOP", "KDE")])), Desktop::Kde);
        assert_eq!(desktop(&env(&[("XDG_CURRENT_DESKTOP", "X-Cinnamon"), ("KDE_FULL_SESSION", "true")])), Desktop::Other);
        assert_eq!(desktop(&env(&[("XDG_CURRENT_DESKTOP", " "), ("DESKTOP_SESSION", "plasma")])), Desktop::Kde);
        assert_eq!(desktop(&env(&[("KDE_FULL_SESSION", "true")])), Desktop::Kde);
        assert_eq!(desktop(&env(&[("GNOME_DESKTOP_SESSION_ID", "this-is-deprecated")])), Desktop::Gnome);
        assert_eq!(desktop(&env(&[])), Desktop::Other);
    }

    #[test]
    fn the_family_follows_the_desktop_then_the_gtk_theme() {
        assert_eq!(family(Desktop::Gnome, Some("Arc")), CaptionFamily::Adwaita);
        assert_eq!(family(Desktop::Kde, None), CaptionFamily::Breeze);
        assert_eq!(family(Desktop::Other, Some("adw-gtk3-dark")), CaptionFamily::Adwaita);
        assert_eq!(family(Desktop::Other, Some("Yaru-blue")), CaptionFamily::Adwaita);
        assert_eq!(family(Desktop::Other, Some("Breeze-Dark")), CaptionFamily::Breeze);
        assert_eq!(family(Desktop::Other, Some("Mint-Y")), CaptionFamily::Generic);
        assert_eq!(family(Desktop::Other, None), CaptionFamily::Generic);
    }

    #[test]
    fn kwin_button_letters_become_a_gtk_layout() {
        assert_eq!(kwin_layout(""), ":minimize,maximize,close", "KWin's defaults");
        let rc = "[Desktops]\nButtonsOnLeft=X\n[org.kde.kdecoration2]\nButtonsOnLeft=XIA\nButtonsOnRight=\nlibrary=org.kde.breeze\n";
        assert_eq!(kwin_layout(rc), "close,minimize,maximize:");
        assert_eq!(ini_value(rc, "org.kde.kdecoration2", "library").as_deref(), Some("org.kde.breeze"));
        assert_eq!(ini_value(rc, "nope", "library"), None);
        // Garbage is just no buttons on that side, never a panic.
        assert_eq!(kwin_layout("[org.kde.kdecoration2]\nButtonsOnRight=ÿ€=X\n"), ":close");
    }

    fn no_icons(_: &str) -> Option<CaptionIcons> {
        None
    }

    #[test]
    fn the_icon_theme_comes_from_the_override_the_setting_or_the_desktop() {
        let found = Found { icon_theme: Some("Papirus".into()), ..Found::default() };
        assert_eq!(icon_theme_name(Desktop::Gnome, &found, &env(&[])).as_deref(), Some("Papirus"));
        assert_eq!(icon_theme_name(Desktop::Gnome, &Found::default(), &env(&[])).as_deref(), Some("Adwaita"));
        assert_eq!(icon_theme_name(Desktop::Kde, &Found::default(), &env(&[])).as_deref(), Some("breeze"));
        assert_eq!(icon_theme_name(Desktop::Kde, &found, &env(&[("PHOTOCRAFT_CAPTION_ICONS", " None ")])), None);
        assert_eq!(icon_theme_name(Desktop::Kde, &found, &env(&[("PHOTOCRAFT_CAPTION_ICONS", "Tela")])).as_deref(), Some("Tela"));
        // The style carries what the loader returns for that name.
        let s = resolve(Desktop::Gnome, &found, &env(&[]), &|t| {
            (t == "Papirus").then(|| CaptionIcons { theme: t.into(), render: std::sync::Arc::new(|_, _| None) })
        });
        assert_eq!(s.icons.map(|i| i.theme).as_deref(), Some("Papirus"));
    }

    #[test]
    fn layouts_and_overrides_resolve_to_a_style() {
        let none = env(&[]);
        let gnome = resolve(Desktop::Gnome, &Found::default(), &none, &no_icons);
        assert_eq!((gnome.family, gnome.left, gnome.right), (CaptionFamily::Adwaita, vec![], vec![Close]), "GNOME's default layout");
        let kde = resolve(Desktop::Kde, &Found { layout: Some(kwin_layout("")), ..Found::default() }, &none, &no_icons);
        assert_eq!((kde.family, kde.right), (CaptionFamily::Breeze, vec![Minimize, Maximize, Close]));
        let found = Found { gtk_theme: Some("Mint-Y".into()), layout: Some("close:".into()), icon_theme: None };
        let other = resolve(Desktop::Other, &found, &none, &no_icons);
        assert_eq!((other.family, other.left, other.right), (CaptionFamily::Generic, vec![Close], vec![]));
        let forced = env(&[("PHOTOCRAFT_CAPTION_STYLE", "Breeze"), ("PHOTOCRAFT_CAPTION_LAYOUT", "minimize,close:maximize")]);
        let s = resolve(Desktop::Gnome, &found, &forced, &no_icons);
        assert_eq!((s.family, s.left, s.right), (CaptionFamily::Breeze, vec![Minimize, Close], vec![Maximize]));
        // An unknown override name falls back to what was found.
        assert_eq!(resolve(Desktop::Kde, &found, &env(&[("PHOTOCRAFT_CAPTION_STYLE", "aqua")]), &no_icons).family, CaptionFamily::Breeze);
    }

    #[test]
    fn portal_strings_unwrap_nested_variants() {
        use zbus::zvariant::{OwnedValue, Value};
        let nested = Value::Value(Box::new(Value::Value(Box::new(Value::from("appmenu:close")))));
        assert_eq!(portal_string(OwnedValue::try_from(nested).unwrap()).as_deref(), Some("appmenu:close"));
        assert_eq!(portal_string(OwnedValue::try_from(Value::U32(7)).unwrap()), None);
    }

    #[test]
    fn settings_files_are_read_from_the_config_dir() {
        let dir = std::env::temp_dir().join(format!("photocraft-caption-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("gtk-3.0")).unwrap();
        std::fs::write(dir.join("gtk-3.0/settings.ini"), "[Settings]\ngtk-theme-name=Yaru\ngtk-decoration-layout=close,maximize:\n").unwrap();
        std::fs::write(dir.join("kwinrc"), "[org.kde.kdecoration2]\nButtonsOnRight=AX\n").unwrap();
        std::fs::write(dir.join("kdeglobals"), "[Icons]\nTheme=breeze-dark\n").unwrap();
        let var = env(&[("XDG_CONFIG_HOME", dir.to_str().unwrap())]);
        let other = from_files(Desktop::Other, &var);
        assert_eq!((other.gtk_theme.as_deref(), other.layout.as_deref()), (Some("Yaru"), Some("close,maximize:")));
        let kde = from_files(Desktop::Kde, &var);
        assert_eq!((kde.layout.as_deref(), kde.icon_theme.as_deref()), (Some(":maximize,close"), Some("breeze-dark")));
        // A relative XDG_CONFIG_HOME is ignored, as the XDG spec says; no HOME: nothing found.
        assert_eq!(from_files(Desktop::Other, &env(&[("XDG_CONFIG_HOME", "rel")])).layout, None);
        let _ = std::fs::remove_dir_all(dir);
    }
}
