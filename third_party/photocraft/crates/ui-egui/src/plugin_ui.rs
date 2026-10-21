//! Filter › Plug-ins: one menu item per installed WebAssembly filter plug-in (`plugin.*` engine
//! commands), with a generated dialog and live preview for plug-ins that take parameters.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::menus::MenuItem;

/// Menu ids of installed plug-ins: `plugin.filter.<plug-in id>`.
const PREFIX: &str = "plugin.filter.";

/// Inserts the installed plug-ins into Filter › Plug-ins, above "Install Plug-in…".
pub fn insert_menu_items(app: &PhotocraftApp, items: &mut Vec<MenuItem>) {
    let plugins = photocraft_plugins::registry::list();
    let Some(at) = items.iter().position(|i| i.id == "plugin.install") else { return };
    // `MenuItem::path` is the untranslated English path from the command's engine spec: the menu
    // renderer groups rows by it and translates only at display time. Translating it here left the
    // plug-in rows under a path no menu node matched, so under any non-English language they were
    // never drawn (#2540). Borrow the path of "Install Plug-in…", the item we sit above.
    let Some(path) = items.get(at).map(|i| i.path.clone()) else { return };
    let enabled = app.session.is_enabled("plugin.run");
    let mut extra: Vec<MenuItem> = plugins
        .iter()
        .map(|p| {
            let m = p.manifest();
            let dots = if m.params.is_empty() || m.name.ends_with('…') { "" } else { "…" };
            MenuItem {
                id: format!("{PREFIX}{}", m.id),
                label: format!("{}{dots}", m.name),
                path: path.clone(),
                shortcut: None,
                enabled,
                checked: None,
                color: None,
            }
        })
        .collect();
    if !extra.is_empty() {
        extra.push(MenuItem { id: "---".into(), label: "---".into(), path: path.clone(), shortcut: None, enabled: false, checked: None, color: None });
    }
    for (k, it) in extra.into_iter().enumerate() {
        items.insert(at + k, it);
    }
}

/// Enabled state of a plug-in menu item (`None` for other ids).
pub fn is_enabled(app: &PhotocraftApp, id: &str) -> Option<bool> {
    let pid = id.strip_prefix(PREFIX)?;
    Some(photocraft_plugins::registry::get(pid).is_some() && app.session.is_enabled("plugin.run"))
}

/// Runs a plug-in menu item: straight away when it has no parameters (or params were given),
/// else through its dialog. "Install Plug-in…" without params opens its path dialog.
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    let given = params.as_object().is_some_and(|o| !o.is_empty());
    if id == "plugin.install" && !given {
        return Some(Ok(json!({"dialog": crate::filter_dialog::open(app, "plugin.install")})));
    }
    let pid = id.strip_prefix(PREFIX)?;
    let Some(plugin) = photocraft_plugins::registry::get(pid) else { return Some(Err(format!("no plug-in {pid:?} is installed"))) };
    let m = plugin.manifest();
    if m.params.is_empty() || given {
        let mut p = params.as_object().cloned().unwrap_or_default();
        p.insert("id".into(), json!(pid));
        return Some(app.run("plugin.run", Value::Object(p)));
    }
    let mut fixed = Map::new();
    fixed.insert("id".into(), json!(pid));
    let d = crate::filter_dialog::open_with_spec(app, "plugin.run", &m.name, &m.params_notation(), fixed);
    Some(Ok(json!({"dialog": d})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_plugins_appear_under_filter_plugins_and_run() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../plugins/tests/fixtures/invert.wasm");
        app.run("plugin.install", json!({"path": path})).unwrap();
        let items = crate::menus::menu_items(&app);
        let pi: Vec<&MenuItem> = items.iter().filter(|i| i.path == ["Filter", "Plug-ins"]).collect();
        let me = pi.iter().find(|i| i.id == "plugin.filter.org.photocraft.example.invert").expect("plug-in listed");
        assert_eq!(me.label, "Invert (WebAssembly)");
        assert!(!me.enabled, "no document yet");
        assert!(pi.iter().any(|i| i.id == "plugin.install"), "Install Plug-in… is in the same submenu");
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        assert!(crate::menus::is_enabled(&app, "plugin.filter.org.photocraft.example.invert"));
        let px = |app: &PhotocraftApp| app.session.active().unwrap().doc.layers[0].surface().unwrap().pixel(1, 1);
        let before = px(&app);
        crate::menus::invoke(&mut app, &ctx, "plugin.filter.org.photocraft.example.invert", json!({})).unwrap();
        assert!((px(&app)[0] - (1.0 - before[0])).abs() < 1e-6);
        assert!(crate::menus::invoke(&mut app, &ctx, "plugin.filter.not.installed", json!({})).is_err());
        // Install Plug-in… without params opens its dialog instead of failing.
        let r = crate::menus::invoke(&mut app, &ctx, "plugin.install", json!({})).unwrap();
        assert!(r.get("dialog").is_some());
    }

    /// Plug-in rows carry the untranslated English menu path like every other item, so they still
    /// land under Filter › Plug-ins when the UI language translates those menu names (#2540).
    #[test]
    fn plugin_menu_path_stays_untranslated_in_other_languages() {
        let ja = crate::i18n::Lang::from_code("ja").expect("registered language");
        // The names on the path must really differ from the English, or the test proves nothing.
        assert_ne!(crate::i18n::tr(ja, "Filter"), "Filter");
        assert_ne!(crate::i18n::tr(ja, "Plug-ins"), "Plug-ins");
        let _lang = crate::i18n::language_scope(ja);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../plugins/tests/fixtures/invert.wasm");
        app.run("plugin.install", json!({"path": path})).unwrap();
        let items = crate::menus::menu_items(&app);
        let me = items.iter().find(|i| i.id == "plugin.filter.org.photocraft.example.invert").expect("plug-in listed");
        assert_eq!(me.path, ["Filter", "Plug-ins"], "plug-in row sits where the menu renderer looks for it");
        let install = items.iter().find(|i| i.id == "plugin.install").expect("Install Plug-in… listed");
        assert_eq!(me.path, install.path, "same submenu as Install Plug-in…");
    }
}
