//! The macOS menu bar, through muda (a standalone native menu crate, not Tauri: no webview, just
//! AppKit menus behind a safe API, no GTK on macOS). The menus themselves, their Mac layout and how their keys reach PhotoCraft are in
//! `photocraft_ui_egui::native_menu`; this file only builds and updates the `NSMenu`s.
//!
//! - Item ids are command ids; a command in two menus (Keyboard Shortcuts is in Edit and Window)
//!   gets `id#2` for its second copy, as muda ids must be unique.
//! - A structure change (a new Open Recent file, a language switch) rebuilds the menu; anything
//!   else (labels, enabled, checked) is updated in place. Opening or closing a document rebuilds
//!   only the Window menu, whose list of documents changed: AppKit adds its own items to that
//!   menu (window tiling, the window list), so muda can't insert rows there by position.
//! - A chosen item is a click or a key equivalent: AppKit's current event says which. Key
//!   equivalents go back to egui as the key press that was made (the event's modifiers, not the
//!   item's: AppKit may match ⌘R to a ⇧⌘R item), so PhotoCraft's key rules still apply.
//! - Hide is a custom item calling `NSApplication hide:`, as the predefined one always takes ⌘H.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, channel};

use muda::accelerator::{Accelerator, Code, Modifiers};
use muda::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use objc2::MainThreadMarker;
use objc2_app_kit::{NSApplication, NSEventModifierFlags, NSEventType};
use photocraft_ui_egui::native_menu::{self, Backend, Chord, Event, MenuBar, MenuRole, Node, Standard};

enum Handle {
    Plain(MenuItem),
    Check(CheckMenuItem),
}

/// One native item.
struct Entry {
    handle: Handle,
    /// The command id (several entries can share one).
    command: String,
    chord: Option<Chord>,
    label: String,
    enabled: bool,
}

/// How an item was chosen, read from AppKit's current event while it calls the handler.
struct Chosen {
    id: String,
    /// A key press (not a mouse click), and the modifiers held: ⌘, ⌃, ⌥, ⇧.
    key: bool,
    mods: [bool; 4],
}

pub struct MacMenu {
    menu: Option<Menu>,
    /// By muda id.
    entries: HashMap<String, Entry>,
    rx: Receiver<Chosen>,
    structure: u64,
    /// The Window menu, the muda ids of its items, and its [`native_menu::window_key`].
    window: Option<(Submenu, Vec<String>, u64)>,
}

impl MacMenu {
    pub fn new(ctx: &egui::Context) -> MacMenu {
        let (tx, rx) = channel();
        let repaint = ctx.clone();
        MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
            // AppKit calls this on the main thread, from the event that chose the item: a key
            // equivalent is a key-down, a click a mouse-up (or ↩ inside the open menu).
            let event = MainThreadMarker::new().and_then(|mtm| NSApplication::sharedApplication(mtm).currentEvent());
            let key = event.as_ref().is_some_and(|ev| ev.r#type() == NSEventType::KeyDown);
            let flags = event.as_ref().map(|ev| ev.modifierFlags()).unwrap_or(NSEventModifierFlags::empty());
            let mods = [NSEventModifierFlags::Command, NSEventModifierFlags::Control, NSEventModifierFlags::Option, NSEventModifierFlags::Shift]
                .map(|m| flags.contains(m));
            let _ = tx.send(Chosen { id: e.id.0, key, mods });
            // Wake egui, so the item runs now rather than on the next mouse move.
            repaint.request_repaint();
        }));
        MacMenu { menu: None, entries: HashMap::new(), rx, structure: 0, window: None }
    }

    fn build(&mut self, bar: &MenuBar) {
        if let Some(old) = self.menu.take() {
            old.remove_for_nsapp();
        }
        self.entries.clear();
        self.window = None;
        let menu = Menu::new();
        let mut special = Vec::new();
        for m in &bar.menus {
            let (sub, ids) = self.submenu(m);
            let _ = menu.append(&sub);
            if m.role == MenuRole::Window {
                self.window = Some((sub.clone(), ids, native_menu::window_key(bar)));
            }
            if matches!(m.role, MenuRole::Window | MenuRole::Help) {
                special.push((m.role, sub));
            }
        }
        menu.init_for_nsapp();
        // Only once the menu is NSApp's: muda looks the submenus up in the installed main menu.
        for (role, sub) in special {
            match role {
                MenuRole::Window => sub.set_as_windows_menu_for_nsapp(),
                _ => sub.set_as_help_menu_for_nsapp(),
            }
        }
        self.menu = Some(menu);
        self.structure = native_menu::structure_key(bar);
    }

    /// A top-level menu, and the muda ids of the items it added.
    fn submenu(&mut self, m: &native_menu::Menu) -> (Submenu, Vec<String>) {
        let before: HashSet<String> = self.entries.keys().cloned().collect();
        let sub = Submenu::new(escape(&m.title), true);
        self.append(&sub, &m.children);
        let ids = self.entries.keys().filter(|id| !before.contains(*id)).cloned().collect();
        (sub, ids)
    }

    /// Replace the Window menu (a document opened or closed), leaving the other menus as they are.
    fn rebuild_window(&mut self, bar: &MenuBar) {
        let (Some(menu), Some((old, ids, _))) = (self.menu.clone(), self.window.take()) else { return self.build(bar) };
        let Some((at, spec)) = bar.menus.iter().enumerate().find(|(_, m)| m.role == MenuRole::Window) else { return self.build(bar) };
        if menu.remove(&old).is_err() {
            return self.build(bar);
        }
        for id in ids {
            self.entries.remove(&id);
        }
        let (sub, ids) = self.submenu(spec);
        if menu.insert(&sub, at).is_err() {
            return self.build(bar);
        }
        sub.set_as_windows_menu_for_nsapp();
        self.window = Some((sub, ids, native_menu::window_key(bar)));
    }

    fn append(&mut self, sub: &Submenu, nodes: &[Node]) {
        for n in nodes {
            match n {
                Node::Separator => {
                    let _ = sub.append(&PredefinedMenuItem::separator());
                }
                Node::Submenu { label, children, .. } => {
                    let s = Submenu::new(escape(label), !children.is_empty());
                    self.append(&s, children);
                    let _ = sub.append(&s);
                }
                Node::Standard(item) => {
                    let p = match item {
                        Standard::Services => PredefinedMenuItem::services(Some(photocraft_ui_egui::i18n::t("Services"))),
                        Standard::HideOthers => PredefinedMenuItem::hide_others(Some(photocraft_ui_egui::i18n::t("Hide Others"))),
                        Standard::ShowAll => PredefinedMenuItem::show_all(Some(photocraft_ui_egui::i18n::t("Show All"))),
                        Standard::Minimize => PredefinedMenuItem::minimize(None),
                        Standard::Zoom => PredefinedMenuItem::maximize(Some(photocraft_ui_egui::i18n::t("Zoom"))),
                        Standard::BringAllToFront => PredefinedMenuItem::bring_all_to_front(None),
                        Standard::CloseWindow => PredefinedMenuItem::close_window(None),
                    };
                    let _ = sub.append(&p);
                }
                Node::Item(it) => {
                    let id = self.unique_id(&it.id);
                    let chord = it.shortcut.as_deref().and_then(Chord::parse);
                    let accel = chord.as_ref().filter(|c| c.native_ok()).and_then(accelerator);
                    let handle = match it.checked {
                        Some(c) => {
                            let h = CheckMenuItem::with_id(id.clone(), escape(&it.label), it.enabled, c, accel);
                            let _ = sub.append(&h);
                            Handle::Check(h)
                        }
                        None => {
                            let h = MenuItem::with_id(id.clone(), escape(&it.label), it.enabled, accel);
                            let _ = sub.append(&h);
                            Handle::Plain(h)
                        }
                    };
                    self.entries.insert(id, Entry { handle, command: it.id.clone(), chord, label: it.label.clone(), enabled: it.enabled });
                }
            }
        }
    }

    /// muda ids must be unique: the second `edit.keyboardShortcuts` becomes `…#2`.
    fn unique_id(&self, id: &str) -> String {
        let mut unique = id.to_string();
        let mut n = 2;
        while self.entries.contains_key(&unique) {
            unique = format!("{id}#{n}");
            n += 1;
        }
        unique
    }
}

impl Backend for MacMenu {
    fn sync(&mut self, bar: &MenuBar) {
        if self.menu.is_none() || native_menu::structure_key(bar) != self.structure {
            self.build(bar);
            return;
        }
        if self.window.as_ref().map(|w| w.2) != Some(native_menu::window_key(bar)) {
            self.rebuild_window(bar);
        }
        let items: HashMap<&str, &native_menu::Item> = bar.items().into_iter().map(|it| (it.id.as_str(), it)).collect();
        for e in self.entries.values_mut() {
            let Some(it) = items.get(e.command.as_str()) else { continue };
            if e.label != it.label {
                let text = escape(&it.label);
                match &e.handle {
                    Handle::Plain(h) => h.set_text(text),
                    Handle::Check(h) => h.set_text(text),
                }
                e.label = it.label.clone();
            }
            if e.enabled != it.enabled {
                match &e.handle {
                    Handle::Plain(h) => h.set_enabled(it.enabled),
                    Handle::Check(h) => h.set_enabled(it.enabled),
                }
                e.enabled = it.enabled;
            }
            // Compare with AppKit's own state: muda toggles a check item when it's clicked.
            if let (Handle::Check(h), Some(c)) = (&e.handle, it.checked)
                && h.is_checked() != c
            {
                h.set_checked(c);
            }
        }
    }

    fn drain(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        while let Ok(chosen) = self.rx.try_recv() {
            // Ids that aren't ours are AppKit's own items (Services, Zoom, …), already handled.
            let Some(e) = self.entries.get(&chosen.id) else { continue };
            if e.command == native_menu::HIDE {
                if let Some(mtm) = MainThreadMarker::new() {
                    NSApplication::sharedApplication(mtm).hide(None);
                }
                continue;
            }
            // A key-down chose it: its key equivalent, unless it was ↩ or Space inside the open
            // menu (no ⌘ or ⌃, and the item's shortcut needs one). The key is the item's (AppKit
            // matched on it); the modifiers are the ones held.
            let [cmd, ctrl, alt, shift] = chosen.mods;
            let key =
                e.chord.as_ref().filter(|c| chosen.key && (cmd || ctrl || !(c.cmd || c.ctrl))).map(|c| Chord { cmd, ctrl, alt, shift, key: c.key.clone() });
            out.push(match key {
                Some(chord) if e.command != native_menu::MINIMIZE => Event::Key(chord),
                _ => Event::Click(e.command.clone()),
            });
        }
        out
    }
}

/// Install the menu bar now (in eframe's creator closure, so winit's default menu doesn't stay).
pub fn install(ctx: &egui::Context, app: &photocraft_ui_egui::PhotocraftApp) -> Option<native_menu::NativeMenu> {
    let language = &app.session.prefs().interface.language;
    let lang = photocraft_ui_egui::i18n::Lang::from_pref(language);
    let bar = native_menu::photocraft_layout(&photocraft_ui_egui::menus::menu_items(app), lang, language).bar;
    // Never crash for a menu: if AppKit refuses, keep the in-window menus.
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut menu = MacMenu::new(ctx);
        menu.sync(&bar);
        menu
    }));
    match built {
        Ok(menu) => Some(native_menu::NativeMenu::new(Box::new(menu))),
        Err(_) => {
            log::warn!("couldn't install the macOS menu bar; using the in-window menus");
            None
        }
    }
}

/// `&` marks a mnemonic in muda labels.
fn escape(s: &str) -> String {
    s.replace('&', "&&")
}

/// A key equivalent for AppKit. `Cmd` is ⌘ (`META`) and `Ctrl` is ⌃: never folded together.
fn accelerator(c: &Chord) -> Option<Accelerator> {
    let mut mods = Modifiers::empty();
    for (on, m) in [(c.cmd, Modifiers::META), (c.ctrl, Modifiers::CONTROL), (c.alt, Modifiers::ALT), (c.shift, Modifiers::SHIFT)] {
        if on {
            mods |= m;
        }
    }
    const LETTERS: [Code; 26] = [
        Code::KeyA,
        Code::KeyB,
        Code::KeyC,
        Code::KeyD,
        Code::KeyE,
        Code::KeyF,
        Code::KeyG,
        Code::KeyH,
        Code::KeyI,
        Code::KeyJ,
        Code::KeyK,
        Code::KeyL,
        Code::KeyM,
        Code::KeyN,
        Code::KeyO,
        Code::KeyP,
        Code::KeyQ,
        Code::KeyR,
        Code::KeyS,
        Code::KeyT,
        Code::KeyU,
        Code::KeyV,
        Code::KeyW,
        Code::KeyX,
        Code::KeyY,
        Code::KeyZ,
    ];
    const DIGITS: [Code; 10] =
        [Code::Digit0, Code::Digit1, Code::Digit2, Code::Digit3, Code::Digit4, Code::Digit5, Code::Digit6, Code::Digit7, Code::Digit8, Code::Digit9];
    const F: [Code; 24] = [
        Code::F1,
        Code::F2,
        Code::F3,
        Code::F4,
        Code::F5,
        Code::F6,
        Code::F7,
        Code::F8,
        Code::F9,
        Code::F10,
        Code::F11,
        Code::F12,
        Code::F13,
        Code::F14,
        Code::F15,
        Code::F16,
        Code::F17,
        Code::F18,
        Code::F19,
        Code::F20,
        Code::F21,
        Code::F22,
        Code::F23,
        Code::F24,
    ];
    let k = c.key.as_str();
    let mut chars = k.chars();
    let code = match (chars.next(), chars.next()) {
        (Some(ch), None) if ch.is_ascii_alphabetic() => *LETTERS.get((ch.to_ascii_uppercase() as usize).checked_sub('A' as usize)?)?,
        (Some(ch), None) if ch.is_ascii_digit() => *DIGITS.get((ch as usize).checked_sub('0' as usize)?)?,
        _ if c.is_function_key() => *F.get(k.get(1..)?.parse::<usize>().ok()?.checked_sub(1)?)?,
        _ => match k {
            "[" => Code::BracketLeft,
            "]" => Code::BracketRight,
            "\\" => Code::Backslash,
            "/" => Code::Slash,
            "=" => Code::Equal,
            "-" => Code::Minus,
            "'" => Code::Quote,
            ";" => Code::Semicolon,
            "," => Code::Comma,
            "." => Code::Period,
            "`" => Code::Backquote,
            "Delete" | "Backspace" => Code::Backspace,
            "Enter" if !mods.is_empty() => Code::Enter,
            "Tab" if !mods.is_empty() => Code::Tab,
            "Space" if !mods.is_empty() => Code::Space,
            _ => return None,
        },
    };
    Some(Accelerator::new(mods, code))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shortcut the Mac menu shows becomes a key equivalent (or stays egui's, for bare keys).
    #[test]
    fn every_menu_shortcut_maps_to_a_key_equivalent() {
        let app = photocraft_ui_egui::PhotocraftApp::new(photocraft_engine::Session::new(), photocraft_ui_egui::Services::default());
        let bar = native_menu::photocraft_layout(&photocraft_ui_egui::menus::menu_items(&app), photocraft_ui_egui::i18n::Lang::EN, "auto").bar;
        let mut native = 0;
        for it in bar.items() {
            let Some(c) = it.shortcut.as_deref().and_then(Chord::parse) else { continue };
            if c.native_ok() {
                assert!(accelerator(&c).is_some(), "{} ({}) has no key equivalent", c.portable(), it.id);
                native += 1;
            }
        }
        assert!(native > 70, "{native} key equivalents");
    }

    #[test]
    fn control_and_command_stay_distinct() {
        let a = accelerator(&Chord::parse("Cmd+M").unwrap()).unwrap();
        let b = accelerator(&Chord::parse("Ctrl+Cmd+M").unwrap()).unwrap();
        assert_ne!(a, b);
    }
}
