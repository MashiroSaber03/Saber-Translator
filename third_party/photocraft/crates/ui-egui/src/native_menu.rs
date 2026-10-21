//! The macOS menu bar: the same menus as the in-window bar ([`crate::menus::menu_items`]), laid
//! out the way a Mac app's are and handed to a platform [`Backend`] (the desktop app's muda one).
//!
//! Pure and platform-free, so it is tested everywhere and builds for the web; only the backend
//! touches AppKit. What the layout does ([`mac_layout`]):
//! - an **app menu** named PhotoCraft: About (from Help), Settings (Edit › Preferences), Language,
//!   Appearance (Window › Theme), Services, Hide, Hide Others, Show All and Quit (File › Exit,
//!   still the app's own `file.exit` so the unsaved-changes prompt runs). The other Craft apps
//!   use the same app menu;
//! - **Window** gets Minimize, Zoom and Bring All to Front, with the open documents after it
//!   (where a Mac app lists its windows), and **Help** the system search field;
//! - **clashes** with the system's keys (⌘H, ⌘M, ⌘W, ⌘,) are resolved in PhotoCraft's favour, and
//!   reported; Hide and Minimize move to ⌃⌘H and ⌃⌘M, as Photoshop does.
//!
//! **Keys stay with egui.** AppKit runs a menu's key equivalents before the window sees the key,
//! which would skip PhotoCraft's key rules (text fields keep ⌘C/⌘V/⌘Z, dialogs and Camera Raw are
//! modal, Liquify has its own ⌘Z). So the backend reports a key equivalent as [`Event::Key`], and
//! [`NativeMenu::raw_input`] feeds it to egui as the key press it was: the shortcut runs through
//! [`crate::shortcuts::handle`] exactly as without a native menu. A click is [`Event::Click`] and
//! runs the command like an in-window menu click.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use egui::{Key, Modifiers};

use crate::PhotocraftApp;
use crate::i18n::{Lang, tr, tr_id};
use crate::menus::MenuItem;

pub const APP_NAME: &str = "PhotoCraft";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MenuBar {
    pub menus: Vec<Menu>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Menu {
    pub title: String,
    pub role: MenuRole,
    pub children: Vec<Node>,
}

/// What a top-level menu is for. macOS treats the app, Window and Help menus specially.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MenuRole {
    #[default]
    Normal,
    App,
    Window,
    Help,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Item(Item),
    Separator,
    Submenu {
        label: String,
        children: Vec<Node>,
        role: Option<ItemRole>,
    },
    /// An item AppKit provides and runs itself (Services, Zoom, …).
    Standard(Standard),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// The command id (`file.open`), or one of [`HIDE`] / [`MINIMIZE`].
    pub id: String,
    pub label: String,
    /// `Cmd+Shift+Z` form; `Ctrl` is the real Control key.
    pub shortcut: Option<String>,
    pub enabled: bool,
    /// `Some` for check items.
    pub checked: Option<bool>,
    pub role: Option<ItemRole>,
}

/// Items macOS expects in fixed places.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemRole {
    About,
    Settings,
    Quit,
    Hide,
    Minimize,
    /// Window › Theme and Next Theme, which become the app menu's Appearance submenu.
    Appearance,
    /// The app menu's Language submenu (its labels are final, never translated by id).
    Language,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Standard {
    Services,
    HideOthers,
    ShowAll,
    Minimize,
    Zoom,
    BringAllToFront,
    CloseWindow,
}

/// Hide PhotoCraft: a custom item (the predefined one always takes ⌘H, which is View › Extras).
pub const HIDE: &str = "app.hide";
/// Minimize: a custom item (the predefined one always takes ⌘M, which is Curves).
pub const MINIMIZE: &str = "window.minimize";

impl Item {
    fn new(id: &str, label: &str) -> Item {
        Item { id: id.into(), label: label.into(), shortcut: None, enabled: true, checked: None, role: None }
    }
}

impl MenuBar {
    /// Every item, depth first.
    pub fn items(&self) -> Vec<&Item> {
        fn walk<'a>(nodes: &'a [Node], out: &mut Vec<&'a Item>) {
            for n in nodes {
                match n {
                    Node::Item(it) => out.push(it),
                    Node::Submenu { children, .. } => walk(children, out),
                    Node::Separator | Node::Standard(_) => {}
                }
            }
        }
        let mut out = Vec::new();
        for m in &self.menus {
            walk(&m.children, &mut out);
        }
        out
    }

    pub fn find(&self, id: &str) -> Option<&Item> {
        self.items().into_iter().find(|it| it.id == id)
    }

    pub fn menu(&self, role: MenuRole) -> Option<&Menu> {
        self.menus.iter().find(|m| m.role == role)
    }

    fn tag_item(&mut self, id: &str, role: ItemRole) {
        fn walk(nodes: &mut [Node], id: &str, role: ItemRole) {
            for n in nodes {
                match n {
                    Node::Item(it) if it.id == id => it.role = Some(role),
                    Node::Submenu { children, .. } => walk(children, id, role),
                    _ => {}
                }
            }
        }
        for m in &mut self.menus {
            walk(&mut m.children, id, role);
        }
    }

    fn tag_submenu(&mut self, top: &str, name: &str, role: ItemRole) {
        let Some(menu) = self.menus.iter_mut().find(|m| m.title == top) else { return };
        for n in &mut menu.children {
            if let Node::Submenu { label, role: r, .. } = n
                && label == name
            {
                *r = Some(role);
            }
        }
    }

    fn tag_menu(&mut self, title: &str, role: MenuRole) {
        if let Some(m) = self.menus.iter_mut().find(|m| m.title == title) {
            m.role = role;
        }
    }
}

/// The menus as a tree, from the flat rows [`crate::menus::menu_items`] returns, in the order the
/// in-window bar draws them: at each depth leaves and separators in order, and a submenu where its
/// first child is.
pub fn from_items(tops: &[&str], items: &[MenuItem]) -> MenuBar {
    fn level(rows: &[&MenuItem], depth: usize) -> Vec<Node> {
        let mut out = Vec::new();
        let mut shown: Vec<&str> = Vec::new();
        for it in rows {
            if it.path.len() == depth {
                if it.label == "---" {
                    out.push(Node::Separator);
                } else {
                    out.push(Node::Item(Item {
                        id: it.id.clone(),
                        label: it.label.clone(),
                        shortcut: it.shortcut.clone(),
                        enabled: it.enabled,
                        checked: it.checked,
                        role: None,
                    }));
                }
            } else if let Some(name) = it.path.get(depth) {
                let name = name.as_str();
                if shown.contains(&name) {
                    continue;
                }
                shown.push(name);
                let child: Vec<&MenuItem> = rows.iter().copied().filter(|c| c.path.get(depth).map(String::as_str) == Some(name)).collect();
                out.push(Node::Submenu { label: name.to_string(), children: level(&child, depth + 1), role: None });
            }
        }
        tidy(out)
    }
    let menus = tops
        .iter()
        .map(|top| {
            let rows: Vec<&MenuItem> = items.iter().filter(|i| i.path.first().map(String::as_str) == Some(*top)).collect();
            Menu { title: top.to_string(), role: MenuRole::Normal, children: level(&rows, 1) }
        })
        .collect();
    MenuBar { menus }
}

/// Drop separators at the start or end of a level and collapse runs of them, at every depth.
fn tidy(nodes: Vec<Node>) -> Vec<Node> {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    for n in nodes {
        let n = match n {
            Node::Submenu { label, children, role } => Node::Submenu { label, children: tidy(children), role },
            other => other,
        };
        if matches!(n, Node::Separator) && matches!(out.last(), None | Some(Node::Separator)) {
            continue;
        }
        out.push(n);
    }
    while matches!(out.last(), Some(Node::Separator)) {
        out.pop();
    }
    out
}

/// A hash of everything a native menu can only change by rebuilding: menus, roles, item ids,
/// kinds (plain or check), shortcuts and submenu labels. Labels, enabled and checked are updated
/// in place. Window's list of open documents is left out: opening or closing a document rebuilds
/// only the Window menu ([`window_key`]).
pub fn structure_key(bar: &MenuBar) -> u64 {
    let mut h = DefaultHasher::new();
    for m in &bar.menus {
        (&m.title, m.role).hash(&mut h);
        if m.role == MenuRole::Window {
            let mut rest: Vec<Node> = m.children.iter().filter(|n| !is_document(n)).cloned().collect();
            while matches!(rest.last(), Some(Node::Separator)) {
                rest.pop();
            }
            hash_nodes(&rest, &mut h);
        } else {
            hash_nodes(&m.children, &mut h);
        }
    }
    h.finish()
}

/// The Window menu's structure, its open documents included: when it changes and nothing else
/// did, a native menu rebuilds just the Window menu.
pub fn window_key(bar: &MenuBar) -> u64 {
    let mut h = DefaultHasher::new();
    if let Some(m) = bar.menu(MenuRole::Window) {
        hash_nodes(&m.children, &mut h);
    }
    h.finish()
}

/// Window › <document> (see [`crate::menus::DOCUMENT_ITEM`]).
fn is_document(n: &Node) -> bool {
    matches!(n, Node::Item(it) if it.id.starts_with(crate::menus::DOCUMENT_ITEM))
}

fn hash_nodes(nodes: &[Node], h: &mut DefaultHasher) {
    for n in nodes {
        match n {
            Node::Item(it) => ('i', &it.id, it.checked.is_some(), &it.shortcut, it.role).hash(h),
            Node::Separator => '-'.hash(h),
            Node::Submenu { label, children, role } => {
                ('[', label, role).hash(h);
                hash_nodes(children, h);
                ']'.hash(h);
            }
            Node::Standard(s) => ('s', s).hash(h),
        }
    }
}

// ----------------------------------------------------------------------------- shortcuts

/// A shortcut as PhotoCraft writes it (`Cmd+Shift+Z`). `Cmd` is ⌘ and `Ctrl` the Control key,
/// never folded together: ⌃⌘M and ⌘M are different shortcuts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Chord {
    pub cmd: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// The key as written: `Z`, `1`, `'`, `F5`, `Delete`, `Enter`.
    pub key: String,
}

impl Chord {
    pub fn parse(s: &str) -> Option<Chord> {
        let mut c = Chord { cmd: false, ctrl: false, alt: false, shift: false, key: String::new() };
        for part in s.split('+') {
            match part {
                "Cmd" => c.cmd = true,
                "Ctrl" => c.ctrl = true,
                "Alt" => c.alt = true,
                "Shift" => c.shift = true,
                // `Cmd++` splits into an empty last part.
                "" => c.key = "+".into(),
                k if c.key.is_empty() => c.key = k.into(),
                _ => return None,
            }
        }
        (!c.key.is_empty()).then_some(c)
    }

    /// The written form, modifiers in a fixed order.
    pub fn portable(&self) -> String {
        let mut parts = Vec::new();
        for (on, name) in [(self.ctrl, "Ctrl"), (self.cmd, "Cmd"), (self.alt, "Alt"), (self.shift, "Shift")] {
            if on {
                parts.push(name);
            }
        }
        parts.push(&self.key);
        parts.join("+")
    }

    pub fn is_function_key(&self) -> bool {
        self.key.strip_prefix('F').and_then(|n| n.parse::<u8>().ok()).is_some_and(|n| (1..=24).contains(&n))
    }

    /// May the native menu show this as a key equivalent? Only with ⌘ or ⌃, or a function key.
    /// Bare letters and ⌥/⇧-only keys are tool keys: as key equivalents AppKit would take a `b`
    /// typed into a text field.
    pub fn native_ok(&self) -> bool {
        self.cmd || self.ctrl || self.is_function_key()
    }

    /// The key press egui would have received for this chord.
    pub fn key_event(&self, pressed: bool) -> Option<egui::Event> {
        let key = match self.key.as_str() {
            "+" => Key::Plus,
            "'" => Key::Quote,
            ";" => Key::Semicolon,
            "=" => Key::Equals,
            "-" => Key::Minus,
            "[" => Key::OpenBracket,
            "]" => Key::CloseBracket,
            "\\" => Key::Backslash,
            "/" => Key::Slash,
            "," => Key::Comma,
            "." => Key::Period,
            k => Key::from_name(k)?,
        };
        let modifiers = Modifiers { alt: self.alt, ctrl: self.ctrl, shift: self.shift, mac_cmd: self.cmd, command: self.cmd };
        Some(egui::Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers })
    }
}

// ----------------------------------------------------------------------------- macOS layout

/// A system key equivalent PhotoCraft already uses for something else.
#[derive(Clone, Debug, PartialEq)]
pub struct Clash {
    pub shortcut: &'static str,
    /// What macOS uses it for.
    pub system: &'static str,
    /// The PhotoCraft command bound to it.
    pub command: String,
    /// What the layout did about it.
    pub resolution: &'static str,
}

pub struct Layout {
    pub bar: MenuBar,
    pub clashes: Vec<Clash>,
}

/// Turn PhotoCraft's menus (tagged with roles) into a Mac menu bar. See the module docs.
pub fn mac_layout(bar: &MenuBar, lang: Lang) -> Layout {
    let mut bar = bar.clone();
    let mut clashes = Vec::new();
    let bound: HashMap<String, String> = bar.items().iter().filter_map(|it| Some((Chord::parse(it.shortcut.as_deref()?)?.portable(), it.id.clone()))).collect();
    let taken = |sc: &str, own: Option<&str>| bound.get(sc).filter(|cmd| Some(cmd.as_str()) != own).cloned();
    let named = |s: &str| crate::i18n::fmt(tr(lang, s), &[("app", APP_NAME)]);

    let about = take(&mut bar, |n| matches!(n, Node::Item(it) if it.role == Some(ItemRole::About)));
    let settings = take(&mut bar, |n| match n {
        Node::Item(it) => it.role == Some(ItemRole::Settings),
        Node::Submenu { role, .. } => *role == Some(ItemRole::Settings),
        _ => false,
    });
    let quit = take(&mut bar, |n| matches!(n, Node::Item(it) if it.role == Some(ItemRole::Quit)));
    let appearance = appearance_node(&mut bar, lang);

    let mut menu = Vec::new();
    if let Some(Node::Item(mut it)) = about {
        it.label = named("About {app}");
        menu.extend([Node::Item(it), Node::Separator]);
    }
    if let Some(s) = settings {
        let own = node_ids(&s);
        let comma = bound.get("Cmd+,").filter(|cmd| !own.contains(*cmd)).cloned();
        menu.push(settings_node(s, comma, lang, &mut clashes));
    }
    menu.extend(appearance);
    if !menu.is_empty() && menu.last() != Some(&Node::Separator) {
        menu.push(Node::Separator);
    }
    menu.extend([Node::Standard(Standard::Services), Node::Separator]);
    let mut hide = Item::new(HIDE, &named("Hide {app}"));
    hide.role = Some(ItemRole::Hide);
    match taken("Cmd+H", None) {
        Some(command) => {
            let free = taken("Ctrl+Cmd+H", None).is_none();
            if free {
                hide.shortcut = Some("Ctrl+Cmd+H".into());
            }
            let resolution = if free { "Hide moves to ⌃⌘H" } else { "Hide has no shortcut" };
            clashes.push(Clash { shortcut: "Cmd+H", system: "Hide", command, resolution });
        }
        None => hide.shortcut = Some("Cmd+H".into()),
    }
    menu.push(Node::Item(hide));
    if let Some(command) = taken("Cmd+Alt+H", None) {
        clashes.push(Clash { shortcut: "Cmd+Alt+H", system: "Hide Others", command, resolution: "AppKit's Hide Others keeps ⌥⌘H" });
    }
    menu.extend([Node::Standard(Standard::HideOthers), Node::Standard(Standard::ShowAll), Node::Separator]);
    if let Some(Node::Item(mut quit)) = quit {
        quit.label = named("Quit {app}");
        match taken("Cmd+Q", Some(&quit.id)) {
            Some(command) => {
                clashes.push(Clash { shortcut: "Cmd+Q", system: "Quit", command, resolution: "Quit has no shortcut" });
                quit.shortcut = None;
            }
            None => quit.shortcut = Some("Cmd+Q".into()),
        }
        menu.push(Node::Item(quit));
    }
    bar.menus.insert(0, Menu { title: APP_NAME.into(), role: MenuRole::App, children: menu });

    match taken("Cmd+W", None) {
        Some(command) => clashes.push(Clash { shortcut: "Cmd+W", system: "Close Window", command, resolution: "no system Close Window item" }),
        None => {
            if let Some(file) = bar.menus.iter_mut().find(|m| m.title == "File") {
                file.children.extend([Node::Separator, Node::Standard(Standard::CloseWindow)]);
            }
        }
    }

    if bar.menu(MenuRole::Window).is_none() {
        let at = bar.menus.iter().position(|m| m.role == MenuRole::Help).unwrap_or(bar.menus.len());
        bar.menus.insert(at, Menu { title: "Window".into(), role: MenuRole::Window, children: Vec::new() });
    }
    let minimize = match taken("Cmd+M", None) {
        Some(command) => {
            let mut it = Item::new(MINIMIZE, tr(lang, "Minimize"));
            it.role = Some(ItemRole::Minimize);
            let free = taken("Ctrl+Cmd+M", None).is_none();
            if free {
                it.shortcut = Some("Ctrl+Cmd+M".into());
            }
            let resolution = if free { "Minimize moves to ⌃⌘M" } else { "Minimize has no shortcut" };
            clashes.push(Clash { shortcut: "Cmd+M", system: "Minimize", command, resolution });
            Node::Item(it)
        }
        None => Node::Standard(Standard::Minimize),
    };
    if let Some(window) = bar.menus.iter_mut().find(|m| m.role == MenuRole::Window) {
        let (docs, rest): (Vec<Node>, Vec<Node>) = std::mem::take(&mut window.children).into_iter().partition(is_document);
        let mut children = vec![minimize, Node::Standard(Standard::Zoom), Node::Separator];
        children.extend(rest);
        children.extend([Node::Separator, Node::Standard(Standard::BringAllToFront), Node::Separator]);
        children.extend(docs);
        window.children = children;
    }

    for m in &mut bar.menus {
        m.children = tidy(std::mem::take(&mut m.children));
    }
    // A menu emptied by the moves goes away (the Help menu stays: it has the search field).
    bar.menus.retain(|m| !m.children.is_empty() || m.role == MenuRole::Help);
    Layout { bar, clashes }
}

/// Settings in the app menu: Edit › Preferences, a submenu of settings pages, becomes "Settings"
/// with ⌘, on its first page unless PhotoCraft binds ⌘, elsewhere.
fn settings_node(node: Node, comma_taken: Option<String>, lang: Lang, clashes: &mut Vec<Clash>) -> Node {
    if let Some(command) = &comma_taken {
        clashes.push(Clash { shortcut: "Cmd+,", system: "Settings", command: command.clone(), resolution: "Settings has no shortcut" });
    }
    let give = |it: &mut Item| {
        if comma_taken.is_none() && it.shortcut.is_none() {
            it.shortcut = Some("Cmd+,".into());
        }
    };
    match node {
        Node::Item(mut it) => {
            it.label = tr(lang, "Settings…").into();
            give(&mut it);
            Node::Item(it)
        }
        Node::Submenu { mut children, role, .. } => {
            if let Some(Node::Item(first)) = children.iter_mut().find(|n| matches!(n, Node::Item(i) if i.enabled)) {
                give(first);
            }
            Node::Submenu { label: tr(lang, "Settings").into(), children, role }
        }
        other => other,
    }
}

/// Appearance in the app menu: the themes (Window › Theme, tagged [`ItemRole::Appearance`]), then
/// Next Theme, both leaving the Window menu. None when there are no themes.
fn appearance_node(bar: &mut MenuBar, lang: Lang) -> Option<Node> {
    let Some(Node::Submenu { mut children, .. }) = take(bar, |n| matches!(n, Node::Submenu { role, .. } if *role == Some(ItemRole::Appearance))) else {
        return None;
    };
    if let Some(Node::Item(mut next)) = take(bar, |n| matches!(n, Node::Item(it) if it.role == Some(ItemRole::Appearance))) {
        // Placed now: translated by id like the themes above it.
        next.role = None;
        children.extend([Node::Separator, Node::Item(next)]);
    }
    Some(Node::Submenu { label: tr(lang, "Appearance").into(), children, role: Some(ItemRole::Appearance) })
}

/// Command ids of the app menu's Language items: `app.language.<code>`, where `auto` follows the
/// system (the `interface.language` preference values).
pub const LANGUAGE_PREFIX: &str = "app.language.";

/// Language in the app menu, after Settings: Auto, then every UI language in its own name, with
/// the current preference checked.
fn language_node(pref: &str, lang: Lang) -> Node {
    let item = |code: &str, label: &str| Item {
        id: format!("{LANGUAGE_PREFIX}{code}"),
        label: label.into(),
        shortcut: None,
        enabled: true,
        checked: Some(pref == code),
        role: Some(ItemRole::Language),
    };
    let mut children = vec![Node::Item(item("auto", tr(lang, "Auto"))), Node::Separator];
    children.extend(Lang::all().map(|l| Node::Item(item(l.code(), l.name()))));
    Node::Submenu { label: tr(lang, "Language").into(), children, role: Some(ItemRole::Language) }
}

fn node_ids(node: &Node) -> Vec<String> {
    match node {
        Node::Item(it) => vec![it.id.clone()],
        Node::Submenu { children, .. } => children.iter().flat_map(node_ids).collect(),
        _ => Vec::new(),
    }
}

/// Remove and return the first node matching `pred`, at any depth.
fn take(bar: &mut MenuBar, pred: impl Fn(&Node) -> bool + Copy) -> Option<Node> {
    fn walk(nodes: &mut Vec<Node>, pred: impl Fn(&Node) -> bool + Copy) -> Option<Node> {
        if let Some(i) = nodes.iter().position(pred) {
            return Some(nodes.remove(i));
        }
        nodes.iter_mut().find_map(|n| match n {
            Node::Submenu { children, .. } => walk(children, pred),
            _ => None,
        })
    }
    bar.menus.iter_mut().find_map(|m| walk(&mut m.children, pred))
}

/// Translate menu titles, submenus and items the layout didn't name itself.
fn translate(bar: &mut MenuBar, lang: Lang) {
    fn walk(nodes: &mut [Node], lang: Lang) {
        for n in nodes {
            match n {
                // A document's name is shown as it is.
                Node::Item(it) if it.role.is_none() && !it.id.starts_with(crate::menus::DOCUMENT_ITEM) => it.label = tr_id(lang, &it.id, &it.label).to_string(),
                Node::Submenu { label, children, role } => {
                    if role.is_none() {
                        *label = tr(lang, label).to_string();
                    }
                    walk(children, lang);
                }
                _ => {}
            }
        }
    }
    for m in &mut bar.menus {
        if m.role != MenuRole::App {
            m.title = tr(lang, &m.title).to_string();
        }
        walk(&mut m.children, lang);
    }
}

/// PhotoCraft's menus as a Mac menu bar, in the UI language; `language` is the
/// `interface.language` preference (`auto` or a code), checked in the app menu's Language.
pub fn photocraft_layout(items: &[MenuItem], lang: Lang, language: &str) -> Layout {
    let mut bar = from_items(&crate::menus::TOP_MENUS, items);
    bar.tag_item("help.about", ItemRole::About);
    bar.tag_item("file.exit", ItemRole::Quit);
    bar.tag_submenu("Edit", "Preferences", ItemRole::Settings);
    bar.tag_submenu("Window", "Theme", ItemRole::Appearance);
    bar.tag_item("window.theme.toggle", ItemRole::Appearance);
    bar.tag_menu("Window", MenuRole::Window);
    bar.tag_menu("Help", MenuRole::Help);
    let mut layout = mac_layout(&bar, lang);
    // Language goes right after Settings (before Appearance), as in the other Craft apps.
    if let Some(app_menu) = layout.bar.menus.iter_mut().find(|m| m.role == MenuRole::App) {
        let settings = |n: &Node| match n {
            Node::Item(it) => it.role == Some(ItemRole::Settings),
            Node::Submenu { role, .. } => *role == Some(ItemRole::Settings),
            _ => false,
        };
        let at = app_menu.children.iter().position(settings).map_or(0, |i| i + 1);
        app_menu.children.insert(at, language_node(language, lang));
    }
    translate(&mut layout.bar, lang);
    layout
}

// ----------------------------------------------------------------------------- the app's side

/// What the native menu reports.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// An item was chosen with the mouse (or the keyboard inside an open menu).
    Click(String),
    /// A key equivalent was pressed: the shortcut, to be handled as the key press it was.
    Key(Chord),
}

/// A platform's native menu bar (the desktop app implements it for macOS with muda).
pub trait Backend {
    /// Bring the menu in step with `bar`: update labels, enabled and checked in place, or
    /// rebuild when its structure changed.
    fn sync(&mut self, bar: &MenuBar);
    /// Events since the last call. Platform items (Hide, Services, Zoom, …) are handled by the
    /// backend and not reported.
    fn drain(&mut self) -> Vec<Event>;
}

/// The native menu bar, while PhotoCraft uses one (the in-window menus are hidden then).
pub struct NativeMenu {
    backend: Box<dyn Backend>,
    /// Hash of the state the menus were last built from.
    state: Option<u64>,
    clicks: Vec<String>,
    /// AppKit toggled a check item itself: re-sync.
    dirty: bool,
    logged: bool,
}

impl NativeMenu {
    pub fn new(backend: Box<dyn Backend>) -> NativeMenu {
        NativeMenu { backend, state: None, clicks: Vec::new(), dirty: true, logged: false }
    }

    /// Before egui sees this frame's input: key equivalents become the key presses they were,
    /// and clicks wait for [`NativeMenu::run`].
    ///
    /// AppKit takes only the key-down: winit still delivers the key-up (it forwards key-ups while
    /// ⌘ is held). So a key equivalent adds just its press, ahead of its key-up when that is
    /// already queued. A second, made-up release made ⌘V paste three times (#1638): a ⌘V release
    /// without a press is how an image paste reaches us (see [`crate::shortcuts::clipboard_keys`]).
    pub fn raw_input(&mut self, raw: &mut egui::RawInput) {
        for e in self.backend.drain() {
            match e {
                Event::Key(chord) => {
                    if let Some(press @ egui::Event::Key { key: pressed_key, .. }) = chord.key_event(true) {
                        let at = raw
                            .events
                            .iter()
                            .position(|e| matches!(e, egui::Event::Key { key, pressed: false, .. } if *key == pressed_key))
                            .unwrap_or(raw.events.len());
                        raw.events.insert(at, press);
                    }
                }
                Event::Click(id) => self.clicks.push(id),
            }
            self.dirty = true;
        }
    }
}

/// Run the native menu clicks that arrived since the last frame, like in-window menu clicks.
pub fn run(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(menu) = app.services.native_menu.as_mut() else { return };
    let clicks = std::mem::take(&mut menu.clicks);
    for id in clicks {
        if id == MINIMIZE {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
            continue;
        }
        // The rows are disabled while a dialog is open; a click that raced the update is dropped.
        if !crate::menus::modal_allows(app, &id) {
            continue;
        }
        if let Err(e) = crate::menus::invoke(app, ctx, &id, serde_json::json!({})) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
}

/// Bring the native menu up to date when anything it shows may have changed: a command ran, the
/// document or selection changed, or the user clicked or pressed a key. Building the rows walks
/// every command's enabled state, so it doesn't run on frames that only animate or scroll.
pub fn sync(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(menu) = app.services.native_menu.as_ref() else { return };
    let state = state_hash(app, had_input(ctx).then_some(app.frame));
    if menu.state == Some(state) && !menu.dirty {
        return;
    }
    let lang = crate::i18n::current();
    let mut items = crate::menus::menu_items(app);
    for it in &mut items {
        it.enabled &= crate::menus::modal_allows(app, &it.id);
    }
    let layout = photocraft_layout(&items, lang, &app.session.prefs().interface.language);
    let Some(menu) = app.services.native_menu.as_mut() else { return };
    if !menu.logged {
        for c in &layout.clashes {
            log::info!("menu: {} is {} on macOS and {} in PhotoCraft; {}", c.shortcut, c.system, c.command, c.resolution);
        }
        menu.logged = true;
    }
    menu.backend.sync(&layout.bar);
    menu.state = Some(state);
    menu.dirty = false;
}

/// A click or key press this frame: it may have changed state [`state_hash`] doesn't list.
pub(crate) fn had_input(ctx: &egui::Context) -> bool {
    ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. } | egui::Event::PointerButton { pressed: false, .. })))
}

/// What the menus' rows depend on, cheaply: commands run, the documents (their names, in tab
/// order, for the Window menu) and the active one's revision,
/// selection, recent files, panels and view state, the language. `input` forces a change on
/// frames with a click or key press, which covers state the hash doesn't list.
pub(crate) fn state_hash(app: &PhotocraftApp, input: Option<u64>) -> u64 {
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    let s = &app.session;
    s.journal.len().hash(&mut h);
    s.active_index().hash(&mut h);
    for d in s.documents() {
        (d.doc.id.0, &d.doc.name).hash(&mut h);
    }
    s.clipboard.is_some().hash(&mut h);
    // A running job greys the commands that would change its document.
    s.has_jobs().hash(&mut h);
    if let Some(d) = s.active() {
        (d.doc.id.0, d.revision, d.active_layer.map(|l| l.0), d.selected_layers.len(), d.isolated_layers.len()).hash(&mut h);
    }
    let ui = &app.ui;
    (&ui.recent_files, &ui.workspace, ui.palette_open, ui.transform.is_some(), ui.text_edit.is_some(), ui.dialogs.len()).hash(&mut h);
    (app.discard.is_some(), app.camera_raw.is_some()).hash(&mut h);
    serde_json::to_string(&(&ui.panels, &ui.extras, &ui.view, ui.theme, ui.tool)).unwrap_or_default().hash(&mut h);
    s.prefs().interface.language.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn app(doc: bool) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        if doc {
            app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
            app.sync_views();
        }
        app
    }

    fn layout(doc: bool) -> Layout {
        photocraft_layout(&crate::menus::menu_items(&app(doc)), Lang::EN, "auto")
    }

    fn ids(nodes: &[Node]) -> Vec<String> {
        nodes
            .iter()
            .map(|n| match n {
                Node::Item(it) => it.id.clone(),
                Node::Separator => "---".into(),
                Node::Submenu { label, .. } => format!("[{label}]"),
                Node::Standard(s) => format!("<{s:?}>"),
            })
            .collect()
    }

    #[test]
    fn photocraft_gets_a_mac_app_menu() {
        let l = layout(true);
        let app = &l.bar.menus[0];
        assert_eq!((app.title.as_str(), app.role), ("PhotoCraft", MenuRole::App));
        assert_eq!(
            ids(&app.children),
            [
                "help.about",
                "---",
                "[Settings]",
                "[Language]",
                "[Appearance]",
                "---",
                "<Services>",
                "---",
                HIDE,
                "<HideOthers>",
                "<ShowAll>",
                "---",
                "file.exit"
            ]
        );
        let quit = l.bar.find("file.exit").unwrap();
        assert_eq!((quit.label.as_str(), quit.shortcut.as_deref()), ("Quit PhotoCraft", Some("Cmd+Q")));
        assert_eq!(l.bar.find("help.about").unwrap().label, "About PhotoCraft");
    }

    #[test]
    fn moved_items_leave_their_old_menus() {
        let l = layout(true);
        let by = |t: &str| l.bar.menus.iter().find(|m| m.title == t).unwrap();
        assert!(!ids(&by("File").children).contains(&"file.exit".to_string()));
        assert!(!ids(&by("Help").children).contains(&"help.about".to_string()));
        assert!(!ids(&by("Edit").children).contains(&"[Preferences]".to_string()));
        let titles: Vec<&str> = l.bar.menus.iter().map(|m| m.title.as_str()).collect();
        assert_eq!(titles, ["PhotoCraft", "File", "Edit", "Image", "Layer", "Type", "Select", "Filter", "View", "Window", "Help"]);
    }

    /// PhotoCraft keeps Photoshop's ⌘H, ⌘M, ⌘W and ⌘, (Hide Layers): the system items give way.
    #[test]
    fn clashes_are_resolved_in_photocrafts_favour() {
        let l = layout(true);
        let clash = |sc: &str| l.clashes.iter().find(|c| c.shortcut == sc).map(|c| c.command.as_str());
        assert_eq!(clash("Cmd+H"), Some("view.extras"));
        assert_eq!(clash("Cmd+M"), Some("image.adjustments.curves"));
        assert_eq!(clash("Cmd+W"), Some("file.close"));
        assert_eq!(clash("Cmd+,"), Some("layer.hideLayers"));
        assert_eq!(l.bar.find(HIDE).unwrap().shortcut.as_deref(), Some("Ctrl+Cmd+H"), "as in Photoshop");
        assert_eq!(l.bar.find(MINIMIZE).unwrap().shortcut.as_deref(), Some("Ctrl+Cmd+M"));
        let window = l.bar.menu(MenuRole::Window).unwrap();
        assert_eq!(ids(&window.children)[..3], [MINIMIZE.to_string(), "<Zoom>".into(), "---".into()]);
        assert!(ids(&window.children).ends_with(&["<BringAllToFront>".into(), "---".into(), "window.document.0".into()]));
        let file = l.bar.menus.iter().find(|m| m.title == "File").unwrap();
        assert!(!file.children.contains(&Node::Standard(Standard::CloseWindow)), "⌘W stays File › Close");
    }

    /// Themes and the appearance-mode cycle move from Window to the app menu's Appearance.
    #[test]
    fn themes_move_to_the_app_menus_appearance() {
        let mut a = app(false);
        a.ui.theme = crate::theme::ThemeKind::Classic;
        let l = photocraft_layout(&crate::menus::menu_items(&a), Lang::EN, "auto");
        let app_menu = l.bar.menu(MenuRole::App).unwrap();
        let Some(Node::Submenu { children, .. }) = app_menu.children.iter().find(|n| matches!(n, Node::Submenu { label, .. } if label == "Appearance")) else {
            panic!("an Appearance submenu");
        };
        let themes = ids(children);
        assert_eq!(themes.last().map(String::as_str), Some("window.theme.toggle"));
        assert!(themes.contains(&"window.theme.pro".to_string()) && themes.contains(&"window.theme.classic".to_string()));
        assert_eq!(l.bar.find("window.theme.classic").unwrap().checked, Some(true));
        assert_eq!(l.bar.find("window.theme.pro").unwrap().checked, Some(false));
        assert_eq!(l.bar.find("window.theme.toggle").unwrap().label, "Next Appearance Mode");
        assert_eq!(l.bar.find("window.theme.system").unwrap().label, "Sync with system");
        assert_eq!(l.bar.find("window.theme.system").unwrap().checked, Some(false));
        let window = ids(&l.bar.menu(MenuRole::Window).unwrap().children);
        assert!(!window.iter().any(|id| id.starts_with("window.theme.") || id == "[Theme]"), "{window:?}");
        a.session.prefs.edit(|p| p.interface.appearance_mode = photocraft_engine::prefs::AppearanceMode::Auto);
        for palette in [crate::theme::ThemeKind::Classic, crate::theme::ThemeKind::ProMedium] {
            a.ui.theme = palette;
            let automatic = photocraft_layout(&crate::menus::menu_items(&a), Lang::EN, "auto");
            assert_eq!(automatic.bar.find("window.theme.system").unwrap().checked, Some(true));
            for kind in crate::theme::ThemeKind::ALL {
                assert_eq!(automatic.bar.find(&format!("window.theme.{}", kind.id())).unwrap().checked, Some(false));
            }
        }
    }

    /// Language follows Settings in the app menu: Auto and every UI language in its own name,
    /// the preference checked; choosing one sets `interface.language`.
    #[test]
    fn the_app_menu_switches_the_ui_language() {
        let l = photocraft_layout(&crate::menus::menu_items(&app(false)), Lang::EN, "ja");
        let app_menu = l.bar.menu(MenuRole::App).unwrap();
        let Some(Node::Submenu { children, .. }) = app_menu.children.iter().find(|n| matches!(n, Node::Submenu { label, .. } if label == "Language")) else {
            panic!("a Language submenu");
        };
        assert_eq!(children.iter().filter(|n| matches!(n, Node::Item(_))).count(), Lang::all().count() + 1);
        assert_eq!(l.bar.find("app.language.ja").map(|it| (it.label.as_str(), it.checked)), Some(("日本語", Some(true))));
        assert_eq!(l.bar.find("app.language.auto").and_then(|it| it.checked), Some(false));
        // Native names stay native in another UI language.
        let de = photocraft_layout(&crate::menus::menu_items(&app(false)), Lang::from_pref("de"), "de");
        assert_eq!(de.bar.find("app.language.ja").map(|it| it.label.as_str()), Some("日本語"));
        let mut a = app(false);
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut a, &ctx, "app.language.fr", json!({})).unwrap();
        assert_eq!(a.session.prefs().interface.language, "fr");
        crate::menus::invoke(&mut a, &ctx, "app.language.auto", json!({})).unwrap();
        assert_eq!(a.session.prefs().interface.language, "auto");
        assert!(crate::menus::invoke(&mut a, &ctx, "app.language.xx-nope", json!({})).is_err(), "an unknown code is an error");
    }

    /// The menus show the shortcut that runs each command (Undo ⌘Z, Copy ⌘C were blank).
    #[test]
    fn menu_rows_show_the_shortcut_that_runs_them() {
        let l = layout(true);
        let sc = |id: &str| l.bar.find(id).and_then(|it| it.shortcut.clone());
        assert_eq!(sc("edit.undo").as_deref(), Some("Cmd+Z"));
        assert_eq!(sc("edit.copy").as_deref(), Some("Cmd+C"));
        assert_eq!(sc("edit.paste").as_deref(), Some("Cmd+V"));
        assert_eq!(sc("layer.hideLayers").as_deref(), Some("Cmd+,"));
        let app = app(true);
        let bindings = crate::shortcut_dispatch::bindings(&app);
        for it in crate::menus::menu_items(&app).iter().filter(|it| it.enabled) {
            let Some(shown) = it.shortcut.as_deref().and_then(crate::shortcuts::parse) else { continue };
            let runs = bindings.iter().find(|(_, b)| *b == shown).map(|(id, _)| id.as_str());
            assert!(runs.is_some(), "{} shows {:?}, which runs nothing", it.id, it.shortcut);
        }
    }

    /// A native menu can't turn a plain item into a check item in place: opening the first
    /// document must not change any item's kind, or the whole menu is rebuilt. Window's list of
    /// documents gains a row, which rebuilds only the Window menu.
    #[test]
    fn opening_a_document_changes_state_not_structure() {
        let (a, b) = (crate::menus::menu_items(&app(false)), crate::menus::menu_items(&app(true)));
        let kind = |items: &[MenuItem]| items.iter().map(|i| (i.id.clone(), i.checked.is_some())).collect::<HashMap<_, _>>();
        let (ka, kb) = (kind(&a), kind(&b));
        let flips: Vec<&String> = ka.iter().filter(|(id, k)| kb.get(*id).is_some_and(|o| o != *k)).map(|(id, _)| id).collect();
        assert!(flips.is_empty(), "items that change kind: {flips:?}");
        let (no_doc, one_doc) = (layout(false).bar, layout(true).bar);
        assert_eq!(structure_key(&no_doc), structure_key(&one_doc));
        assert_ne!(window_key(&no_doc), window_key(&one_doc), "the Window menu is rebuilt for its new row");
    }

    /// #2991: the open documents follow Bring All to Front, where a Mac app lists its windows,
    /// the active one checked. Switching documents moves the check without a rebuild.
    #[test]
    fn the_documents_follow_bring_all_to_front() {
        let none = ids(&layout(false).bar.menu(MenuRole::Window).unwrap().children);
        assert_eq!(none.last().map(String::as_str), Some("<BringAllToFront>"));

        let mut a = app(true);
        a.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        a.sync_views();
        let bar = photocraft_layout(&crate::menus::menu_items(&a), Lang::EN, "auto").bar;
        let window = ids(&bar.menu(MenuRole::Window).unwrap().children);
        assert_eq!(window[window.len() - 4..], ["<BringAllToFront>", "---", "window.document.0", "window.document.1"]);
        assert_eq!(bar.find("window.document.1").and_then(|i| i.checked), Some(true));
        assert_eq!(bar.find("window.document.0").and_then(|i| i.checked), Some(false));

        a.session.set_active(0);
        let switched = photocraft_layout(&crate::menus::menu_items(&a), Lang::EN, "auto").bar;
        assert_eq!(switched.find("window.document.0").and_then(|i| i.checked), Some(true));
        assert_eq!((structure_key(&switched), window_key(&switched)), (structure_key(&bar), window_key(&bar)), "a check moves in place");
    }

    #[test]
    fn native_key_equivalents_are_command_and_function_keys_only() {
        assert!(Chord::parse("Cmd+B").unwrap().native_ok());
        assert!(Chord::parse("F5").unwrap().native_ok());
        assert!(!Chord::parse("B").unwrap().native_ok());
        assert!(!Chord::parse("Alt+F").unwrap().native_ok());
        assert!(!Chord::parse("Shift+X").unwrap().native_ok());
        assert_eq!(Chord::parse("Ctrl+Cmd+M").unwrap().portable(), "Ctrl+Cmd+M");
        assert_eq!(Chord::parse("Cmd++").unwrap().key, "+");
    }

    /// Every menu shortcut is written so it parses and turns back into the key press egui gets.
    #[test]
    fn every_shortcut_becomes_a_key_press() {
        for it in layout(true).bar.items() {
            let Some(sc) = &it.shortcut else { continue };
            let c = Chord::parse(sc).unwrap_or_else(|| panic!("{sc} on {} doesn't parse", it.id));
            assert_eq!(&c.portable(), sc, "{} writes its modifiers in another order", it.id);
            let Some(egui::Event::Key { key, modifiers, .. }) = c.key_event(true) else { panic!("{sc} on {} has no egui key", it.id) };
            let parsed = crate::shortcuts::parse(sc).unwrap();
            assert!(crate::shortcuts::key_matches(&parsed, key, modifiers), "{sc} on {} doesn't match its own key press", it.id);
        }
    }

    fn frame(ctx: &egui::Context, raw: egui::RawInput, f: impl FnMut(&mut egui::Ui)) {
        let mut out = ctx.run_ui(raw, f);
        out.textures_delta.clear();
    }

    struct Fake {
        synced: std::rc::Rc<std::cell::RefCell<Vec<MenuBar>>>,
        events: Vec<Event>,
    }

    impl Backend for Fake {
        fn sync(&mut self, bar: &MenuBar) {
            self.synced.borrow_mut().push(bar.clone());
        }
        fn drain(&mut self) -> Vec<Event> {
            std::mem::take(&mut self.events)
        }
    }

    /// A key equivalent reaches PhotoCraft's own key handling as the key press it was, so the
    /// shortcut runs once, through the same rules as without a native menu.
    #[test]
    fn key_equivalents_run_through_the_normal_shortcut_path() {
        let synced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut app = app(true);
        let ctx = egui::Context::default();
        let rulers = Chord::parse("Cmd+R").unwrap();
        app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: synced.clone(), events: vec![Event::Key(rulers)] })));
        let before = app.ui.extras.rulers;
        let mut raw = egui::RawInput::default();
        if let Some(m) = app.services.native_menu.as_mut() {
            m.raw_input(&mut raw);
        }
        assert_eq!(raw.events.len(), 1, "the press: winit delivers the release");
        frame(&ctx, raw, |ui| {
            run(&mut app, ui.ctx());
            crate::shortcuts::handle(&mut app, ui.ctx());
            sync(&mut app, ui.ctx());
        });
        assert_eq!(app.ui.extras.rulers, !before, "⌘R toggled View › Rulers once");
        assert_eq!(synced.borrow().len(), 1, "the menu was built");
        // Nothing changed: no rebuild of the rows on the next frame.
        frame(&ctx, egui::RawInput::default(), |ui| sync(&mut app, ui.ctx()));
        assert_eq!(synced.borrow().len(), 1);
    }

    /// #1638: ⌘V through the Mac menu pasted three layers. AppKit takes the key-down for Edit ›
    /// Paste's key equivalent, but winit still delivers the key-up (it forwards key-ups while ⌘ is
    /// held). The menu's press plus that real release must paste once, whether the release
    /// arrives in a later frame or already sits in the same input as the menu event.
    #[test]
    fn a_paste_key_equivalent_pastes_once() {
        let release = || egui::Event::Key { key: Key::V, physical_key: Some(Key::V), pressed: false, repeat: false, modifiers: Modifiers::COMMAND };
        for same_frame in [false, true] {
            let mut app = app(true);
            let ctx = egui::Context::default();
            app.run("select.rect", json!({"x": 0, "y": 0, "width": 8, "height": 4})).unwrap();
            app.run("edit.copy", json!({})).unwrap();
            let layers = |app: &PhotocraftApp| app.session.active().unwrap().doc.layer_count();
            let before = layers(&app);
            let paste = Chord::parse("Cmd+V").unwrap();
            app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: Default::default(), events: vec![Event::Key(paste)] })));
            let step = |app: &mut PhotocraftApp, events: Vec<egui::Event>| {
                let mut raw = egui::RawInput { events, ..Default::default() };
                eframe::App::raw_input_hook(app, &ctx, &mut raw);
                frame(&ctx, raw, |ui| {
                    run(app, ui.ctx());
                    crate::shortcuts::handle(app, ui.ctx());
                });
            };
            if same_frame {
                // A quick tap: the real key-up is already queued when the menu event is drained.
                step(&mut app, vec![egui::Event::ModifiersChanged(Modifiers::COMMAND), release()]);
            } else {
                step(&mut app, vec![egui::Event::ModifiersChanged(Modifiers::COMMAND)]);
                step(&mut app, vec![release()]);
            }
            step(&mut app, vec![egui::Event::ModifiersChanged(Modifiers::NONE)]);
            assert_eq!(layers(&app), before + 1, "one ⌘V pastes one layer (key-up in the same frame: {same_frame})");
        }
    }

    #[test]
    fn clicks_run_the_command_and_resync() {
        let synced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut app = app(true);
        let ctx = egui::Context::default();
        app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: synced.clone(), events: vec![Event::Click("window.panel.layers".into())] })));
        let before = app.ui.panels.layers;
        let mut raw = egui::RawInput::default();
        if let Some(m) = app.services.native_menu.as_mut() {
            m.raw_input(&mut raw);
        }
        frame(&ctx, raw, |ui| {
            run(&mut app, ui.ctx());
            sync(&mut app, ui.ctx());
        });
        assert_eq!(app.ui.panels.layers, !before);
        let bar = synced.borrow().last().cloned().unwrap();
        assert_eq!(bar.find("window.panel.layers").unwrap().checked, Some(!before));
    }

    /// #1354: a dialog is modal for the menu bar too. Its rows are disabled (view navigation
    /// aside), and a click that arrives anyway doesn't run under the dialog.
    #[test]
    fn an_open_dialog_blocks_menu_clicks_except_view_navigation() {
        let synced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut app = app(true);
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "image.imageSize", json!({})).unwrap();
        assert!(!app.ui.dialogs.is_empty(), "Image Size opened its dialog");
        let size = |app: &PhotocraftApp| app.session.active().map(|d| (d.doc.size.width, d.doc.size.height));
        let zoom = app.ui.views[0].zoom;
        let events = vec![Event::Click("image.imageRotation.90cw".into()), Event::Click("view.zoomIn".into())];
        app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: synced.clone(), events })));
        let mut raw = egui::RawInput::default();
        if let Some(m) = app.services.native_menu.as_mut() {
            m.raw_input(&mut raw);
        }
        frame(&ctx, raw, |ui| {
            run(&mut app, ui.ctx());
            sync(&mut app, ui.ctx());
        });
        assert_eq!(size(&app), Some((64, 48)), "the rotation didn't run under the dialog");
        assert!(app.ui.views[0].zoom > zoom, "Zoom In still works under a dialog");
        let bar = synced.borrow().last().cloned().unwrap();
        assert!(!bar.find("image.imageRotation.90cw").unwrap().enabled);
        assert!(bar.find("view.zoomIn").unwrap().enabled);
        // With the dialog closed the rows come back and a click runs.
        app.ui.dialogs.clear();
        frame(&ctx, egui::RawInput::default(), |ui| sync(&mut app, ui.ctx()));
        assert!(synced.borrow().last().unwrap().find("image.imageRotation.90cw").unwrap().enabled);
    }

    /// Edit > Cut uses the type editor's selection, like the clipboard Cut event (#1355).
    #[test]
    fn cut_selected_type_from_native_menu_matches_keyboard_cut() {
        for native_click in [true, false] {
            let synced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let mut app = app(true);
            let ctx = egui::Context::default();
            let id = photocraft_doc::LayerId(app.run("type.create", json!({"text": "Hello", "size": 12, "x": 4, "y": 24})).unwrap()["layer"].as_u64().unwrap());
            crate::menus::invoke(&mut app, &ctx, "type.editText", json!({})).unwrap();
            let edit = app.ui.text_edit.as_ref().unwrap();
            assert_eq!((edit.anchor, edit.caret), (0, 5));
            let events = if native_click { vec![Event::Click("edit.cut".into())] } else { Vec::new() };
            app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: synced.clone(), events })));
            frame(&ctx, egui::RawInput::default(), |ui| sync(&mut app, ui.ctx()));
            assert!(synced.borrow().last().unwrap().find("edit.cut").unwrap().enabled, "Cut must be enabled for selected characters");

            let mut raw = egui::RawInput::default();
            app.services.native_menu.as_mut().unwrap().raw_input(&mut raw);
            if !native_click {
                raw.events.push(egui::Event::Cut);
            }
            let mut out = ctx.run_ui(raw, |ui| {
                run(&mut app, ui.ctx());
                crate::shortcuts::handle(&mut app, ui.ctx());
                sync(&mut app, ui.ctx());
            });
            out.textures_delta.clear();
            assert_eq!(out.platform_output.commands, [egui::OutputCommand::CopyText("Hello".into())]);
            let photocraft_doc::LayerContent::Text(text) = &app.session.active().unwrap().doc.layer(id).unwrap().content else {
                panic!("type layer changed kind")
            };
            assert!(text.text.is_empty());
            let edit = app.ui.text_edit.as_ref().unwrap();
            assert_eq!((edit.anchor, edit.caret), (0, 0));
            assert!(!synced.borrow().last().unwrap().find("edit.cut").unwrap().enabled, "Cut must be disabled at an empty caret");
            assert!(app.session.clipboard.is_none(), "text Cut must not copy rendered pixels");
        }
    }

    /// Select > All, Edit > Copy and Edit > Paste work on the text being typed, not on pixels (#2893).
    #[test]
    fn copy_and_paste_type_from_native_menu() {
        let synced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut app = app(true);
        let ctx = egui::Context::default();
        let id = photocraft_doc::LayerId(app.run("type.create", json!({"text": "Hello", "size": 12, "x": 4, "y": 24})).unwrap()["layer"].as_u64().unwrap());
        crate::menus::invoke(&mut app, &ctx, "type.editText", json!({})).unwrap();
        if let Some(e) = app.ui.text_edit.as_mut() {
            (e.anchor, e.caret) = (5, 5);
        }
        let layers = app.session.active().unwrap().doc.layers.len();
        let click = |app: &mut PhotocraftApp, id: &str| {
            app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: synced.clone(), events: vec![Event::Click(id.into())] })));
            let mut raw = egui::RawInput::default();
            app.services.native_menu.as_mut().unwrap().raw_input(&mut raw);
            let mut out = ctx.run_ui(raw, |ui| {
                run(app, ui.ctx());
                sync(app, ui.ctx());
            });
            out.textures_delta.clear();
            out
        };
        let text = |app: &PhotocraftApp| match &app.session.active().unwrap().doc.layer(id).unwrap().content {
            photocraft_doc::LayerContent::Text(t) => t.text.clone(),
            _ => panic!("type layer changed kind"),
        };

        click(&mut app, "select.all");
        let edit = app.ui.text_edit.as_ref().unwrap();
        assert_eq!((edit.anchor, edit.caret), (0, 5), "Select > All selects the characters");
        let bar = synced.borrow().last().cloned().unwrap();
        assert!(bar.find("edit.copy").unwrap().enabled, "Copy is enabled for selected characters");
        assert!(bar.find("edit.paste").unwrap().enabled, "Paste is enabled while typing");

        let out = click(&mut app, "edit.copy");
        assert_eq!(out.platform_output.commands, [egui::OutputCommand::CopyText("Hello".into())]);
        assert!(app.session.clipboard.is_none(), "text Copy must not copy rendered pixels");

        let out = click(&mut app, "edit.paste");
        let asked = out.viewport_output.values().any(|v| v.commands.contains(&egui::ViewportCommand::RequestPaste));
        assert!(asked, "Edit > Paste asks the window for the clipboard text");
        // The text arrives as a Paste event next frame and replaces the selection.
        let raw = egui::RawInput { events: vec![egui::Event::Paste("World".into())], ..Default::default() };
        frame(&ctx, raw, |ui| crate::shortcuts::handle(&mut app, ui.ctx()));
        assert_eq!(text(&app), "World");
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers, "no pixel layer was pasted");
        assert!(!app.ui.status_error, "{}", app.ui.status);
    }

    #[test]
    fn native_cut_without_type_edit_keeps_pixel_clipboard_behavior() {
        let synced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut app = app(false);
        app.run("file.new", json!({"width": 4, "height": 4, "background": "transparent"})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        assert_eq!(app.run("document.pixel", json!({"x": 1, "y": 1})).unwrap(), json!([1.0, 0.0, 0.0, 1.0]));
        let ctx = egui::Context::default();
        app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: synced.clone(), events: vec![Event::Click("edit.cut".into())] })));
        frame(&ctx, egui::RawInput::default(), |ui| sync(&mut app, ui.ctx()));
        assert_eq!(synced.borrow().last().unwrap().find("edit.cut").unwrap().enabled, app.session.is_enabled("edit.cut"));
        assert!(app.session.is_enabled("edit.cut"));
        let mut raw = egui::RawInput::default();
        app.services.native_menu.as_mut().unwrap().raw_input(&mut raw);
        frame(&ctx, raw, |ui| {
            run(&mut app, ui.ctx());
            sync(&mut app, ui.ctx());
        });
        assert_eq!(app.run("document.pixel", json!({"x": 1, "y": 1})).unwrap(), json!([0.0, 0.0, 0.0, 0.0]));
        let clip = app.session.clipboard.as_ref().unwrap();
        assert_eq!((clip.bounds.width(), clip.bounds.height()), (4, 4));
        let mut px = vec![[0u8; 4]; 16];
        clip.surface.read_rgba8_into(clip.bounds, &mut px);
        assert_eq!(px, vec![[255u8, 0, 0, 255]; 16]);
        assert!(app.ui.text_edit.is_none());
    }

    /// With the Mac menu bar the title bar draws no menu titles: each title shows once fewer.
    #[test]
    fn the_title_bar_hides_its_menus_with_the_mac_menu_bar() {
        use egui_kittest::{Harness, kittest::Queryable};
        let counts = |native: bool| -> Vec<usize> {
            let mut h = Harness::builder().with_size(egui::vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
                PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                let mut app = app(true);
                if native {
                    app.services.native_menu = Some(NativeMenu::new(Box::new(Fake { synced: Default::default(), events: Vec::new() })));
                }
                app
            });
            h.run_steps(4);
            crate::menus::TOP_MENUS.iter().map(|t| h.query_all_by_label(t).count()).collect()
        };
        let (in_window, native) = (counts(false), counts(true));
        for (i, title) in crate::menus::TOP_MENUS.iter().enumerate() {
            assert_eq!(native[i] + 1, in_window[i], "{title}: {} in-window, {} with the Mac menu bar", in_window[i], native[i]);
        }
    }
}
