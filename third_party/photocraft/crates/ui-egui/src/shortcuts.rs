//! Keyboard shortcuts: Photoshop defaults from the command registry, plus single-key tools.

use egui::{Key, KeyboardShortcut, Modifiers};
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// Parse `Cmd+Shift+N` style strings. `Cmd` maps to ⌘ on macOS and Ctrl elsewhere.
pub fn parse(s: &str) -> Option<KeyboardShortcut> {
    // A trailing '+' names the Plus key (both "Cmd+" and "Cmd++" are accepted).
    // Split the key off first: otherwise an unrecognised component could be ignored
    // when a later component overwrites it with a valid key.
    let (prefix, key_name) = if let Some(without_plus) = s.strip_suffix('+') {
        (without_plus.strip_suffix('+').unwrap_or(without_plus), "+")
    } else {
        s.rsplit_once('+').unwrap_or(("", s))
    };
    let mut mods = Modifiers::NONE;
    if !prefix.is_empty() {
        for part in prefix.split('+') {
            match part {
                "Cmd" if !mods.command => mods |= Modifiers::COMMAND,
                "Shift" if !mods.shift => mods |= Modifiers::SHIFT,
                "Alt" if !mods.alt => mods |= Modifiers::ALT,
                "Ctrl" if !mods.ctrl => mods |= Modifiers::CTRL,
                // Unknown keys, duplicate modifiers, and extra components must not
                // silently turn a malformed binding into a different shortcut.
                _ => return None,
            }
        }
    }
    let key = match key_name {
        "+" => Some(Key::Plus),
        "=" => Some(Key::Equals),
        "-" => Some(Key::Minus),
        "[" => Some(Key::OpenBracket),
        ";" => Some(Key::Semicolon),
        "'" => Some(Key::Quote),
        "]" => Some(Key::CloseBracket),
        name => Key::from_name(name),
    }?;
    Some(KeyboardShortcut::new(mods, key))
}

/// Human-readable form for menus, tooltips and inline hints (`Cmd+Shift+N` → `⌘⇧N` on
/// macOS, `Ctrl+Shift+N` elsewhere). Works for lone keys too: `pretty("Alt")` is `⌥`/`Alt`,
/// `pretty("Enter")` is `↩`/`Enter`.
pub fn pretty(s: &str) -> String {
    let mac = cfg!(target_os = "macos");
    let mut parts: Vec<&str> = s.split('+').collect();
    if s.ends_with('+') {
        // A final '+' is the Plus key, not an empty component. This matters on
        // macOS, where the modifier separator is hidden ("Cmd++" is "⌘+").
        parts.pop();
        if parts.last().copied() == Some("") {
            parts.pop();
        }
        parts.push("+");
    }
    parts
        .into_iter()
        .map(|p| match (p, mac) {
            ("Cmd", true) => "⌘".to_string(),
            ("Cmd", false) => "Ctrl".to_string(),
            ("Shift", true) => "⇧".to_string(),
            ("Alt", true) => "⌥".to_string(),
            ("Enter", true) => "↩".to_string(),
            ("Delete", true) => "⌫".to_string(),
            (other, _) => other.to_string(),
        })
        .collect::<Vec<_>>()
        .join(if mac { "" } else { "+" })
}

/// The command's effective shortcut in the platform's notation (`Ctrl+N`), or `None` when
/// the command has no binding (the user removed it in Edit › Keyboard Shortcuts).
pub fn shortcut_label(app: &PhotocraftApp, command: &str) -> Option<String> {
    effective_shortcut(app, command, default_shortcut(command).as_deref()).map(|s| pretty(&s))
}

/// `label` (English, shown in the UI language) followed by the command's effective shortcut
/// (`New document…     Ctrl+N`); bare `label` when the binding was removed.
pub fn command_label(app: &PhotocraftApp, label: &str, command: &str) -> String {
    let label = tl!(label);
    shortcut_label(app, command).map_or_else(|| label.to_string(), |sc| format!("{label}     {sc}"))
}

/// `label` (English, shown in the UI language) with the command's effective shortcut in
/// parentheses (`Rulers  (Ctrl+R)`); bare `label` when the binding was removed.
pub fn tip_label(app: &PhotocraftApp, label: &str, command: &str) -> String {
    let label = tl!(label);
    shortcut_label(app, command).map_or_else(|| label.to_string(), |sc| format!("{label}  ({sc})"))
}

/// The shortcut in effect for a command: the user's override, else `default`.
pub fn effective_shortcut(app: &PhotocraftApp, id: &str, default: Option<&str>) -> Option<String> {
    app.session.prefs().shortcut(id, default).map(str::to_string)
}

/// Default (unmodified) shortcut of a command id.
pub fn default_shortcut(id: &str) -> Option<String> {
    crate::menus::UI_COMMANDS
        .iter()
        .find(|c| c.0 == id)
        .and_then(|c| c.3)
        .or_else(|| photocraft_engine::commands::find(id).and_then(|c| c.shortcut))
        .or_else(|| crate::menu_catalog::CATALOG.iter().find(|c| c.3 == id).and_then(|c| c.2))
        .or_else(|| photocraft_engine::prefs::TEMPORARY_TOOLS.iter().find(|t| t.0 == id).map(|t| t.2))
        .map(str::to_string)
        .or_else(|| crate::tool_shortcuts::tool(id).and_then(crate::tool_shortcuts::default))
}

/// With ⇧ held the OS reports the shifted character as the logical key (US layout): ⇧; is `:`.
fn shifted(k: Key) -> Option<Key> {
    Some(match k {
        Key::Equals => Key::Plus,
        Key::Semicolon => Key::Colon,
        Key::OpenBracket => Key::OpenCurlyBracket,
        Key::CloseBracket => Key::CloseCurlyBracket,
        Key::Slash => Key::Questionmark,
        Key::Backslash => Key::Pipe,
        Key::Num1 => Key::Exclamationmark,
        _ => return None,
    })
}

/// Does a key press match a shortcut, the way Photoshop matches them?
///
/// - Modifiers match exactly (⌘⌥I is not ⌘I), with ⌘ = Ctrl off the Mac.
/// - `=` also matches `+` (⌘+ with ⇧ on a US layout, the numpad `+`, the `+` key of Nordic and
///   German layouts): Photoshop zooms in on both.
/// - Delete also matches Backspace (the Mac's "delete" key), and Backspace matches Delete (⌥⌫
///   and ⌥Delete both fill on Windows).
/// - A ⇧ shortcut on a punctuation key matches the shifted character (⌘⇧; arrives as ⌘`:`).
pub fn key_matches(sc: &KeyboardShortcut, key: Key, mods: Modifiers) -> bool {
    let k = sc.logical_key;
    let direct = key == k || (k == Key::Delete && key == Key::Backspace) || (k == Key::Backspace && key == Key::Delete);
    let plus = k == Key::Equals && key == Key::Plus;
    let via_shift = sc.modifiers.shift && shifted(k) == Some(key);
    if !(direct || plus || via_shift) {
        return false;
    }
    // Typing `+` (or another shifted symbol) may need ⇧ on this layout: don't require it absent.
    let shifted_symbol =
        matches!(key, Key::Plus | Key::Colon | Key::OpenCurlyBracket | Key::CloseCurlyBracket | Key::Questionmark | Key::Pipe | Key::Exclamationmark);
    let shift_ok = mods.shift == sc.modifiers.shift || (shifted_symbol && !sc.modifiers.shift);
    mods.alt == sc.modifiers.alt && shift_ok && mods.cmd_ctrl_matches(sc.modifiers)
}

/// Consume a press of `sc` (see [`key_matches`]); true when one was found.
pub fn consume(ctx: &egui::Context, sc: &KeyboardShortcut) -> bool {
    ctx.input_mut(|i| {
        let before = i.events.len();
        i.events.retain(|e| !matches!(e, egui::Event::Key { key, pressed: true, modifiers, .. } if key_matches(sc, *key, *modifiers)));
        i.events.len() != before
    })
}

/// View navigation that Photoshop keeps live while a dialog is open (so the image can be judged
/// at any zoom): Zoom In / Out, Fit on Screen, 100%.
pub const NAV_COMMANDS: [&str; 4] = ["view.zoomIn", "view.zoomOut", "view.fitOnScreen", "view.actualPixels"];

/// ⌘+ / ⌘- / ⌘0 / ⌘1 (or their Edit › Keyboard Shortcuts overrides). Returns true when one ran.
/// Layer › New › Layer's shortcut with ⌥ added (⌥⇧⌘N by default) makes the layer at once, as in
/// Photoshop: the plain shortcut asks with the New Layer dialog first, like the menu item, and
/// the Layers panel button's click and ⌥-click are the same pair the other way round (#4794).
/// A command bound to that key itself keeps it.
fn new_layer_without_dialog(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    const NEW_LAYER: &str = crate::new_layer_ui::COMMAND;
    let Some(mut sc) = effective_shortcut(app, NEW_LAYER, default_shortcut(NEW_LAYER).as_deref()).and_then(|s| parse(&s)) else {
        return false;
    };
    if sc.modifiers.alt {
        return false;
    }
    sc.modifiers.alt = true;
    let pressed =
        ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Key { key, pressed: true, modifiers, .. } if key_matches(&sc, *key, *modifiers))));
    if !pressed || crate::shortcut_dispatch::bindings(app).iter().any(|(_, b)| *b == sc) || !consume(ctx, &sc) {
        return false;
    }
    if let Err(e) = app.run(NEW_LAYER, json!({})) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    true
}

fn nav_keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    for id in NAV_COMMANDS {
        let sc = crate::menus::UI_COMMANDS.iter().find(|c| c.0 == id).and_then(|c| parse(app.session.prefs().shortcut(id, c.3)?));
        if let Some(sc) = sc
            && consume(ctx, &sc)
        {
            if crate::menus::is_enabled(app, id) {
                let _ = crate::menus::invoke(app, ctx, id, json!({}));
            }
            return true;
        }
    }
    false
}

/// The key press a clipboard event stands for outside text fields, with `mods` held: Copy, Cut
/// and Paste are ⌘C, ⌘X and ⌘V (with ⇧/⌥ as held). On Windows egui-winit also sends Cut for
/// Shift+Delete, which is Edit › Fill here (#530): a Cut with ⇧ and without Ctrl is that key.
fn clipboard_press(e: &egui::Event, mods: Modifiers, is_windows: bool) -> Option<(Key, Modifiers)> {
    let held = if mods.command { mods } else { Modifiers::COMMAND };
    match e {
        egui::Event::Copy => Some((Key::C, held)),
        egui::Event::Cut if is_windows && mods.shift && !mods.command => Some((Key::Delete, mods)),
        egui::Event::Cut => Some((Key::X, held)),
        egui::Event::Paste(_) => Some((Key::V, held)),
        _ => None,
    }
}

/// egui-winit turns ⌘C / ⌘X / ⌘V into `Event::Copy` / `Cut` / `Paste` and drops the key press, and
/// drops ⌘V entirely when the clipboard holds no text (an image). Outside text fields, give the
/// pixel commands their key presses back: Copy/Cut/Paste become ⌘C/⌘X/⌘V presses (with ⇧/⌥ as
/// held, so ⇧⌘C is Copy Merged), and a ⌘V release without a ⌘V press (a paste event, or the Mac
/// menu's key equivalent) becomes a ⌘V press.
///
/// The web build gets both: eframe forwards the ⌘V key-down, then the browser's `paste` event
/// (the key-down's default action) arrives as `Event::Paste` in the same frame. A clipboard event
/// right after its own ⌘ key press is that press, not a second one (#2638).
///
/// Ctrl+Insert copies and Shift+Insert pastes: egui-winit maps them on Windows (a Shift+Insert
/// release without a paste event pastes the image, like ⌘V), and we map them here on Linux.
pub fn clipboard_keys(ctx: &egui::Context, typing: bool, raw: &mut egui::RawInput) {
    let press = |key, modifiers| egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers };
    if cfg!(target_os = "linux") {
        raw.events.retain_mut(|e| {
            let egui::Event::Key { key: Key::Insert, pressed: true, modifiers: m, .. } = *e else { return true };
            match (m.ctrl, m.shift) {
                (true, false) => *e = egui::Event::Copy,
                // Text fields need the clipboard text: egui-winit reads it and sends a Paste event.
                (false, true) if typing => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaste);
                    return false;
                }
                (false, true) => *e = press(Key::V, Modifiers::COMMAND),
                _ => {}
            }
            true
        });
    }
    if typing {
        // A text field copies and pastes on egui's Copy / Cut / Paste events, which egui-winit
        // makes from the window's key and so never sends as key presses. A ⌘C / ⌘X / ⌘V press
        // here is the Mac menu's key equivalent (AppKit takes the key while Edit › Copy, Cut or
        // Paste is enabled), which a text field ignores: pasting text worked only while
        // PhotoCraft's own clipboard was empty. Give those presses back as the events; Paste asks
        // the window for the clipboard text, which arrives as a Paste event next frame.
        raw.events.retain_mut(|e| {
            let egui::Event::Key { key, pressed: true, modifiers: m, .. } = *e else { return true };
            if !m.command || m.alt || m.shift {
                return true;
            }
            match key {
                Key::C => *e = egui::Event::Copy,
                Key::X => *e = egui::Event::Cut,
                Key::V => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaste);
                    return false;
                }
                _ => {}
            }
            true
        });
        return;
    }
    let seen = egui::Id::new("pc-paste-key-seen");
    // The modifiers held at each event: last frame's, updated as the events report changes.
    let mut mods = ctx.input(|i| i.modifiers);
    let mut out = Vec::with_capacity(raw.events.len());
    // The last ⌘C/⌘X/⌘V press this frame whose clipboard event hasn't arrived (yet).
    let mut unpaired: Option<Key> = None;
    for e in raw.events.drain(..) {
        if let egui::Event::ModifiersChanged(m) | egui::Event::Key { modifiers: m, .. } = &e {
            mods = *m;
        }
        if let Some((key, held)) = clipboard_press(&e, mods, cfg!(target_os = "windows")) {
            if key == Key::V {
                ctx.data_mut(|d| d.insert_temp(seen, true));
            }
            if unpaired.take_if(|k| *k == key).is_none() {
                out.push(press(key, held));
            }
            continue;
        }
        match &e {
            // A ⌘C/⌘X/⌘V press egui-winit didn't swallow (the web, the Mac menu's key equivalent,
            // Shift+Insert on Linux): its clipboard event and, for ⌘V, its release must not run
            // the command a second time (#2638, #1638).
            egui::Event::Key { key: key @ (Key::C | Key::X | Key::V), pressed: true, modifiers, .. } if modifiers.command => {
                unpaired = Some(*key);
                if *key == Key::V {
                    ctx.data_mut(|d| d.insert_temp(seen, true));
                }
            }
            // A paste key's release ends that press, with or without ⌘ still held (else a stale
            // mark would swallow the next image paste); without a press it is the paste itself.
            egui::Event::Key { key: key @ (Key::V | Key::Insert), pressed: false, modifiers, .. } => {
                let had_press = ctx.data_mut(|d| d.remove_temp::<bool>(seen)).unwrap_or(false);
                let held = match key {
                    Key::V if modifiers.command => Some(*modifiers),
                    Key::Insert if cfg!(target_os = "windows") && modifiers.shift && !modifiers.ctrl => Some(Modifiers::COMMAND),
                    _ => None,
                };
                if let Some(held) = held
                    && !had_press
                {
                    out.push(press(Key::V, held));
                }
            }
            _ => {}
        }
        out.push(e);
    }
    raw.events = out;
}

pub fn handle(app: &mut PhotocraftApp, ctx: &egui::Context) {
    // A held tool letter released: a temporary tool gives the previous one back (#2672).
    crate::spring_tools::tick(app, ctx);
    // However the tool was picked (toolbar, flyout, shortcut, automation), its group's key brings
    // it back later (#2608).
    app.ui.remember_group_tool();
    // Camera Raw is modal like Photoshop's filter dialog: no application shortcut (Save, Undo,
    // tools) runs beneath it, and it handles its own keys (Y, U, O, S). Unlike the other dialogs
    // below it covers the canvas, so canvas zoom keys are blocked too.
    if app.camera_raw.is_some() {
        return;
    }
    // An open menu owns the keyboard (arrows, ↩, Esc), like a native menu; so does an open
    // dropdown list, whose letters pick an option rather than a tool (#1481).
    if crate::menu_nav::is_open(ctx) || crate::widgets::dropdown_list_open(ctx) {
        return;
    }
    // ⌘Z in a filter or adjustment dialog steps the dialog's settings, not the document (#4697).
    crate::edit_history::keys(app, ctx);
    // Stroke Options owns plain navigation/activation keys, including while its child
    // dropdown is open. This runs before Popup::show registers the current pass, so use
    // the editor's explicit state; otherwise Escape deselects or commits a pending Pen path.
    // Modified commands such as Undo still follow the ordinary dispatch below.
    if app.ui.stroke_editor.is_some() && !ctx.input(|i| i.modifiers.command || i.modifiers.ctrl) {
        return;
    }
    // So does the right-click menu of text being edited: Esc closes only it, and keys typed while
    // it is open don't reach the text beneath.
    if app.ui.canvas_tool_menu.as_ref().is_some_and(|m| m.text) {
        return;
    }
    let symmetry_editing = app.ui.symmetry_transform.is_some();
    crate::symmetry_ui::track(app, ctx);
    if symmetry_editing && (app.ui.symmetry_transform.is_none() || egui::Popup::is_any_open(ctx)) {
        return;
    }
    // Before any key handler: per-layout symbol keys and the bracket fallback (#3102, #3109).
    if !ctx.text_edit_focused() {
        crate::layout_keys::apply(ctx);
    }
    // Liquify is a full-window custom dialog with focusable sliders. egui can therefore claim
    // keyboard input before the distortion-mode handler below runs. Give Liquify's local
    // shortcuts first refusal (Undo, brush size, tool keys), but never steal keys from text edits.
    if app.distort.liquify.is_some() {
        // Undo and redo still apply while a numeric field (Size, Pressure, …) has focus.
        // Other Liquify keys must not steal characters from that field (#4027).
        if ctx.text_edit_focused() {
            crate::liquify_ui::history_keys(app, ctx);
        } else {
            crate::liquify_ui::keys(app, ctx);
        }
    }
    use crate::shortcut_dispatch::{Focus, dispatch_pressed};
    let focus = Focus::of(ctx);
    // Dialogs (and Liquify's panel while one of its controls has focus) keep canvas zoom; a
    // focused text field keeps its keys.
    let liquify_focus = app.distort.liquify.is_some() && focus != Focus::None;
    if liquify_focus || !app.ui.dialogs.is_empty() || app.discard.is_some() {
        if focus != Focus::Text {
            nav_keys(app, ctx);
        }
        return;
    }
    if focus == Focus::Text {
        // A focused field keeps its typing and editing keys; menu shortcuts (⌘J, ⌘S, F7…) still
        // fire, as in Photoshop.
        dispatch_pressed(app, ctx, focus, false);
        return;
    }
    // Liquify / Puppet Warp / Perspective Warp: ↩ commits, Esc cancels.
    if crate::distort_ui::keys(app, ctx) {
        return;
    }
    // Free Transform: ↩ commits, Esc cancels.
    if app.ui.transform.is_some() {
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
            crate::transform_tool::commit(app);
            return;
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            crate::transform_tool::cancel(app);
            return;
        }
    }
    // Content-Aware Move's Transform On Drop box: ↩ computes the move, Esc drops it.
    if crate::cam_transform::active(app) {
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
            crate::cam_transform::commit(app);
            return;
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            crate::cam_transform::cancel(app);
            return;
        }
    }
    // A Properties Levels/Curves eyedropper: Esc disarms it.
    if crate::layer_picker::armed(app).is_some() && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
        crate::layer_picker::cancel(app);
        return;
    }
    // Crop tool with a pending frame: arrows nudge it (⇧ ×10) and X swaps its orientation, before
    // the selection / layer nudge and the X colour swap (#1919).
    if focus == Focus::None && crate::crop_ui::keys(app, ctx) {
        return;
    }
    // Perspective Crop with a pending quad: ↵ commits, Esc cancels.
    if focus == Focus::None && crate::perspective_crop_ui::keys(app, ctx) {
        return;
    }
    // Move tool / Free Transform: arrows nudge (⇧ ×10, ⌥ duplicates first).
    if focus == Focus::None && crate::move_mods::arrow_keys(app, ctx) {
        return;
    }
    // Moved pixels still floating: ↩ drops them.
    if focus == Focus::None
        && app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some())
        && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter))
    {
        if let Err(e) = app.run("select.drop", json!({})) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        return;
    }
    // A Pen path's points are uncommitted gesture state, not History entries (#1466).
    // Intercept Cmd/Ctrl+Z before the normal Edit › Undo shortcut, as well as unmodified
    // Backspace/Delete. Once the last anchor is removed, regular Undo works again.
    if app.ui.pen.as_ref().is_some_and(|pen| !pen.knots.is_empty()) {
        let mods = ctx.input(|i| i.modifiers);
        let command_undo = effective_shortcut(app, "edit.undo", default_shortcut("edit.undo").as_deref())
            .and_then(|shortcut| parse(&shortcut))
            .is_some_and(|shortcut| consume(ctx, &shortcut));
        let remove = !mods.command
            && !mods.ctrl
            && !mods.shift
            && !mods.alt
            && ctx.input_mut(|i| i.consume_key(mods, Key::Backspace) || i.consume_key(mods, Key::Delete));
        if command_undo || remove {
            crate::vector_ui::pen_undo_last_point(app);
            return;
        }
    }
    // Pen path in progress: ↩ and Esc both end it as an open path and keep it, as in Photoshop (#1769).
    if app.ui.pen.is_some() && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter) || i.consume_key(Modifiers::NONE, Key::Escape)) {
        crate::vector_ui::pen_commit(app, false);
        return;
    }
    // Magnetic Lasso: ↩ closes, Esc cancels, ⌫ removes a fastening point (not Edit › Clear),
    // [ ] change the detection width (not the brush size).
    if crate::magnetic_lasso_ui::keys(app, ctx) {
        return;
    }
    // While placing polygonal lasso vertices, Backspace/Delete retracts the last point;
    // it must win over the regular Edit › Clear shortcut, which edits document pixels.
    if crate::lasso_ui::waiting_for_vertex(app) {
        let mods = ctx.input(|i| i.modifiers);
        if !mods.command && !mods.ctrl && !mods.shift && ctx.input_mut(|i| i.consume_key(mods, Key::Backspace) || i.consume_key(mods, Key::Delete)) {
            crate::lasso_ui::undo_last_vertex(app);
            return;
        }
    }
    // The Polygonal Lasso tool likewise: ⌫/Delete removes the last vertex (#1229), and never
    // reaches Edit › Clear while a polygon is being drawn.
    if !app.ui.polygon.is_empty() {
        let mods = ctx.input(|i| i.modifiers);
        if !mods.command && !mods.ctrl && !mods.shift && ctx.input_mut(|i| i.consume_key(mods, Key::Backspace) || i.consume_key(mods, Key::Delete)) {
            crate::canvas::polygon_retract(app);
            return;
        }
    }
    // Space+F (⇧Space+F backwards) cycles the pasteboard colour, as in Photoshop: F while the
    // Hand key is held, before F's own Cycle Screen Mode.
    if !matches!(focus, Focus::Text) && app.ui.text_edit.is_none() && crate::hold_keys::reposition_held(app, ctx) {
        let back = ctx.input_mut(|i| i.consume_key(Modifiers::SHIFT, Key::F));
        if back || ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::F)) {
            if let Err(e) = crate::canvas_tool_menu::cycle_canvas_color(app, back) {
                app.ui.status = e;
            }
            return;
        }
    }
    // Inline type editing eats text and navigation keys; ⌘-shortcuts still reach the menus.
    let editing = crate::type_tool::handle_keys(app, ctx);
    // ⌥⇧⌘N: a new layer at once, without the New Layer dialog ⇧⌘N opens (#4794).
    if new_layer_without_dialog(app, ctx) {
        return;
    }
    // Registry, UI and menu-catalogue shortcuts (see [`crate::shortcut_dispatch::bindings`]).
    if dispatch_pressed(app, ctx, focus, editing) {
        return;
    }
    if editing {
        return;
    }
    // Crop tool showing a box: O cycles its overlay, ⇧O the overlay's orientation (#1919).
    if crate::crop_overlay::keys(app, ctx) {
        return;
    }
    // ... and H toggles Show Cropped Area.
    if crate::crop_shield::keys(app, ctx) {
        return;
    }
    // ... and I fills W x H x Resolution from the document (Front Image, #2443).
    if crate::crop_size::keys(app, ctx) {
        return;
    }
    // Single-key tools (no modifiers). D and X (`tools.defaultColors` / `tools.swapColors`) and the
    // brush keys [ ] ⇧[ ⇧] (`tools.decreaseBrushSize`…) are commands, dispatched above with any
    // Keyboard Shortcuts override.
    let pressed = |k: Key| ctx.input_mut(|i| i.consume_key(Modifiers::NONE, k));
    // Remove tool with strokes waiting (Remove after each stroke off): ↩ applies them, Esc drops them.
    if !app.remove_pending.is_empty() {
        if pressed(Key::Enter) {
            crate::retouch_ui::remove_pending_commit(app);
            return;
        }
        if pressed(Key::Escape) {
            crate::retouch_ui::remove_pending_cancel(app);
            return;
        }
    }
    // Enter / Escape commit or cancel in-progress tool state (polygonal lasso, crop).
    if crate::lasso_ui::active(app) {
        let mods = ctx.input(|i| i.modifiers);
        if ctx.input_mut(|i| i.consume_key(mods, Key::Escape)) {
            app.drag = None;
            return;
        }
        if ctx.input_mut(|i| i.consume_key(mods, Key::Enter)) {
            crate::lasso_ui::commit(app);
            return;
        }
    }
    if !app.ui.polygon.is_empty() || app.ui.crop_rect.is_some() {
        if pressed(Key::Enter) {
            if !app.ui.polygon.is_empty() {
                crate::canvas::commit_polygon(app);
            } else {
                crate::canvas::commit_crop(app);
            }
            return;
        }
        if pressed(Key::Escape) {
            app.ui.polygon.clear();
            app.ui.polygon_mode.clear();
            crate::crop_ui::cancel(app);
            return;
        }
    }
    // Esc with no active operation leaves a full screen mode, else deselects (#2674), like ⌘D.
    // The key is never taken from a state that handles Esc itself: menus, dialogs, text fields
    // and inline type returned above, transform, pen, lasso and crop consumed it, and these own
    // theirs later in the frame: the palette, the canvas and layer menus, the brush picker,
    // Rotate View, Adaptive Wide Angle, the Filter Gallery, the TIFF prompt, any open egui popup
    // and the toolbar tool flyout. A canvas drag in progress (a marquee being drawn) keeps it too.
    let esc_taken = app.drag.is_some()
        || app.ui.palette_open
        || app.ui.canvas_tool_menu.is_some()
        || app.ui.layer_menu.is_some()
        || app.ui.brush_picker.is_some()
        || app.ui.tool == Tool::RotateView
        || app.wide_angle.is_some()
        || app.save_options_open()
        || app.distort.gallery.is_some()
        || ctx.any_popup_open()
        || ctx.data(|d| d.get_temp::<(egui::Id, egui::Rect)>(egui::Id::new("tool-flyout")).is_some());
    // Both full screen modes (with and without the menu bar) return to Standard, as in Photoshop
    // (#4116), through the menu command so the window and the Screen Mode check follow.
    if !esc_taken && app.ui.view.screen_mode != "standard" && pressed(Key::Escape) {
        if let Err(e) = crate::menus::invoke(app, ctx, "view.screenMode.standard", json!({})) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        return;
    }
    if !esc_taken && app.session.active().is_some_and(|st| st.doc.selection.is_some()) && pressed(Key::Escape) {
        if let Err(e) = app.run("select.deselect", json!({})) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        return;
    }
    // Tool keys; pressing the key of the current group cycles within it. With Preferences ›
    // Tools › Use Shift Key for Tool Switch, only ⇧+key cycles and the plain key keeps the
    // group's current tool. From another group, the key picks the group's last-used tool.
    // A held tool letter is a spring-loaded tool (#2672), not a burst of presses: its key
    // repeats are dropped rather than cycling the group.
    ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Key { key, pressed: true, repeat: true, .. } if crate::spring_tools::is_tool_key(*key))));
    let shift_switch = app.session.prefs().tools.use_shift_key_for_tool_switch;
    for t in Tool::ALL {
        if !crate::tool_shortcuts::uses_default(app, t) || (app.services.host_save.is_some() && !crate::tool_shortcuts::page_tool(t)) {
            continue;
        }
        let Some(k) = crate::spring_tools::key_of(t.key()) else { continue };
        let cycle_shift = shift_switch && ctx.input_mut(|i| i.consume_key(Modifiers::SHIFT, k));
        if cycle_shift || pressed(k) {
            let before = app.ui.tool;
            let group: Vec<Tool> = Tool::ALL
                .iter()
                .copied()
                .filter(|x| {
                    x.key() == t.key()
                        && crate::tool_shortcuts::uses_default(app, *x)
                        && (app.services.host_save.is_none() || crate::tool_shortcuts::page_tool(*x))
                })
                .collect();
            app.ui.tool = match group.iter().position(|x| *x == app.ui.tool) {
                Some(i) if shift_switch && !cycle_shift => group[i],
                Some(i) => group[(i + 1) % group.len()],
                None => app.ui.group_tool(t.key()).filter(|last| group.contains(last)).unwrap_or(group[0]),
            };
            // `consume_key` ignores extra ⇧ and ⌥: a ⇧/⌥ switch is never spring-loaded.
            let (plain, now) = ctx.input(|i| (!cycle_shift && !i.modifiers.shift && !i.modifiers.alt, i.time));
            crate::spring_tools::pressed(app, t.key(), before, plain, now);
            return;
        }
    }
    // 1–0 set opacity, ⇧1–0 flow or fill (`opacity_keys`).
    crate::opacity_keys::handle(app, ctx);
    // ⇧+ / ⇧− step the blend mode (`blend_keys`).
    crate::blend_keys::handle(app, ctx);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_tool_keys_cannot_activate_page_resize_tools() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.services.host_save = Some(Box::new(|| {}));
        app.ui.tool = Tool::Move;
        let ctx = egui::Context::default();
        for (key, modifiers, expected) in
            [(Key::C, Modifiers::NONE, Tool::Move), (Key::V, Modifiers::SHIFT, Tool::Move), (Key::B, Modifiers::NONE, Tool::Brush)]
        {
            ctx.begin_pass(egui::RawInput {
                events: vec![egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }],
                ..Default::default()
            });
            handle(&mut app, &ctx);
            ctx.end_pass().textures_delta.clear();
            assert_eq!(app.ui.tool, expected);
        }
    }

    #[test]
    fn tool_key_brings_back_the_groups_last_used_tool() {
        use crate::state::Tool;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        let press = |app: &mut PhotocraftApp, key, modifiers| {
            let raw = egui::RawInput {
                events: vec![
                    egui::Event::ModifiersChanged(modifiers),
                    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers },
                    egui::Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers },
                ],
                ..Default::default()
            };
            ctx.begin_pass(raw);
            handle(app, &ctx);
            ctx.end_pass().textures_delta.clear();
        };
        // Plain-key cycling (the pref is on by default, as in Photoshop).
        app.session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = false);
        // Polygonal Lasso picked from the toolbar flyout, then the Brush (#2608).
        app.ui.tool = Tool::PolygonLasso;
        press(&mut app, Key::B, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::Brush);
        press(&mut app, Key::L, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::PolygonLasso);
        // Within the group, L still cycles; leaving and coming back keeps the new pick.
        press(&mut app, Key::L, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::MagneticLasso);
        press(&mut app, Key::M, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::RectMarquee);
        press(&mut app, Key::L, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::MagneticLasso);
        // Use Shift Key for Tool Switch: the plain key also returns to the last-used tool.
        app.session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = true);
        press(&mut app, Key::B, Modifiers::NONE);
        press(&mut app, Key::L, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::MagneticLasso);
    }

    /// #3296: by default T keeps the current type tool and only Shift+T cycles, as in Photoshop.
    #[test]
    fn shift_key_tool_switch_is_the_default() {
        use crate::state::Tool;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        let press = |app: &mut PhotocraftApp, modifiers| {
            let raw = egui::RawInput {
                events: vec![
                    egui::Event::ModifiersChanged(modifiers),
                    egui::Event::Key { key: Key::T, physical_key: None, pressed: true, repeat: false, modifiers },
                    egui::Event::Key { key: Key::T, physical_key: None, pressed: false, repeat: false, modifiers },
                ],
                ..Default::default()
            };
            ctx.begin_pass(raw);
            handle(app, &ctx);
            ctx.end_pass().textures_delta.clear();
        };
        app.ui.tool = Tool::Type;
        press(&mut app, Modifiers::NONE);
        press(&mut app, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::Type);
        press(&mut app, Modifiers::SHIFT);
        assert_ne!(app.ui.tool, Tool::Type);
        // A stored "off" restores plain-key cycling.
        app.session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = false);
        app.ui.tool = Tool::Type;
        press(&mut app, Modifiers::NONE);
        assert_ne!(app.ui.tool, Tool::Type);
    }

    /// #3109: on a German layout Ö / # (physical `;` / `\`) step the brush size and ⇧Ö / ⇧#
    /// (⇧# types `'`) its hardness, as Photoshop's German shortcuts.
    #[test]
    fn german_o_umlaut_and_hash_are_the_brush_keys() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.ui.tool = Tool::Brush;
        app.session.tools.brush.hardness = 0.5;
        let ctx = egui::Context::default();
        let typed = |app: &mut PhotocraftApp, key, physical_key, text: &str, modifiers| {
            let press = egui::Event::Key { key, physical_key: Some(physical_key), pressed: true, repeat: false, modifiers };
            let raw = egui::RawInput { events: vec![egui::Event::ModifiersChanged(modifiers), press, egui::Event::Text(text.into())], ..Default::default() };
            ctx.begin_pass(raw);
            handle(app, &ctx);
            ctx.end_pass().textures_delta.clear();
        };
        let size = app.session.tools.brush.size;
        typed(&mut app, Key::Backslash, Key::Backslash, "#", Modifiers::NONE);
        assert!(app.session.tools.brush.size > size, "# is ]");
        typed(&mut app, Key::Semicolon, Key::Semicolon, "ö", Modifiers::NONE);
        assert_eq!(app.session.tools.brush.size, size, "ö is [");
        typed(&mut app, Key::Quote, Key::Backslash, "'", Modifiers::SHIFT);
        assert!(app.session.tools.brush.hardness > 0.5, "⇧# is ⇧]");
        typed(&mut app, Key::Semicolon, Key::Semicolon, "Ö", Modifiers::SHIFT);
        assert_eq!(app.session.tools.brush.hardness, 0.5, "⇧Ö is ⇧[");
    }

    #[test]
    fn pen_command_z_and_backspace_retract_points_before_document_undo() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", serde_json::json!({"width": 200, "height": 200})).unwrap();
        app.ui.tool = crate::state::Tool::Pen;
        app.run("shape.create", serde_json::json!({"kind": "rect", "rect": [10, 10, 40, 40], "fill": "#ff0000"})).unwrap();
        let history = app.session.active().unwrap().history.past_len();
        for (x, y) in [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0)] {
            crate::vector_ui::pen_down(&mut app, x, y);
            crate::vector_ui::pen_up(&mut app);
        }
        let ctx = egui::Context::default();
        let press = |app: &mut PhotocraftApp, key, modifiers| {
            let raw = egui::RawInput {
                events: vec![egui::Event::ModifiersChanged(modifiers), egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }],
                ..Default::default()
            };
            ctx.begin_pass(raw);
            handle(app, &ctx);
            ctx.end_pass().textures_delta.clear();
        };

        press(&mut app, Key::Z, Modifiers::COMMAND);
        assert_eq!(app.ui.pen.as_ref().unwrap().knots.len(), 2);
        assert_eq!(app.session.active().unwrap().history.past_len(), history);
        press(&mut app, Key::Backspace, Modifiers::NONE);
        assert_eq!(app.ui.pen.as_ref().unwrap().knots.len(), 1);
        press(&mut app, Key::Delete, Modifiers::NONE);
        assert!(app.ui.pen.is_none());
        assert_eq!(app.session.active().unwrap().history.past_len(), history);
        // No unfinished Pen points remain, so regular Undo can now affect the document.
        app.run("edit.undo", serde_json::json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().history.past_len(), history - 1);
    }

    #[test]
    fn pen_custom_undo_binding_and_menu_retract_pending_points() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", serde_json::json!({"width": 200, "height": 200})).unwrap();
        app.ui.tool = crate::state::Tool::Pen;
        for (x, y) in [(10.0, 10.0), (50.0, 10.0)] {
            crate::vector_ui::pen_down(&mut app, x, y);
            crate::vector_ui::pen_up(&mut app);
        }
        app.session.prefs.edit(|p| p.shortcuts.insert("edit.undo".into(), "Cmd+Shift+Y".into()));
        let ctx = egui::Context::default();
        let press = |app: &mut PhotocraftApp, key: Key, modifiers: Modifiers| {
            ctx.begin_pass(egui::RawInput {
                events: vec![egui::Event::ModifiersChanged(modifiers), egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }],
                ..Default::default()
            });
            handle(app, &ctx);
            ctx.end_pass().textures_delta.clear();
        };
        // Once rebound, the default key must no longer intercept pending points.
        press(&mut app, Key::Z, Modifiers::COMMAND);
        assert_eq!(app.ui.pen.as_ref().unwrap().knots.len(), 2);
        press(&mut app, Key::Y, Modifiers::COMMAND | Modifiers::SHIFT);
        assert_eq!(app.ui.pen.as_ref().unwrap().knots.len(), 1);
        assert!(crate::menus::is_enabled(&app, "edit.undo"));
        crate::menus::invoke(&mut app, &ctx, "edit.undo", serde_json::json!({})).unwrap();
        assert!(app.ui.pen.is_none());
    }

    #[test]
    fn parses_registry_shortcuts() {
        let sc = parse("Cmd+Shift+N").unwrap();
        assert_eq!(sc.logical_key, Key::N);
        assert!(sc.modifiers.command && sc.modifiers.shift);
        assert_eq!(parse("Cmd+]").unwrap().logical_key, Key::CloseBracket);
        assert_eq!(parse("Cmd+=").unwrap().logical_key, Key::Equals);
        assert_eq!(parse("F7").unwrap().logical_key, Key::F7);
        assert!(parse("Cmd+Nonsense").is_none());
    }

    #[test]
    fn malformed_bindings_do_not_silently_match_a_different_shortcut() {
        for invalid in ["Cmd+Typo+Z", "Cmd+X+Z", "Cmd+Z+X", "Cmd+Z+", "Cmd+Cmd+Z", "Ctrl+Ctrl+Z", "Shift+Shift+Z", "Cmd+Shift+++"] {
            assert!(parse(invalid).is_none(), "{invalid} must not bind a command");
        }
        // '+' is a key name as well as a separator; keep existing shortcut spellings.
        for plus in ["+", "Cmd+", "Cmd++", "Cmd+Shift++"] {
            assert_eq!(parse(plus).unwrap().logical_key, Key::Plus, "{plus}");
        }
        assert!(parse("Cmd+Shift+Z").is_some());
        assert_eq!(parse("Cmd+;").unwrap().logical_key, Key::Semicolon);
    }

    #[test]
    fn keys_match_like_photoshop() {
        let cmd = Modifiers::COMMAND;
        let zoom_in = parse("Cmd+=").unwrap();
        // ⌘= , ⌘+ typed with ⇧ (US), ⌘+ from the numpad or a Nordic/German `+` key.
        assert!(key_matches(&zoom_in, Key::Equals, cmd));
        assert!(key_matches(&zoom_in, Key::Plus, cmd | Modifiers::SHIFT));
        assert!(key_matches(&zoom_in, Key::Plus, cmd));
        assert!(!key_matches(&zoom_in, Key::Plus, Modifiers::NONE));
        assert!(!key_matches(&zoom_in, Key::Equals, cmd | Modifiers::ALT));
        assert!(key_matches(&parse("Cmd+-").unwrap(), Key::Minus, cmd));
        assert!(key_matches(&parse("Cmd+0").unwrap(), Key::Num0, cmd));
        // Ctrl (Windows) is ⌘.
        assert!(key_matches(&zoom_in, Key::Plus, Modifiers::CTRL | Modifiers { command: true, ..Default::default() }));
        // Exact modifiers: ⌘⌥I is Image Size, never Invert (⌘I); ⇧⌘Z is not ⌘Z.
        let invert = parse("Cmd+I").unwrap();
        assert!(key_matches(&invert, Key::I, cmd));
        assert!(!key_matches(&invert, Key::I, cmd | Modifiers::ALT));
        assert!(!key_matches(&parse("Cmd+Z").unwrap(), Key::Z, cmd | Modifiers::SHIFT));
        // ⇧ shortcuts on punctuation arrive as the shifted character.
        assert!(key_matches(&parse("Cmd+Shift+;").unwrap(), Key::Colon, cmd | Modifiers::SHIFT));
        assert!(key_matches(&parse("Cmd+Shift+]").unwrap(), Key::CloseCurlyBracket, cmd | Modifiers::SHIFT));
        assert!(!key_matches(&parse("Cmd+;").unwrap(), Key::Colon, cmd | Modifiers::SHIFT));
        // Delete and the Mac's delete (Backspace) both clear.
        let clear = parse("Delete").unwrap();
        assert!(key_matches(&clear, Key::Delete, Modifiers::NONE) && key_matches(&clear, Key::Backspace, Modifiers::NONE));
        assert!(!key_matches(&clear, Key::Backspace, cmd));
    }

    #[test]
    fn pretty_uses_the_platform_notation() {
        let mac = cfg!(target_os = "macos");
        assert_eq!(pretty("Cmd+Shift+N"), if mac { "⌘⇧N" } else { "Ctrl+Shift+N" });
        // Lone keys and inline-hint combos.
        assert_eq!(pretty("Alt"), if mac { "⌥" } else { "Alt" });
        assert_eq!(pretty("Shift+Alt"), if mac { "⇧⌥" } else { "Shift+Alt" });
        assert_eq!(pretty("Enter"), if mac { "↩" } else { "Enter" });
        assert_eq!(pretty("Cmd+Enter"), if mac { "⌘↩" } else { "Ctrl+Enter" });
    }

    #[test]
    fn labels_show_the_effective_shortcut_in_platform_notation() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let new = command_label(&app, "New document…", "file.new");
        if cfg!(target_os = "macos") {
            assert_eq!(new, "New document…     ⌘N");
        } else {
            assert_eq!(new, "New document…     Ctrl+N");
        }
        assert_eq!(tip_label(&app, "Rulers", "view.rulers"), format!("Rulers  ({})", pretty("Cmd+R")));
        // A user override is picked up…
        app.session.prefs.edit(|p| p.shortcuts.insert("file.new".into(), "Cmd+Shift+N".into()));
        assert_eq!(command_label(&app, "New document…", "file.new"), format!("New document…     {}", pretty("Cmd+Shift+N")));
        // …and a removed binding hides the shortcut entirely.
        app.session.prefs.edit(|p| p.shortcuts.insert("file.new".into(), String::new()));
        assert_eq!(command_label(&app, "New document…", "file.new"), "New document…");
    }

    #[test]
    fn clipboard_events_become_pixel_shortcuts_outside_text() {
        let ctx = egui::Context::default();
        let raw = |mut events: Vec<egui::Event>, modifiers| {
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
            egui::RawInput { events, ..Default::default() }
        };
        let keys = |r: &egui::RawInput| -> Vec<(Key, bool, bool)> {
            r.events
                .iter()
                .filter_map(|e| if let egui::Event::Key { key, pressed, modifiers, .. } = e { Some((*key, *pressed, modifiers.shift)) } else { None })
                .collect()
        };
        let cmd_shift = Modifiers::COMMAND | Modifiers::SHIFT;
        let mut r = raw(vec![egui::Event::Copy, egui::Event::Cut], cmd_shift);
        clipboard_keys(&ctx, false, &mut r);
        assert_eq!(keys(&r), vec![(Key::C, true, true), (Key::X, true, true)]);
        // A text paste becomes ⌘V; its key release then adds nothing.
        let up = egui::Event::Key { key: Key::V, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::COMMAND };
        let mut r = raw(vec![egui::Event::Paste("x".into()), up.clone()], Modifiers::COMMAND);
        clipboard_keys(&ctx, false, &mut r);
        assert_eq!(keys(&r), vec![(Key::V, true, false), (Key::V, false, false)]);
        // An image on the clipboard: egui-winit sends only the release. It still pastes.
        let mut r = raw(vec![up.clone()], Modifiers::COMMAND);
        clipboard_keys(&ctx, false, &mut r);
        assert_eq!(keys(&r), vec![(Key::V, true, false), (Key::V, false, false)]);
        // In a text field the events stay text clipboard events.
        let mut r = raw(vec![egui::Event::Paste("x".into())], Modifiers::COMMAND);
        clipboard_keys(&ctx, true, &mut r);
        assert!(matches!(r.events.as_slice(), [_, egui::Event::Paste(_)]));
    }

    /// In a text field the Mac menu's ⌘C / ⌘X key equivalents become the field's Copy / Cut, and
    /// its ⌘V asks the window for the clipboard text, which the field pastes. Other keys, and ⇧ / ⌥
    /// chords, pass through.
    #[test]
    fn menu_key_equivalents_reach_a_text_field_as_clipboard_events() {
        let ctx = egui::Context::default();
        let key = |key, pressed, modifiers| egui::Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers };
        let cmd = Modifiers::COMMAND;
        let events = vec![
            key(Key::C, true, cmd),
            key(Key::X, true, cmd),
            key(Key::V, true, cmd),
            key(Key::V, false, cmd),
            key(Key::A, true, cmd),
            key(Key::C, true, cmd | Modifiers::SHIFT),
        ];
        let mut r = egui::RawInput { events: [vec![egui::Event::ModifiersChanged(cmd)], events].concat(), ..Default::default() };
        // In a frame, as the app calls it, so the request for the clipboard text is in its output.
        ctx.begin_pass(egui::RawInput::default());
        clipboard_keys(&ctx, true, &mut r);
        let mut out = ctx.end_pass();
        out.textures_delta.clear();
        assert!(
            matches!(
                &r.events[1..],
                [
                    egui::Event::Copy,
                    egui::Event::Cut,
                    egui::Event::Key { key: Key::V, pressed: false, .. },
                    egui::Event::Key { key: Key::A, .. },
                    egui::Event::Key { key: Key::C, .. }
                ]
            ),
            "{:?}",
            r.events
        );
        // The ⌘V press became a request for the clipboard's text.
        let asked = out.viewport_output.values().flat_map(|v| &v.commands).any(|c| matches!(c, egui::ViewportCommand::RequestPaste));
        assert!(asked, "the window is asked for the clipboard text");
    }

    /// #1638: every ⌘V is one press, however its key-down and key-up arrive.
    #[test]
    fn each_paste_key_is_one_press() {
        let ctx = egui::Context::default();
        let key = |key, pressed, modifiers| egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers };
        let presses = |events: Vec<egui::Event>| {
            let mut r = egui::RawInput { events, ..Default::default() };
            clipboard_keys(&ctx, false, &mut r);
            r.events.iter().filter(|e| matches!(e, egui::Event::Key { key: Key::V, pressed: true, .. })).count()
        };
        // A ⌘V press that reached us (the Mac menu's key equivalent), then its release.
        assert_eq!(presses(vec![key(Key::V, true, Modifiers::COMMAND)]), 1);
        assert_eq!(presses(vec![key(Key::V, false, Modifiers::COMMAND)]), 0, "the release of a press doesn't paste again");
        // A text paste whose V is released after ⌘: the next image paste still pastes.
        assert_eq!(presses(vec![egui::Event::Paste("x".into())]), 1);
        assert_eq!(presses(vec![egui::Event::ModifiersChanged(Modifiers::NONE), key(Key::V, false, Modifiers::NONE)]), 0);
        assert_eq!(presses(vec![egui::Event::ModifiersChanged(Modifiers::COMMAND), key(Key::V, false, Modifiers::COMMAND)]), 1, "image paste");
    }

    /// #2638: on the web one ⌘V brings its key press and the browser's paste event; that is one paste.
    #[test]
    fn web_paste_key_and_paste_event_are_one_press() {
        let ctx = egui::Context::default();
        let key = |key, pressed| egui::Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers: Modifiers::COMMAND };
        let mut r =
            egui::RawInput { events: vec![key(Key::V, true), egui::Event::Paste("x".into()), key(Key::X, true), egui::Event::Cut], ..Default::default() };
        clipboard_keys(&ctx, false, &mut r);
        let presses = |k| r.events.iter().filter(|e| matches!(e, egui::Event::Key { key, pressed: true, .. } if *key == k)).count();
        assert_eq!((presses(Key::V), presses(Key::X)), (1, 1));
        let mut r = egui::RawInput { events: vec![key(Key::V, false)], ..Default::default() };
        clipboard_keys(&ctx, false, &mut r);
        assert!(!r.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. })), "its release doesn't paste again");
    }

    /// #530: egui-winit sends Cut for Shift+Delete on Windows; on the canvas it opens Fill.
    #[test]
    fn windows_shift_delete_fills_instead_of_cutting() {
        let shift = Modifiers::SHIFT;
        let ctrl = Modifiers::CTRL | Modifiers::COMMAND;
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let bound = |held| {
            let (key, mods) = clipboard_press(&egui::Event::Cut, held, true)?;
            assert_eq!((key, mods), (Key::Delete, held));
            crate::shortcut_dispatch::bindings(&app).into_iter().find(|(_, sc)| key_matches(sc, key, mods)).map(|(id, _)| id)
        };
        assert_eq!(bound(shift).as_deref(), Some("edit.fill"));
        // ⇧⌥Delete keeps ⌥: fill with the foreground, keeping transparency.
        assert_eq!(bound(shift | Modifiers::ALT).as_deref(), Some("edit.fillForegroundPreserve"));
        // Ctrl+X (and Ctrl+Shift+X) still cut; elsewhere a Cut is always ⌘X.
        assert_eq!(clipboard_press(&egui::Event::Cut, ctrl, true), Some((Key::X, ctrl)));
        assert_eq!(clipboard_press(&egui::Event::Cut, ctrl | shift, true), Some((Key::X, ctrl | shift)));
        assert_eq!(clipboard_press(&egui::Event::Cut, shift, false), Some((Key::X, Modifiers::COMMAND)));
        // Shift+Insert pastes on Windows.
        assert_eq!(clipboard_press(&egui::Event::Paste("x".into()), shift, true), Some((Key::V, Modifiers::COMMAND)));
        // Through the raw-input hook on this platform; a text field keeps its Cut.
        let ctx = egui::Context::default();
        let cut = || egui::RawInput { events: vec![egui::Event::ModifiersChanged(shift), egui::Event::Cut], ..Default::default() };
        let mut r = cut();
        clipboard_keys(&ctx, false, &mut r);
        let want = if cfg!(target_os = "windows") { Key::Delete } else { Key::X };
        assert!(matches!(r.events.as_slice(), [_, egui::Event::Key { key, pressed: true, .. }] if *key == want), "{:?}", r.events);
        let mut r = cut();
        clipboard_keys(&ctx, true, &mut r);
        assert!(matches!(r.events.as_slice(), [_, egui::Event::Cut]));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ctrl_and_shift_insert_copy_and_paste_on_linux() {
        let ctx = egui::Context::default();
        let insert = |modifiers| egui::RawInput {
            events: vec![egui::Event::Key { key: Key::Insert, physical_key: None, pressed: true, repeat: false, modifiers }],
            ..Default::default()
        };
        let pressed = |r: &egui::RawInput| match r.events.as_slice() {
            [egui::Event::Key { key, pressed: true, modifiers, .. }] if modifiers.command => Some(*key),
            _ => None,
        };
        // On the canvas they become the pixel commands' ⌘C / ⌘V.
        let mut r = insert(Modifiers::CTRL);
        clipboard_keys(&ctx, false, &mut r);
        assert_eq!(pressed(&r), Some(Key::C));
        let mut r = insert(Modifiers::SHIFT);
        clipboard_keys(&ctx, false, &mut r);
        assert_eq!(pressed(&r), Some(Key::V));
        // In a text field Ctrl+Insert copies the text; Shift+Insert asks egui-winit for the text.
        let mut r = insert(Modifiers::CTRL);
        clipboard_keys(&ctx, true, &mut r);
        assert!(matches!(r.events.as_slice(), [egui::Event::Copy]));
        let mut r = insert(Modifiers::SHIFT);
        clipboard_keys(&ctx, true, &mut r);
        assert!(r.events.is_empty());
        // Plain Insert is left alone.
        let mut r = insert(Modifiers::NONE);
        clipboard_keys(&ctx, false, &mut r);
        assert!(matches!(r.events.as_slice(), [egui::Event::Key { key: Key::Insert, .. }]));
    }

    #[test]
    fn every_registered_shortcut_parses() {
        for c in photocraft_engine::command_specs() {
            if let Some(sc) = c.shortcut {
                assert!(parse(sc).is_some(), "{}: {sc}", c.id);
            }
        }
        for (id, _, _, sc) in crate::menus::UI_COMMANDS {
            if let Some(sc) = sc {
                assert!(parse(sc).is_some(), "{id}: {sc}");
            }
        }
    }

    #[test]
    fn no_duplicate_shortcuts() {
        let mut seen = std::collections::HashMap::new();
        let all = photocraft_engine::command_specs()
            .iter()
            .filter_map(|c| c.shortcut.map(|s| (c.id, s)))
            .chain(crate::menus::UI_COMMANDS.iter().filter_map(|(id, _, _, s)| s.map(|s| (*id, s))));
        for (id, s) in all {
            if let Some(prev) = seen.insert(s, id) {
                panic!("shortcut {s} bound to both {prev} and {id}");
            }
        }
    }
}
