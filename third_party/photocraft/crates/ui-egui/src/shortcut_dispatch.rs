//! Keyboard shortcut table and dispatch (#127): every shortcut the menus display fires.
//!
//! The menus show Photoshop's default shortcuts from [`crate::menu_catalog::CATALOG`], while
//! commands also declare their own (`CommandSpec::shortcut`, [`crate::menus::UI_COMMANDS`]).
//! The binding table merges all three, with Edit › Keyboard Shortcuts overrides on top, so a
//! shortcut printed next to a live menu item always reaches that item. A shortcut whose command
//! is disabled in the current state reports why on the status bar instead of doing nothing
//! silently, as the greyed menu item shows.

use egui::{Key, KeyboardShortcut};
use serde_json::json;

use crate::PhotocraftApp;
use crate::shortcuts::{key_matches, parse};

/// What a dispatched shortcut did (kept for tests and `ui.inspect`-style debugging).
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Ran,
    Disabled(String),
    Failed(String),
}

fn log_id() -> egui::Id {
    egui::Id::new("pc-shortcut-log")
}

fn dry_run_id() -> egui::Id {
    egui::Id::new("pc-shortcut-dry-run")
}

/// Tests: resolve and check shortcuts without running them (File › Exit, Open, browser links…).
pub fn set_dry_run(ctx: &egui::Context, on: bool) {
    ctx.data_mut(|d| d.insert_temp(dry_run_id(), on));
}

/// The shortcuts dispatched since the last call (command id and outcome), oldest first.
pub fn take_log(ctx: &egui::Context) -> Vec<(String, Outcome)> {
    ctx.data_mut(|d| d.remove_temp::<Vec<(String, Outcome)>>(log_id())).unwrap_or_default()
}

/// Alternative default shortcuts Photoshop gives some commands besides the one its menu shows:
/// (command, shortcut). Edit › Fill… is Shift+F5 and also Shift+Backspace (Shift+Delete on a Mac).
pub const SECONDARY: &[(&str, &str)] = &[("edit.fill", "Shift+Backspace")];

/// Every key binding, most modifiers first (so ⇧⌘Z wins over ⌘Z), each key owned by one command:
/// command and shell shortcuts, then the menu catalogue's for live items without their own,
/// then Edit › Keyboard Shortcuts assignments to any other menu item. D and X
/// (`tools.defaultColors` / `tools.swapColors`) and the fill keys are commands like any other, so
/// their overrides apply. Held temporary tools (Space…) are not here: see [`crate::hold_keys`].
pub fn bindings(app: &PhotocraftApp) -> Vec<(String, KeyboardShortcut)> {
    let prefs = app.session.prefs();
    let action_ids: Vec<String> = app.session.actions.list.iter().map(|a| crate::actions::shortcut_id(&a.name)).collect();
    let actions = action_ids.iter().map(|id| (id.as_str(), prefs.shortcut(id, None)));
    // The Keyboard Shortcuts dialog lists Photoshop's Window › <panel> item (`window.panel.*`)
    // in place of the shell command it runs (`window.toggle.*`, see `menus::panel_alias`), so an
    // edit to that row reassigns or removes the shell command's default too (#1272).
    let aliased = |id: &str, sc: Option<&str>| {
        !prefs.shortcuts.contains_key(id)
            && prefs.shortcuts.keys().any(|k| {
                crate::menus::panel_alias(k) == Some(id) && crate::menu_catalog::CATALOG.iter().any(|c| c.3 == k.as_str() && c.2.is_some() && c.2 == sc)
            })
    };
    let ui = crate::menus::UI_COMMANDS.iter().map(|(id, _, _, sc)| (*id, if aliased(id, *sc) { None } else { prefs.shortcut(id, *sc) }));
    let engine = photocraft_engine::command_specs().iter().map(|c| (c.id, prefs.shortcut(c.id, c.shortcut)));
    let own: std::collections::HashSet<&str> = crate::menus::UI_COMMANDS
        .iter()
        .map(|c| c.0)
        .chain(photocraft_engine::command_specs().iter().filter(|c| c.shortcut.is_some() || prefs.shortcuts.contains_key(c.id)).map(|c| c.id))
        .collect();
    let catalog = crate::menu_catalog::CATALOG
        .iter()
        .filter(|(_, _, sc, id)| sc.is_some() && !own.contains(id) && crate::menus::is_live(id))
        .map(|&(_, _, sc, id)| (id, prefs.shortcut(id, sc)));
    let overrides = prefs
        .shortcuts
        .iter()
        .filter(|(id, sc)| {
            !sc.is_empty()
                && !id.starts_with(crate::actions::SHORTCUT_PREFIX)
                && photocraft_engine::commands::find(id).is_none()
                && !crate::menus::UI_COMMANDS.iter().any(|c| c.0 == id.as_str())
                && !crate::hold_keys::is_temporary(id)
        })
        .map(|(id, sc)| (id.as_str(), Some(sc.as_str())));
    // Photoshop's second shortcuts, kept while the command's main one is the default.
    let secondary = SECONDARY.iter().filter(|(id, _)| !prefs.shortcuts.contains_key(*id)).map(|&(id, sc)| (id, Some(sc)));
    // A key the user assigned in Edit › Keyboard Shortcuts belongs to that command, ahead of any
    // default that still names it (#1272): OK clears the old owner's row, but that row can be
    // another id for the same item, or a default the dialog doesn't list.
    let mut candidates: Vec<(&str, Option<&str>)> = actions.chain(ui).chain(engine).chain(catalog).chain(overrides).chain(secondary).collect();
    candidates.sort_by_key(|(id, sc)| !(sc.is_some() && prefs.shortcuts.get(*id).map(String::as_str) == *sc));
    let mut all: Vec<(String, KeyboardShortcut)> = Vec::new();
    for (id, sc) in candidates {
        let Some(sc) = sc.and_then(parse) else { continue };
        if !all.iter().any(|(_, b)| *b == sc) {
            all.push((id.to_string(), sc));
        }
    }
    all.sort_by_key(|(_, sc)| std::cmp::Reverse(sc.modifiers.shift as u8 + sc.modifiers.alt as u8 + sc.modifiers.command as u8));
    all
}

/// Where keyboard focus is: nowhere (the canvas), on a widget (a slider, the Curves graph, a
/// button reached with Tab), or in a text field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Focus {
    None,
    Widget,
    /// A focused widget that edits with ⌫ / Delete itself (a Curves graph deletes its selected
    /// point), see [`claim_delete_keys`].
    DeletingWidget,
    Text,
}

/// Where [`claim_delete_keys`] keeps the id of the widget that owns ⌫ / Delete while focused.
fn delete_owner_id() -> egui::Id {
    egui::Id::new("shortcut_dispatch::delete_owner")
}

/// Called by a widget that has keyboard focus and uses ⌫ / Delete itself (the Curves graph):
/// while it keeps focus those keys reach it instead of Edit › Clear or Delete Layer. Every other
/// focused widget (a Layers row, a slider) leaves them to the shortcuts, as in Photoshop (#1534).
pub fn claim_delete_keys(ctx: &egui::Context, id: egui::Id) {
    ctx.data_mut(|d| d.insert_temp(delete_owner_id(), id));
}

impl Focus {
    pub fn of(ctx: &egui::Context) -> Focus {
        // A layer rename keeps its keys even in the frame egui drops its focus (Esc, Tab).
        if ctx.text_edit_focused() || crate::layer_row_ui::rename_active(ctx) {
            Focus::Text
        } else if ctx.egui_wants_keyboard_input() {
            let owner = ctx.data(|d| d.get_temp::<egui::Id>(delete_owner_id()));
            if owner.is_some() && ctx.memory(|m| m.focused()) == owner { Focus::DeletingWidget } else { Focus::Widget }
        } else {
            Focus::None
        }
    }

    /// May `sc` fire with this focus? A focused widget keeps its navigation keys (arrows, ↩,
    /// Space, Esc…, plus ⌫ / Delete when it claimed them); a text field keeps everything but ⌘ shortcuts (minus its own editing
    /// ones: select all, clipboard, undo) and the function keys. Ctrl+Tab and Ctrl+Shift+Tab switch
    /// documents whatever has focus, as in Photoshop: no widget or text field edits with them (#2340).
    pub fn allows(self, sc: &KeyboardShortcut) -> bool {
        let k = sc.logical_key;
        if k == Key::Tab && sc.modifiers.ctrl {
            return true;
        }
        let function = matches!(
            k,
            Key::F1
                | Key::F2
                | Key::F3
                | Key::F4
                | Key::F5
                | Key::F6
                | Key::F7
                | Key::F8
                | Key::F9
                | Key::F10
                | Key::F11
                | Key::F12
                | Key::F13
                | Key::F14
                | Key::F15
        );
        let navigation = matches!(
            k,
            Key::ArrowLeft
                | Key::ArrowRight
                | Key::ArrowUp
                | Key::ArrowDown
                | Key::Enter
                | Key::Space
                | Key::Tab
                | Key::Escape
                | Key::Home
                | Key::End
                | Key::PageUp
                | Key::PageDown
        );
        match self {
            Focus::None => true,
            Focus::Widget => sc.modifiers.command || !navigation,
            Focus::DeletingWidget => sc.modifiers.command || !(navigation || matches!(k, Key::Backspace | Key::Delete)),
            Focus::Text => {
                let text_edit = !sc.modifiers.alt && matches!(k, Key::A | Key::C | Key::X | Key::V | Key::Z | Key::Y)
                    || navigation
                    || matches!(k, Key::Backspace | Key::Delete);
                function || (sc.modifiers.command && !text_edit)
            }
        }
    }
}

/// What decides which keys [`crate::shortcuts::handle`] gives the shortcuts: an open menu, a
/// dialog, the unsaved-changes prompt, a warp or the Filter Gallery, inline type.
fn key_owner(app: &PhotocraftApp, ctx: &egui::Context) -> (bool, usize, bool, bool, bool) {
    let distort = app.distort.active() || app.distort.gallery.is_some();
    (crate::menu_nav::is_open(ctx), app.ui.dialogs.len(), app.discard.is_some(), distort, app.ui.text_edit.is_some())
}

/// Keys whose shortcuts follow the key's position, as in Photoshop: ⇧⌘. is ⇧⌘> on a US layout,
/// but types `:` on Nordic and German ones, so its character would never match.
const POSITIONAL: [Key; 2] = [Key::Period, Key::Comma];

/// Dispatch this frame's shortcut presses in the order they arrived (⌘Z then ⌘S undoes, then
/// saves the undone state; #440), consuming each. A press matching several bindings goes to the
/// first in table order (⇧⌘Z before ⌘Z). Stops after a shortcut that hands the keyboard to
/// someone else (opens a dialog, a prompt, a menu or inline type), leaving the later presses to
/// it. While typing (inline type), clipboard and select-all shortcuts belong to the text.
/// Returns true when a shortcut was dispatched.
pub fn dispatch_pressed(app: &mut PhotocraftApp, ctx: &egui::Context, focus: Focus, editing: bool) -> bool {
    // Building the table walks the registry: only do it when a key went down.
    if !ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. }))) {
        return false;
    }
    const TEXT_OWNED: [&str; 6] = ["edit.copy", "edit.cut", "edit.paste", "edit.copyMerged", "select.all", "edit.pasteSpecial.pasteInPlace"];
    let table: Vec<(String, KeyboardShortcut)> =
        bindings(app).into_iter().filter(|(id, sc)| focus.allows(sc) && !(editing && TEXT_OWNED.contains(&id.as_str()))).collect();
    let command = |e: &egui::Event| match e {
        egui::Event::Key { key, physical_key, pressed: true, modifiers, .. } => {
            let by_position = physical_key
                .filter(|p| POSITIONAL.contains(p))
                .and_then(|p| table.iter().find(|(_, sc)| sc.logical_key == p && key_matches(sc, p, *modifiers)));
            by_position.or_else(|| table.iter().find(|(_, sc)| key_matches(sc, *key, *modifiers))).map(|(id, _)| (id.clone(), *key == Key::Tab))
        }
        _ => None,
    };
    let mut ran = false;
    loop {
        let next = ctx.input_mut(|i| {
            let (at, hit) = i.events.iter().enumerate().find_map(|(at, e)| Some((at, command(e)?)))?;
            i.events.remove(at);
            Some(hit)
        });
        let Some((id, tab)) = next else { break };
        if tab {
            // egui already read this Tab as "focus the next widget" (Photoshop's Tab hides the
            // panels instead, #1313): the shortcut took it, so cancel that move.
            ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
        }
        let id = clip_key_command(app, delete_key_command(app, id));
        let owner = key_owner(app, ctx);
        dispatch(app, ctx, &id);
        ran = true;
        if key_owner(app, ctx) != owner {
            break;
        }
    }
    ran
}

/// Clear's key (Delete / Backspace) with nothing selected deletes the selected layers, of any
/// kind, as in Photoshop (#1077): Edit › Clear only clears pixel layers, so on an adjustment, fill,
/// type or shape layer the key did nothing. A selection, a single channel or Quick Mask keeps Clear.
/// With the active layer's mask targeted the key edits the mask, never the layer's pixels: a
/// selection is filled with the background colour in the mask (Photoshop's Delete on a mask), and
/// with no selection the mask itself is deleted, as Layer › Layer Mask › Delete, unless several
/// layers are selected: then they are deleted.
fn delete_key_command(app: &PhotocraftApp, id: String) -> String {
    if id != "edit.clear" {
        return id;
    }
    let Some(st) = app.session.active() else { return id };
    let composite = st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::Composite && st.doc.quick_mask.is_none();
    let has_mask = st.active_layer.and_then(|l| st.doc.layer(l)).is_some_and(|l| l.mask.is_some());
    let colors_off = st.channel_view.visible_colors(photocraft_engine::channel_cmds::color_count(&st.doc)) == 0;
    let alpha_on = (0..st.doc.channels.len()).any(|i| st.channel_view.alpha_shown(i));
    let solo_mask = colors_off && has_mask && !alpha_on && st.doc.quick_mask.is_none();
    let layer_mask = st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::LayerMask;
    let mask = (app.ui.mask_target || solo_mask || layer_mask) && has_mask;
    // Several selected layers outrank the active one's targeted mask (#4025): a new adjustment
    // layer targets its mask, so the first press used to delete only that mask.
    let several = st.selected_layers().len() > 1;
    if mask && (composite || layer_mask) && !(several && composite && st.doc.selection.is_none()) {
        let cmd = if st.doc.selection.is_some() { "edit.fillBackground" } else { "layer.layerMask.delete" };
        return cmd.into();
    }
    if st.doc.selection.is_some() || !composite {
        return id;
    }
    "layer.delete".into()
}

/// Create Clipping Mask's key (⌥⌘G) releases when there is nothing left to clip, as in Photoshop,
/// whose Layer menu has one item that reads Create or Release Clipping Mask with the same
/// shortcut: pressing it again takes the clip off. That is when the layers Create Clipping Mask
/// would clip are all clipped already: the selected layer, or with several selected, every one
/// above the lowest.
pub(crate) fn clip_key_releases(app: &PhotocraftApp) -> bool {
    let Some(st) = app.session.active() else { return false };
    let selected = st.selected_layers();
    let clips = match selected.as_slice() {
        [] => return false,
        [one] => std::slice::from_ref(one),
        [_lowest, above @ ..] => above,
    };
    clips.iter().all(|id| st.doc.layer(*id).is_some_and(|l| l.clipped))
}

fn clip_key_command(app: &PhotocraftApp, id: String) -> String {
    if id == "layer.createClippingMask" && clip_key_releases(app) { "layer.releaseClippingMask".into() } else { id }
}

/// Why `id` can't run now (the engine's reason when it has one).
pub fn disabled_reason(app: &PhotocraftApp, id: &str) -> String {
    photocraft_engine::commands::find(id).and_then(|_| app.session.disabled_reason_with(id, &app.with_mask_target(id, serde_json::Value::Null))).unwrap_or_else(
        || match id {
            "tools.decreaseBrushHardness" | "tools.increaseBrushHardness" => "the current tool has no brush tip".into(),
            _ if app.session.active().is_none() => "no document open".into(),
            _ => "not available in the current state".into(),
        },
    )
}

fn label(id: &str) -> String {
    crate::menu_catalog::CATALOG
        .iter()
        .find(|c| c.3 == id)
        .map(|c| c.1)
        .or_else(|| crate::menus::UI_COMMANDS.iter().find(|c| c.0 == id).map(|c| c.1))
        .or_else(|| photocraft_engine::commands::find(id).map(|c| c.label))
        .or_else(|| crate::tool_shortcuts::tool(id).map(|tool| tool.label()))
        .unwrap_or(id)
        .trim_end_matches('…')
        .to_string()
}

/// Run a shortcut's command like its menu item (dialogs included), or say why it can't run.
pub fn dispatch(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str) {
    let action = id.strip_prefix(crate::actions::SHORTCUT_PREFIX);
    let command_id = if action.is_some() { "actions.play" } else { id };
    let outcome = if !crate::menus::is_enabled(app, command_id) {
        let why = disabled_reason(app, command_id);
        let template = "{label} is not available: {reason}";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("label", crate::i18n::tr_id(crate::i18n::current(), id, &label(id))), ("reason", tl!(&why))]);
        app.ui.status_error = true;
        Outcome::Disabled(why)
    } else if ctx.data(|d| d.get_temp::<bool>(dry_run_id())).unwrap_or(false) {
        Outcome::Ran
    } else {
        let r = if let Some(name) = action {
            // Use the normal command path: synthetic input and every nested action
            // step must still pass the automation authorizer.
            crate::menus::invoke(app, ctx, "actions.play", json!({"action": name})).and_then(|v| {
                crate::actions::report_play(app, &v);
                if v.get("failed").is_some_and(|f| f.is_object()) { Err(app.ui.status.clone()) } else { Ok(v) }
            })
        } else if crate::adjust_dialog::has_dialog(id) {
            crate::adjust_dialog::open(app, id);
            Ok(serde_json::Value::Null)
        } else {
            crate::menus::invoke(app, ctx, id, json!({}))
        };
        match r {
            Ok(_) => Outcome::Ran,
            Err(e) => {
                app.ui.status = e.clone();
                app.ui.status_error = true;
                Outcome::Failed(e)
            }
        }
    };
    ctx.data_mut(|d| {
        let log = d.get_temp_mut_or_default::<Vec<(String, Outcome)>>(log_id());
        if log.len() >= 64 {
            log.remove(0);
        }
        log.push((id.to_string(), outcome));
    });
}

#[cfg(test)]
mod tests;
