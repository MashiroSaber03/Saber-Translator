//! #127: every shortcut the UI displays (menu items, shell commands, tool tooltips) dispatches its
//! command, or the item is visibly disabled with a reason — through the whole app, with a
//! realistic layered document, from the places focus usually is.

use egui::accesskit::Role;
use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::LayerContent;
use serde_json::json;

use super::{Focus, Outcome, bindings, set_dry_run, take_log};
use crate::PhotocraftApp;
use crate::shortcuts::parse;
use crate::state::Tool;

/// Background, a filled pixel layer "paint" (active, with a selection), a type layer and a group
/// holding a layer: what a real layout PSD looks like.
fn realistic() -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 240, "height": 160})).unwrap();
    s.execute("layer.new.layer", json!({"name": "in-group"})).unwrap();
    s.execute("layer.groupLayers", json!({"name": "group"})).unwrap();
    s.execute("type.create", json!({"x": 20, "y": 60, "text": "Hello"})).unwrap();
    let paint = s.execute("layer.new.layer", json!({"name": "paint"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("select.rect", json!({"x": 20, "y": 20, "width": 80, "height": 60})).unwrap();
    s.execute("edit.fill", json!({"color": "#3070c0"})).unwrap();
    s.execute("layer.select", json!({"layer": paint})).unwrap();
    s
}

fn services() -> crate::Services {
    // Saving is available in the app (an exporter is installed); nothing is written here.
    crate::Services { export: Some(Box::new(|_, _, _| Err("tests don't write files".into()))), ..Default::default() }
}

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(realistic(), services())
    });
    h.run_steps(8);
    h
}

/// The platform's modifiers for a shortcut's: `Cmd` is ⌘ on the Mac and Ctrl elsewhere, as
/// egui-winit reports them (`command` plus the physical key).
fn platform(m: Modifiers) -> Modifiers {
    let mut out = Modifiers { alt: m.alt, shift: m.shift, ctrl: m.ctrl, ..Default::default() };
    if m.command {
        out.command = true;
        if cfg!(target_os = "macos") {
            out.mac_cmd = true;
        } else {
            out.ctrl = true;
        }
    }
    out
}

fn press(h: &mut Harness<'_, PhotocraftApp>, sc: &str) {
    let sc = parse(sc).unwrap_or_else(|| panic!("unparsable shortcut {sc}"));
    let m = platform(sc.modifiers);
    h.event(egui::Event::ModifiersChanged(m));
    h.event(egui::Event::Key { key: sc.logical_key, physical_key: None, pressed: true, repeat: false, modifiers: m });
    h.event(egui::Event::Key { key: sc.logical_key, physical_key: None, pressed: false, repeat: false, modifiers: m });
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

fn active_name(h: &Harness<'_, PhotocraftApp>) -> String {
    let st = h.state().session.active().unwrap();
    st.active_layer.and_then(|id| st.doc.layer(id)).map(|l| l.name.clone()).unwrap_or_default()
}

/// Where focus is when the shortcut is pressed.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// The pointer over the canvas, nothing focused.
    Canvas,
    /// Just after clicking the "paint" row in the Layers panel.
    LayersRow,
    /// Just after clicking into the Layers panel's Opacity field (a text field now).
    OpacityField,
    /// A non-text widget focused (a Layers panel row, through accessibility focus: Tab from
    /// nothing focused is Photoshop's Show/Hide All Panels, #1313).
    FocusedWidget,
}

fn put_focus(h: &mut Harness<'_, PhotocraftApp>, place: Place) {
    match place {
        Place::Canvas => {
            let c = h.state().last_canvas_rect.center();
            h.hover_at(c);
            h.run_steps(2);
        }
        Place::LayersRow => {
            h.get_by_role_and_label(Role::Button, "paint").scroll_to_me();
            h.run_steps(4);
            let row = h.get_by_role_and_label(Role::Button, "paint").rect();
            click(h, row.center() + vec2(row.width() / 4.0, 0.0));
            assert_eq!(active_name(h), "paint");
        }
        Place::OpacityField => {
            // The Layers panel's Opacity field: the DragValue in the right dock reading 100.
            let layers = crate::dock::last_rects(&h.ctx).into_iter().find(|(g, _)| *g == crate::dock::Group::Layers).expect("Layers drawn").1;
            let at = h
                .query_all_by_role(Role::SpinButton)
                .map(|n| n.rect())
                .filter(|r| layers.contains(r.center()))
                .min_by(|a, b| a.center().y.total_cmp(&b.center().y))
                .expect("Layers opacity field")
                .center();
            click(h, at);
            assert_eq!(Focus::of(&h.ctx), Focus::Text, "clicking a value field edits it");
        }
        Place::FocusedWidget => {
            h.get_by_role_and_label(Role::Button, "paint").scroll_to_me();
            h.run_steps(4);
            h.get_by_role_and_label(Role::Button, "paint").focus();
            h.run_steps(2);
            assert_eq!(Focus::of(&h.ctx), Focus::Widget, "a widget is focused");
        }
    }
    take_log(&h.ctx);
}

/// Is `bound` the same action as the menu item `shown`? (Window › Layers is `window.panel.layers`
/// in the catalogue and `window.toggle.layers` in the shell; Release Clipping Mask shows Create
/// Clipping Mask's key, which releases while it reads so.)
fn same(shown: &str, bound: &str) -> bool {
    shown == bound
        || crate::menus::panel_alias(shown) == Some(bound)
        || crate::menus::panel_alias(bound) == Some(shown)
        || (shown == "layer.releaseClippingMask" && bound == "layer.createClippingMask")
}

/// Every shortcut shown in the menus (and the shell's unlisted ones), with whether it is live.
fn displayed(app: &PhotocraftApp) -> Vec<(String, String, bool)> {
    let mut v: Vec<(String, String, bool)> = crate::menus::menu_items(app)
        .into_iter()
        .filter(|i| i.id != "---")
        .filter_map(|i| Some((i.id.clone(), i.shortcut?, crate::menus::is_live(&i.id))))
        .collect();
    for (id, _, menu, sc) in crate::menus::UI_COMMANDS {
        if let Some(sc) = sc
            && menu.is_empty()
        {
            v.push((id.to_string(), sc.to_string(), true));
        }
    }
    v
}

#[test]
fn every_displayed_shortcut_is_bound_to_its_item() {
    let h = harness();
    let app = h.state();
    let table = bindings(app);
    let mut broken = Vec::new();
    for (id, sc, live) in displayed(app) {
        let parsed = parse(&sc).unwrap_or_else(|| panic!("{id}: unparsable {sc}"));
        let owner = table.iter().find(|(_, b)| *b == parsed).map(|(o, _)| o.clone());
        match owner {
            Some(o) if same(&id, &o) => {}
            _ if !live => {
                // Not implemented yet: shown greyed (the parity report tracks these).
                let item = crate::menus::menu_items(app).into_iter().find(|i| i.id == id);
                assert!(item.is_none_or(|i| !i.enabled), "{id} isn't live but shows enabled");
            }
            o => broken.push(format!("{id} shows {sc} but it is bound to {o:?}")),
        }
    }
    assert!(broken.is_empty(), "{} displayed shortcuts don't reach their item:\n{}", broken.len(), broken.join("\n"));
}

/// Press every displayed shortcut from each place focus usually is: each dispatches its command,
/// or the menu shows it disabled and the press reports why. (Dry run: File › Exit, Open, Help
/// links… resolve and check enablement without running.)
#[test]
fn every_displayed_shortcut_dispatches_from_where_focus_usually_is() {
    let mut failures = Vec::new();
    for place in [Place::Canvas, Place::LayersRow, Place::FocusedWidget, Place::OpacityField] {
        let mut h = harness();
        set_dry_run(&h.ctx, true);
        put_focus(&mut h, place);
        let focus = Focus::of(&h.ctx);
        let items = crate::menus::menu_items(h.state());
        for (id, sc, live) in displayed(h.state()) {
            if !live {
                continue;
            }
            let parsed = parse(&sc).unwrap();
            press(&mut h, &sc);
            let log = take_log(&h.ctx);
            if !focus.allows(&parsed) {
                // A focused text field keeps typing / editing keys: nothing may fire.
                if !log.is_empty() {
                    failures.push(format!("{place:?}: {sc} ({id}) fired {log:?} into a focused field"));
                }
                continue;
            }
            let enabled = items.iter().find(|i| i.id == id).map_or_else(|| crate::menus::is_enabled(h.state(), &id), |i| i.enabled);
            match log.as_slice() {
                [(bound, Outcome::Ran)] if same(&id, bound) && enabled => {}
                [(bound, Outcome::Disabled(why))] if same(&id, bound) && !enabled && !why.is_empty() => {}
                other => failures.push(format!("{place:?}: {sc} ({id}, menu enabled: {enabled}) -> {other:?}")),
            }
        }
        // Tool letters from the tools panel tooltips ("Brush Tool  (B)").
        if !matches!(focus, Focus::Text) {
            for t in Tool::ALL.into_iter().filter(|t| t.key() != '\0') {
                h.state_mut().ui.tool = if t.key() == 'H' { Tool::Move } else { Tool::Hand };
                press(&mut h, &t.key().to_string());
                if h.state().ui.tool.key() != t.key() {
                    failures.push(format!("{place:?}: tool key {} left {:?}", t.key(), h.state().ui.tool));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} shortcut failures:\n{}", failures.len(), failures.join("\n"));
}

/// The common shortcuts really run (not a dry run) in the layered document after clicking around
/// the panels: the user report was "shortcut keys don't do anything".
#[test]
fn common_shortcuts_work_after_clicking_panels() {
    for place in [Place::Canvas, Place::LayersRow, Place::FocusedWidget, Place::OpacityField] {
        let mut h = harness();
        put_focus(&mut h, place);
        let count = |h: &Harness<'_, PhotocraftApp>| h.state().session.active().unwrap().doc.walk().len();
        let n = count(&h);
        press(&mut h, "Cmd+J");
        assert_eq!(count(&h), n + 1, "{place:?}: ⌘J duplicates the layer");
        // In a focused text field ⌘Z / ⇧⌘Z undo the typing, as in Photoshop.
        let text = matches!(place, Place::OpacityField);
        if !text {
            press(&mut h, "Cmd+Z");
            assert_eq!(count(&h), n, "{place:?}: ⌘Z undoes");
            press(&mut h, "Cmd+Shift+Z");
            assert_eq!(count(&h), n + 1, "{place:?}: ⇧⌘Z redoes");
            press(&mut h, "Cmd+Z");
        }
        press(&mut h, "Cmd+D");
        assert!(h.state().session.active().unwrap().doc.selection.is_none(), "{place:?}: ⌘D deselects");
        press(&mut h, "Cmd+Shift+D");
        assert!(h.state().session.active().unwrap().doc.selection.is_some(), "{place:?}: ⇧⌘D reselects");
        let z = h.state().current_zoom();
        press(&mut h, "Cmd+=");
        assert!(h.state().current_zoom() > z, "{place:?}: ⌘+ zooms in");
        let rulers = h.state().ui.extras.rulers;
        press(&mut h, "Cmd+R");
        assert_ne!(h.state().ui.extras.rulers, rulers, "{place:?}: ⌘R toggles rulers");
        let layers = h.state().ui.panels.layers;
        press(&mut h, "F7");
        assert_ne!(h.state().ui.panels.layers, layers, "{place:?}: F7 toggles Layers");
        press(&mut h, "F7");
        // Dialogs: Image Size (⌥⌘I, shown in the menu but unbound before #127) and Levels (⌘L).
        for (sc, what) in [("Cmd+Alt+I", "Image Size"), ("Cmd+Alt+C", "Canvas Size"), ("Cmd+L", "Levels"), ("Cmd+M", "Curves")] {
            press(&mut h, sc);
            assert_eq!(h.state().ui.dialogs.len(), 1, "{place:?}: {sc} opens {what}");
            h.state_mut().ui.dialogs.clear();
            h.run_steps(2);
        }
        if !text {
            // From another group (B again would cycle to the Pencil).
            h.state_mut().ui.tool = Tool::Move;
            press(&mut h, "B");
            assert_eq!(h.state().ui.tool, Tool::Brush, "{place:?}: B picks the Brush");
            press(&mut h, "V");
            assert_eq!(h.state().ui.tool, Tool::Move, "{place:?}: V picks Move");
            // The colour keys from the tools panel tooltips: X swaps, D resets.
            let fg = h.state().session.tools.foreground;
            press(&mut h, "X");
            assert_eq!(h.state().session.tools.background, fg, "{place:?}: X swaps colours");
            press(&mut h, "D");
            assert_eq!(h.state().session.tools.foreground, [0.0, 0.0, 0.0, 1.0], "{place:?}: D resets colours");
        }
        press(&mut h, "Cmd+G");
        let group = h.state().session.active().unwrap().active_layer.and_then(|id| h.state().session.active().unwrap().doc.layer(id).cloned());
        assert!(group.is_some_and(|l| matches!(l.content, LayerContent::Group(_))), "{place:?}: ⌘G groups");
    }
}

#[test]
fn a_disabled_shortcut_says_why() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    // Merge Down on the bottom layer: Photoshop greys it; the key press reports the reason.
    let bg = h.state().session.active().unwrap().doc.layers[0].id;
    h.state_mut().session.select_layer(bg).unwrap();
    h.run_steps(2);
    press(&mut h, "Cmd+E");
    let log = take_log(&h.ctx);
    assert!(matches!(log.as_slice(), [(id, Outcome::Disabled(_) | Outcome::Failed(_))] if id == "layer.mergeLayers"), "{log:?}");
    assert!(!h.state().ui.status.is_empty() && h.state().ui.status_error, "the status bar says why");
}

#[test]
fn a_disabled_shortcut_uses_the_command_id_translation() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    crate::i18n::with_language(crate::i18n::Lang::from_code("cs").unwrap(), || {
        super::dispatch(&mut app, &egui::Context::default(), "select.all");
        assert!(app.ui.status.starts_with("Vybrat vše není k dispozici:"), "{}", app.ui.status);
        assert!(app.ui.status_error);
    });
}

/// #352: ⌥[ ⌥] ⌥, ⌥. and ⇧⌥[ walk the Layers panel's rows from the keyboard.
#[test]
fn alt_brackets_walk_the_layers_panel() {
    let mut h = harness();
    // Rows, top to bottom: paint, Hello (type), group, in-group, Background.
    assert_eq!(active_name(&h), "paint");
    for (key, want) in [("Alt+[", "Hello"), ("Alt+[", "group"), ("Alt+[", "in-group"), ("Alt+]", "group"), ("Alt+,", "Background"), ("Alt+.", "paint")] {
        press(&mut h, key);
        assert_eq!(active_name(&h), want, "after {key}");
    }
    press(&mut h, "Alt+Shift+[");
    assert_eq!(active_name(&h), "Hello");
    assert_eq!(h.state().session.active().unwrap().selected_layers().len(), 2, "⇧⌥[ adds to the selection");
}

/// #626: [ ] ⇧[ ⇧] are commands listed under Tools in Edit › Keyboard Shortcuts, so they rebind.
#[test]
fn brush_keys_can_be_rebound() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    let items = crate::prefs_ui::shortcut_items(h.state());
    let listed = |id: &str| items.iter().find(|i| i.0 == id).map(|i| (i.1.clone(), i.2.clone(), i.3.clone()));
    assert_eq!(listed("tools.decreaseBrushSize"), Some(("Decrease Brush Size".into(), vec!["Tools".into()], Some("[".into()))));
    assert_eq!(listed("tools.increaseBrushHardness").and_then(|i| i.2), Some("Shift+]".into()));
    let app = h.state_mut();
    app.ui.tool = Tool::Brush;
    app.run("tools.setBrush", json!({"brush": {"size": 40, "hardness": 0.5}})).unwrap();
    app.run("edit.keyboardShortcuts", json!({"set": {"tools.increaseBrushSize": "Alt+W", "tools.increaseBrushHardness": "Alt+Shift+W"}})).unwrap();
    let brush = |h: &Harness<'_, PhotocraftApp>| (h.state().session.tools.brush.size, h.state().session.tools.brush.hardness);
    press(&mut h, "]");
    press(&mut h, "Shift+]");
    assert_eq!(brush(&h), (40.0, 0.5), "the old keys are free");
    press(&mut h, "Alt+W");
    press(&mut h, "Alt+Shift+W");
    assert_eq!(brush(&h), (50.0, 0.75), "the new keys step size and hardness");
    press(&mut h, "[");
    press(&mut h, "Shift+[");
    assert_eq!(brush(&h), (40.0, 0.5), "the others keep their defaults");
    // Hardness only steps for a tool with a brush tip; the press says why it did nothing.
    h.state_mut().ui.tool = Tool::Move;
    take_log(&h.ctx);
    press(&mut h, "Alt+Shift+W");
    assert_eq!(brush(&h), (40.0, 0.5));
    assert!(matches!(take_log(&h.ctx).as_slice(), [(id, Outcome::Disabled(_))] if id == "tools.increaseBrushHardness"));
}

/// Bytes the app saved, newest last.
type Saved = std::rc::Rc<std::cell::RefCell<Vec<Vec<u8>>>>;

/// [`harness`] with a document that saves in place, an exporter writing its top-level layer count
/// and a writer recording it; then one new layer on top (history to undo).
fn saving_harness() -> (Harness<'static, PhotocraftApp>, Saved) {
    let saved = Saved::default();
    let out = saved.clone();
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let services = crate::Services {
            export: Some(Box::new(|doc, _, _| Ok((doc.layers.len().to_string().into_bytes(), Vec::new())))),
            write: Some(Box::new(move |_, bytes| {
                out.borrow_mut().push(bytes.to_vec());
                Ok(())
            })),
            ..Default::default()
        };
        let mut session = realistic();
        session.active_mut().unwrap().path = Some("/tmp/layout.psd".into());
        session.execute("layer.new.layer", json!({"name": "added"})).unwrap();
        PhotocraftApp::new(session, services)
    });
    h.run_steps(8);
    put_focus(&mut h, Place::Canvas);
    (h, saved)
}

fn layer_count(h: &Harness<'_, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().doc.layers.len()
}

/// Press `shortcuts` one after another within a single frame (`Harness::event` runs a frame per
/// event).
fn press_in_one_frame(h: &mut Harness<'_, PhotocraftApp>, shortcuts: &[&str]) {
    let events = &mut h.input_mut().events;
    for sc in shortcuts {
        let sc = parse(sc).unwrap();
        let m = platform(sc.modifiers);
        events.push(egui::Event::ModifiersChanged(m));
        events.push(egui::Event::Key { key: sc.logical_key, physical_key: None, pressed: true, repeat: false, modifiers: m });
        events.push(egui::Event::Key { key: sc.logical_key, physical_key: None, pressed: false, repeat: false, modifiers: m });
    }
    events.push(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
}

fn logged(h: &Harness<'_, PhotocraftApp>) -> Vec<String> {
    take_log(&h.ctx).into_iter().map(|(id, _)| id).collect()
}

/// #440: shortcuts arriving in one frame run in the order they were pressed.
#[test]
fn shortcuts_in_one_frame_run_in_order() {
    let (mut h, saved) = saving_harness();
    assert_eq!(layer_count(&h), 5);
    // ⌘Z then ⌘S: the undone document is saved.
    press_in_one_frame(&mut h, &["Cmd+Z", "Cmd+S"]);
    assert_eq!(layer_count(&h), 4, "⌘Z undid the new layer");
    assert_eq!(saved.borrow().as_slice(), [b"4".to_vec()], "⌘S saved after the undo");
    assert_eq!(logged(&h), ["edit.undo", "file.save"]);
    // ⇧⌘Z redoes; then ⌘S before ⌘Z saves first and undoes after.
    press_in_one_frame(&mut h, &["Cmd+Shift+Z"]);
    assert_eq!(layer_count(&h), 5);
    logged(&h);
    saved.borrow_mut().clear();
    press_in_one_frame(&mut h, &["Cmd+S", "Cmd+Z"]);
    assert_eq!(saved.borrow().as_slice(), [b"5".to_vec()], "saved before the undo");
    assert_eq!(layer_count(&h), 4, "then undone");
    assert_eq!(logged(&h), ["file.save", "edit.undo"]);
}

/// A shortcut that opens a dialog leaves the frame's later keys to it: ⌘L then ⌘Z doesn't undo
/// behind the Levels dialog.
#[test]
fn a_shortcut_opening_a_dialog_takes_the_frames_later_keys() {
    let (mut h, _) = saving_harness();
    press_in_one_frame(&mut h, &["Cmd+L", "Cmd+Z"]);
    assert_eq!(h.state().ui.dialogs.len(), 1, "⌘L opened Levels");
    assert_eq!(layer_count(&h), 5, "⌘Z didn't undo behind the dialog");
    assert_eq!(logged(&h), ["image.adjustments.levels"]);
}

fn names(h: &Harness<'_, PhotocraftApp>) -> Vec<String> {
    h.state().session.active().unwrap().doc.walk().into_iter().map(|(_, _, l)| l.name.clone()).collect()
}

/// #1077: with nothing selected, Delete (and Backspace) deletes the selected layer whatever its
/// kind, from the canvas or the Layers panel, in one undo step; it used to say Clear needs a
/// pixel layer on an adjustment layer.
#[test]
fn delete_without_a_selection_deletes_the_selected_layer() {
    for place in [Place::Canvas, Place::LayersRow, Place::FocusedWidget] {
        for key in ["Delete", "Backspace"] {
            let mut h = harness();
            put_focus(&mut h, place);
            let s = &mut h.state_mut().session;
            s.execute("select.deselect", json!({})).unwrap();
            let adj = photocraft_doc::LayerId(s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap());
            h.run_steps(2);
            let before = names(&h);
            press(&mut h, key);
            assert_eq!(logged(&h), ["layer.delete"], "{place:?} {key}");
            assert!(h.state().session.active().unwrap().doc.layer(adj).is_none(), "{place:?} {key}: the adjustment layer is gone");
            assert_eq!(names(&h).len(), before.len() - 1, "{place:?} {key}: only it");
            press(&mut h, "Cmd+Z");
            assert_eq!(names(&h), before, "{place:?} {key}: one undo brings it back");
        }
    }
}

/// With the layer mask targeted, Delete / Backspace edit the mask and never the layer's pixels:
/// a selection fills the mask with the background colour there, and with no selection the mask
/// itself is deleted.
#[test]
fn delete_with_the_mask_targeted_edits_the_mask_not_the_layer() {
    for key in ["Delete", "Backspace"] {
        let mut h = harness();
        put_focus(&mut h, Place::Canvas);
        let s = &mut h.state_mut().session;
        let layer = s.active().unwrap().active_layer.unwrap();
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        let sel = s.active().unwrap().doc.selection.as_ref().expect("the harness keeps a selection").content_bounds();
        let (x, y) = ((sel.x0 + sel.x1) / 2, (sel.y0 + sel.y1) / 2);
        h.state_mut().ui.mask_target = true;
        h.run_steps(2);
        let layer_pixels = |h: &Harness<'_, PhotocraftApp>| {
            h.state().session.active().unwrap().doc.layer(layer).and_then(|l| l.surface().cloned()).map(|s| s.read_region(sel))
        };
        let mask_at = |h: &Harness<'_, PhotocraftApp>| {
            h.state().session.active().unwrap().doc.layer(layer).and_then(|l| l.mask.as_ref()).map(|m| m.surface.sample_channel(x, y, 0))
        };
        let before = layer_pixels(&h);
        let bg = h.state().session.tools.background;
        press(&mut h, key);
        assert_eq!(logged(&h), ["edit.fillBackground"], "{key}");
        assert_eq!(mask_at(&h), Some(bg[0]), "{key}: the mask takes the background colour in the selection");
        assert_eq!(layer_pixels(&h), before, "{key}: the layer's pixels are untouched");
        h.state_mut().session.execute("select.deselect", json!({})).unwrap();
        h.run_steps(2);
        press(&mut h, key);
        assert_eq!(logged(&h), ["layer.layerMask.delete"], "{key}");
        assert_eq!(mask_at(&h), None, "{key}: no selection deletes the mask");
        assert!(h.state().session.active().unwrap().doc.layer(layer).is_some(), "{key}: the layer stays");
        assert_eq!(layer_pixels(&h), before, "{key}: the layer's pixels are untouched");
    }
}

/// Every selected layer goes, pixel layers too (Photoshop deletes, it doesn't clear the layer).
#[test]
fn delete_without_a_selection_deletes_every_selected_layer() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    let s = &mut h.state_mut().session;
    s.execute("select.deselect", json!({})).unwrap();
    let paint = s.active().unwrap().active_layer.unwrap();
    let adj = photocraft_doc::LayerId(s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.select", json!({"layer": paint.0})).unwrap();
    s.execute("layer.select", json!({"layer": adj.0, "mode": "add"})).unwrap();
    h.run_steps(2);
    let n = names(&h).len();
    press(&mut h, "Delete");
    let doc = &h.state().session.active().unwrap().doc;
    assert!(doc.layer(paint).is_none() && doc.layer(adj).is_none(), "both selected layers are gone");
    assert_eq!(names(&h).len(), n - 2);
}

/// #4025: with several layers selected and no selection, a targeted mask on the active one doesn't
/// take the key: Delete removes the selected layers in one press, as it does with the layer
/// targeted. A selection still fills the targeted mask.
#[test]
fn delete_with_several_layers_selected_deletes_them_even_with_the_mask_targeted() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    let s = &mut h.state_mut().session;
    let first = photocraft_doc::LayerId(s.execute("layer.newAdjustmentLayer.curves", json!({})).unwrap()["layer"].as_u64().unwrap());
    let second = photocraft_doc::LayerId(s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.select", json!({"layer": first.0, "mode": "add"})).unwrap();
    let st = s.active().unwrap();
    assert_eq!(st.selected_layers().len(), 2);
    assert!(st.active_layer.and_then(|l| st.doc.layer(l)).is_some_and(|l| l.mask.is_some()), "the active adjustment layer has a mask");
    assert!(st.doc.selection.is_some(), "the harness keeps a selection");
    h.state_mut().ui.mask_target = true;
    h.run_steps(2);
    let n = names(&h).len();
    press(&mut h, "Delete");
    assert_eq!(logged(&h), ["edit.fillBackground"], "a selection edits the mask");
    assert_eq!(names(&h).len(), n);
    h.state_mut().session.execute("select.deselect", json!({})).unwrap();
    h.run_steps(2);
    press(&mut h, "Delete");
    assert_eq!(logged(&h), ["layer.delete"]);
    let doc = &h.state().session.active().unwrap().doc;
    assert!(doc.layer(first).is_none() && doc.layer(second).is_none(), "both selected layers are gone");
    assert_eq!(names(&h).len(), n - 2);
}

/// A selection keeps Edit › Clear: the selected pixels go, the layer stays.
#[test]
fn delete_with_a_selection_clears_it() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    let before = names(&h);
    assert!(h.state().session.active().unwrap().doc.selection.is_some());
    press(&mut h, "Delete");
    assert_eq!(logged(&h), ["edit.clear"]);
    assert_eq!(names(&h), before, "no layer was deleted");
}

fn is_clipped(h: &Harness<'_, PhotocraftApp>, id: photocraft_doc::LayerId) -> bool {
    h.state().session.active().unwrap().doc.layer(id).unwrap().clipped
}

/// The Layer menu's clipping mask items: (id, label, shortcut, enabled).
fn clip_items(h: &Harness<'_, PhotocraftApp>) -> Vec<(String, String, Option<String>, bool)> {
    crate::menus::menu_items(h.state()).into_iter().filter(|i| i.id.ends_with("ClippingMask")).map(|i| (i.id, i.label, i.shortcut, i.enabled)).collect()
}

fn clip_item(id: &str, label: &str) -> Vec<(String, String, Option<String>, bool)> {
    vec![(id.into(), label.into(), Some("Cmd+Alt+G".into()), true)]
}

/// ⌥⌘G clips the selected layer and pressing it again releases it, as Photoshop's one Create /
/// Release Clipping Mask item does; the Layer menu reads Release Clipping Mask, with the same
/// shortcut, while the key would release.
#[test]
fn clip_key_toggles_the_clipping_mask() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    let paint = h.state().session.active().unwrap().active_layer.unwrap();
    assert_eq!(clip_items(&h), clip_item("layer.createClippingMask", "Create Clipping Mask"));
    press(&mut h, "Cmd+Alt+G");
    assert_eq!(logged(&h), ["layer.createClippingMask"]);
    assert!(is_clipped(&h, paint));
    assert_eq!(clip_items(&h), clip_item("layer.releaseClippingMask", "Release Clipping Mask"));
    press(&mut h, "Cmd+Alt+G");
    assert_eq!(logged(&h), ["layer.releaseClippingMask"]);
    assert!(!is_clipped(&h, paint), "the second press releases it");
    assert_eq!(clip_items(&h), clip_item("layer.createClippingMask", "Create Clipping Mask"));
    press(&mut h, "Cmd+Z");
    assert!(is_clipped(&h, paint), "one undo puts the clip back");
}

/// With several layers selected the key clips every one above the lowest, and releases them once
/// they all are: a selection where only some are clipped clips the rest first.
#[test]
fn clip_key_toggles_several_selected_layers() {
    let mut h = harness();
    put_focus(&mut h, Place::Canvas);
    let s = &mut h.state_mut().session;
    let paint = s.active().unwrap().active_layer.unwrap();
    let base = s.active().unwrap().doc.walk().into_iter().find(|(_, _, l)| l.name == "Hello").unwrap().2.id;
    let top = photocraft_doc::LayerId(s.execute("layer.new.layer", json!({"name": "top"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.createClippingMask", json!({"layer": paint.0})).unwrap();
    s.execute("layer.select", json!({"layer": base.0, "mode": "replace"})).unwrap();
    for id in [paint, top] {
        s.execute("layer.select", json!({"layer": id.0, "mode": "add"})).unwrap();
    }
    assert_eq!(s.active().unwrap().selected_layers().len(), 3);
    h.run_steps(2);
    press(&mut h, "Cmd+Alt+G");
    assert_eq!(logged(&h), ["layer.createClippingMask"], "the top layer isn't clipped yet");
    assert!(is_clipped(&h, paint) && is_clipped(&h, top) && !is_clipped(&h, base));
    press(&mut h, "Cmd+Alt+G");
    assert_eq!(logged(&h), ["layer.releaseClippingMask"]);
    assert!(!is_clipped(&h, paint) && !is_clipped(&h, top) && !is_clipped(&h, base));
}

#[test]
fn named_action_function_key_replays_and_survives_preferences_roundtrip() {
    let mut h = harness();
    let action = photocraft_engine::actions_cmds::Action {
        name: "LivePrint 2R".into(),
        steps: vec![("layer.new.layer".into(), json!({"name": "F6 overlay"}))],
        disabled: Default::default(),
        ..Default::default()
    };
    h.state_mut().session.actions.list.push(action);
    crate::actions::assign_shortcut(h.state_mut(), "LivePrint 2R", "F6").unwrap();
    let saved = serde_json::to_value(h.state().session.prefs()).unwrap();
    let restored = serde_json::from_value(saved).unwrap();
    h.state_mut().session.edit_prefs(|p| *p = restored);
    h.state_mut().ui.actions.selected = None;
    press(&mut h, "F6");
    assert_eq!(active_name(&h), "F6 overlay");
    assert_eq!(h.state().ui.status, "Played 1 step");
    assert!(!h.state().ui.status_error);
    assert_eq!(take_log(&h.ctx), vec![(crate::actions::shortcut_id("LivePrint 2R"), Outcome::Ran)]);
    h.state_mut().run("actions.delete", json!({"action": "LivePrint 2R"})).unwrap();
    assert!(bindings(h.state()).iter().all(|(id, _)| !id.starts_with(crate::actions::SHORTCUT_PREFIX)));
    assert!(bindings(h.state()).iter().any(|(_, sc)| *sc == parse("F6").unwrap()));
}

#[test]
fn assigning_action_function_key_moves_it_and_reports_failed_steps() {
    let mut h = harness();
    for name in ["first", "second"] {
        h.state_mut().session.actions.list.push(photocraft_engine::actions_cmds::Action {
            name: name.into(),
            steps: vec![("no.such.command".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
    }
    crate::actions::assign_shortcut(h.state_mut(), "first", "F6").unwrap();
    crate::actions::assign_shortcut(h.state_mut(), "second", "F6").unwrap();
    assert_eq!(h.state().session.prefs().shortcuts[&crate::actions::shortcut_id("first")], "");
    h.state_mut().ui.actions.selected = Some(0);
    press(&mut h, "F6");
    assert!(h.state().ui.status_error);
    assert!(h.state().ui.status.contains("Step 1 (no.such.command) failed"));
    assert!(matches!(&take_log(&h.ctx)[..], [(id, Outcome::Failed(_))] if id == "actions.play:second"));
}

#[test]
fn action_function_key_checks_automation_authorization() {
    let mut h = harness();
    h.state_mut().session.actions.list.push(photocraft_engine::actions_cmds::Action {
        name: "restricted".into(),
        steps: vec![("layer.new.layer".into(), json!({"name": "should not exist"}))],
        disabled: Default::default(),
        ..Default::default()
    });
    crate::actions::assign_shortcut(h.state_mut(), "restricted", "F6").unwrap();
    h.state_mut().services.automation_command = Some(Box::new(|id, _| if id == "actions.play" { Err("denied action playback".into()) } else { Ok(()) }));
    h.state_mut().automation_input = true;
    super::dispatch(h.state_mut(), &egui::Context::default(), "actions.play:restricted");
    assert_eq!(h.state().ui.status, "denied action playback");
    assert_eq!(active_name(&h), "paint");
}

#[test]
fn fit_shortcut_records_and_f6_fits_the_resized_canvas() {
    let mut h = harness();
    h.state_mut().run("actions.record", json!({"name": "Resize and fit"})).unwrap();
    h.state_mut().run("image.canvasSize", json!({"width": 1800, "height": 1080})).unwrap();
    press(&mut h, "Cmd+0");
    h.state_mut().run("actions.stop", json!({})).unwrap();
    assert_eq!(h.state().session.actions.list[0].steps.iter().map(|s| s.0.as_str()).collect::<Vec<_>>(), ["image.canvasSize", "view.fitOnScreen"]);
    let item = crate::menus::menu_items(h.state()).into_iter().find(|i| i.id == "view.fitOnScreen").unwrap();
    assert_eq!(item.shortcut.as_deref(), Some("Cmd+0"));
    crate::actions::assign_shortcut(h.state_mut(), "Resize and fit", "F6").unwrap();
    h.state_mut().ui.views[0].zoom = 0.1;
    h.state_mut().ui.views[0].center = [0.0, 0.0];
    press(&mut h, "F6");
    let view = &h.state().ui.views[0];
    assert!(!view.fit_pending, "the real canvas frame consumed Fit on Screen");
    assert!(view.zoom > 0.1 && view.zoom < 1.0, "{}", view.zoom);
    assert_eq!(view.center, [900.0, 540.0]);
    assert_eq!(h.state().ui.status, "Played 2 steps");
}

#[test]
fn f6_while_recording_records_a_named_call_not_the_child_steps() {
    let mut h = harness();
    h.state_mut().session.actions.list.push(photocraft_engine::actions_cmds::Action {
        name: "LivePrint 2R".into(),
        steps: vec![("layer.new.layer".into(), json!({"name":"Nested overlay"})), ("view.fitOnScreen".into(), json!({}))],
        disabled: Default::default(),
        ..Default::default()
    });
    crate::actions::assign_shortcut(h.state_mut(), "LivePrint 2R", "F6").unwrap();
    h.state_mut().run("actions.record", json!({"name":"Action 2"})).unwrap();
    press(&mut h, "F6");
    assert_eq!(active_name(&h), "Nested overlay");
    h.state_mut().run("actions.stop", json!({})).unwrap();
    assert_eq!(h.state().session.actions.list[1].steps, [("actions.play".into(), json!({"action":"LivePrint 2R"}))]);
}

/// ⇧⌘> / ⇧⌘< follow the keys' position: a Nordic or German layout reports ⇧⌘. as `:`, which is
/// also ⇧⌘; (View › Snap) on a US layout. While typing they size only the selected characters.
#[test]
fn type_size_keys_follow_key_position_and_the_typed_selection() {
    let mut h = harness();
    let st = h.state().session.active().unwrap();
    let id = st.doc.walk().into_iter().find(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).unwrap().2.id.0;
    h.state_mut().session.execute("layer.select", json!({"layer": id})).unwrap();
    let sizes = |h: &Harness<'_, PhotocraftApp>| match &h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
        LayerContent::Text(t) => t.char_runs().iter().map(|r| (r.len, r.style.size_pt)).collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let positional = |h: &mut Harness<'_, PhotocraftApp>, key, physical| {
        let m = platform(Modifiers::COMMAND | Modifiers::SHIFT);
        h.event(egui::Event::ModifiersChanged(m));
        h.event(egui::Event::Key { key, physical_key: Some(physical), pressed: true, repeat: false, modifiers: m });
        h.event(egui::Event::Key { key, physical_key: Some(physical), pressed: false, repeat: false, modifiers: m });
        h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
        h.run_steps(2);
    };
    let base = sizes(&h)[0].1;
    positional(&mut h, egui::Key::Colon, egui::Key::Period);
    assert_eq!(logged(&h), ["type.increaseSize"]);
    assert_eq!(sizes(&h), [(5, base + 2.0)], "the whole layer grows from the canvas");

    crate::type_tool::edit_active(h.state_mut()).unwrap();
    if let Some(ed) = h.state_mut().ui.text_edit.as_mut() {
        (ed.anchor, ed.caret) = (0, 2);
    }
    positional(&mut h, egui::Key::Semicolon, egui::Key::Comma);
    assert_eq!(logged(&h), ["type.decreaseSize"]);
    assert_eq!(sizes(&h), [(2, base), (3, base + 2.0)], "only the selected \"He\" shrinks");
}
