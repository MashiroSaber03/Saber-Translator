//! #1276: a document tab strip with more tabs than fit keeps the active document's tab on the
//! strip, shrinks and elides the others, and moves the rest into a » menu (whole app, real tree).

use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use crate::PhotocraftApp;
use crate::theme::ThemeKind;

use super::{TabStrip, pro_tab_title};

/// A real app window with `docs` documents named "ch1-pNN.png".
fn harness(docs: usize, theme: ThemeKind) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, theme);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        // The theme is an interface preference, applied on the first frame: set the preference.
        app.run("prefs.set", json!({"values": {"interface.theme": theme.id()}})).unwrap();
        for n in 1..=docs {
            app.run("file.new", json!({"width": 400, "height": 300, "name": format!("ch1-p{n:02}.png")})).unwrap();
        }
        app
    });
    h.run_steps(4);
    h
}

fn strip(h: &Harness<'_, PhotocraftApp>) -> TabStrip {
    h.state().tab_strip.clone().expect("the strip is drawn")
}

fn has_menu(h: &Harness<'_, PhotocraftApp>) -> bool {
    h.query_by_label("More documents").is_some()
}

#[test]
fn overflowing_tabs_stay_on_the_strip_and_the_rest_move_to_the_menu() {
    let h = harness(14, ThemeKind::Pro);
    assert_eq!(h.state().session.documents().len(), 14);
    let s = strip(&h);
    assert!(s.tabs.len() < 14, "some tabs must overflow into the menu: {} shown", s.tabs.len());
    for (_, r) in &s.tabs {
        assert!(s.rect.contains_rect(*r), "a tab runs past the strip: {r:?} in {:?}", s.rect);
    }
    assert!(has_menu(&h), "the strip carries a » menu while tabs are hidden");
    // The active document is the first thing the fit keeps: its tab is on the strip, not behind
    // the menu, however many tabs are open (#1276).
    let title = pro_tab_title(h.state(), h.state().session.active_index().unwrap());
    assert!(h.query_by_label(&title).is_some(), "the active tab is on the strip: {title}");
}

#[test]
fn the_active_document_brings_its_tab_onto_the_strip() {
    let mut h = harness(14, ThemeKind::Pro);
    for pick in [0, 13, 7] {
        h.state_mut().session.set_active(pick);
        h.run_steps(2);
        let title = pro_tab_title(h.state(), pick);
        assert!(h.query_by_label(&title).is_some(), "tab {pick} is focused on the strip: {title}");
    }
}

#[test]
fn tabs_that_fit_keep_their_natural_widths_and_no_menu() {
    let h = harness(3, ThemeKind::Pro);
    let s = strip(&h);
    assert_eq!(s.tabs.len(), 3, "every tab stays on the strip when they fit");
    assert!(h.query_by_label("More documents").is_none(), "no overflow menu when the tabs fit");
    for i in 0..3 {
        assert!(h.query_by_label(&pro_tab_title(h.state(), i)).is_some(), "tab {i} is on the strip");
    }
}

#[test]
fn studio_tabs_overflow_into_the_same_menu() {
    let h = harness(14, ThemeKind::Studio);
    let s = strip(&h);
    assert!(s.tabs.len() < 14, "{} shown", s.tabs.len());
    for (_, r) in &s.tabs {
        assert!(s.rect.contains_rect(*r), "a tab runs past the strip: {r:?} in {:?}", s.rect);
    }
    assert!(has_menu(&h), "the Studio strip carries a » menu too");
    let active = h.state().session.active_index().unwrap();
    let name = h.state().session.documents()[active].doc.name.clone();
    assert!(h.query_by_label(&name).is_some(), "the active tab is on the strip: {name}");
}

#[test]
fn saved_document_tabs_track_file_identity() {
    for theme in ThemeKind::ALL {
        let mut h = harness(2, theme);
        h.state_mut().services.export = Some(Box::new(|_, _, _| Ok((vec![1], Vec::new()))));
        h.state_mut().services.write = Some(Box::new(|_, _| Ok(())));
        h.state_mut().save_as(Some("folder/My saved image.psd".into())).unwrap();
        h.run_steps(2);
        let pro = crate::theme::Tokens::get(&h.ctx).pro;
        let app = h.state();
        let label = if pro { pro_tab_title(app, 1) } else { "My saved image.psd".into() };
        assert!(h.query_by_label(&label).is_some(), "saved tab in {theme:?}: {label}");
        assert_eq!(app.session.active().unwrap().doc.name, "My saved image.psd");
        h.state_mut().session.set_active(0);
        h.run_steps(2);
        assert_eq!(h.state().session.active().unwrap().doc.name, "ch1-p01.png");
        h.state_mut().session.set_active(1);
        h.state_mut().run("layer.new.layer", json!({})).unwrap();
        h.run_steps(2);
        let label = if pro { pro_tab_title(h.state(), 1) } else { "My saved image.psd".into() };
        assert!(h.query_by_label(&label).is_some());
        assert_eq!(h.state().session.active().unwrap().doc.name, "My saved image.psd");
        assert!(h.state().session.active().unwrap().is_dirty());
    }
}

/// What a Ctrl press reports: on Windows and Linux Ctrl is also the command key, on macOS it isn't.
fn ctrl() -> egui::Modifiers {
    egui::Modifiers { ctrl: true, command: !cfg!(target_os = "macos"), ..egui::Modifiers::NONE }
}

/// #2340: Ctrl+Tab shows the next document's tab and Ctrl+Shift+Tab the previous one, wrapping
/// around at either end, as in Photoshop. Plain Tab still hides the panels instead.
#[test]
fn ctrl_tab_cycles_through_the_document_tabs() {
    let mut h = harness(3, ThemeKind::Pro);
    let active = |h: &Harness<'_, PhotocraftApp>| h.state().session.active_index();
    assert_eq!(active(&h), Some(2), "the last new document is active");
    let press = |h: &mut Harness<'_, PhotocraftApp>, shift: bool| {
        h.key_press_modifiers(if shift { ctrl() | egui::Modifiers::SHIFT } else { ctrl() }, egui::Key::Tab);
        h.run_steps(2);
    };
    press(&mut h, false);
    assert_eq!(active(&h), Some(0), "Ctrl+Tab on the last tab wraps to the first");
    press(&mut h, false);
    assert_eq!(active(&h), Some(1));
    press(&mut h, true);
    assert_eq!(active(&h), Some(0), "Ctrl+Shift+Tab goes back");
    press(&mut h, true);
    assert_eq!(active(&h), Some(2), "and wraps to the last tab");
    assert!(h.state().ui.panels.dock, "Ctrl+Tab doesn't hide the panels");
    // Tab order is document order, so a dragged tab is cycled through at its new place.
    h.state_mut().session.move_document(0, 2);
    h.state_mut().session.set_active(1);
    press(&mut h, false);
    assert_eq!(h.state().session.active().unwrap().doc.name, "ch1-p01.png");
    h.key_press(egui::Key::Tab);
    h.run_steps(2);
    assert_eq!(active(&h), Some(2), "plain Tab doesn't switch documents");
    assert!(!h.state().ui.panels.dock, "plain Tab hides the panels");
}

/// One document: Ctrl+Tab stays on it quietly; a focused widget or text field doesn't keep the key.
#[test]
fn ctrl_tab_with_one_document_or_a_focused_field() {
    let mut h = harness(1, ThemeKind::Pro);
    h.key_press_modifiers(ctrl(), egui::Key::Tab);
    h.run_steps(2);
    assert_eq!(h.state().session.active_index(), Some(0));
    assert!(crate::shortcut_dispatch::take_log(&h.ctx).iter().all(|(_, o)| *o == crate::shortcut_dispatch::Outcome::Ran), "no disabled-shortcut report");
    use crate::shortcut_dispatch::Focus;
    for (key, sc) in [("Ctrl+Tab", "window.nextDocument"), ("Ctrl+Shift+Tab", "window.previousDocument")] {
        let parsed = crate::shortcuts::parse(key).unwrap();
        assert!(Focus::Widget.allows(&parsed) && Focus::Text.allows(&parsed), "{key} reaches {sc} from a focused field");
        let bound = crate::shortcut_dispatch::bindings(h.state()).into_iter().find(|(_, s)| *s == parsed).map(|(id, _)| id);
        assert_eq!(bound.as_deref(), Some(sc));
    }
    assert!(!Focus::Widget.allows(&crate::shortcuts::parse("Tab").unwrap()), "a focused widget keeps plain Tab");
}
