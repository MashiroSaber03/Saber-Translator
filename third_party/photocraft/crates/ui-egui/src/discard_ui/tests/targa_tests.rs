//! Targa Options on Save As and Save a Copy: the bits per pixel it asks for, the scripted
//! `tgaBits` that skips it, and the close prompt waiting behind it.
use super::*;
use crate::{FileDialogAnswer, targa_options_ui};
use egui_kittest::kittest::Queryable;
use photocraft_io::TgaBits;
use std::{cell::RefCell, rc::Rc};

/// Each write: the path, and the Targa bits the export was asked for (as JSON).
type Writes = Rc<RefCell<Vec<(String, Vec<u8>)>>>;

/// The JSON a write records for `bits`.
fn asked(bits: Option<u8>) -> Vec<u8> {
    serde_json::to_vec(&bits).unwrap()
}

/// One 8×8 RGB document (with an "Alpha 1" channel when `alpha_channel`) whose exports record
/// their Targa bits, in a harness showing Targa Options and answering a save dialog with `to`.
fn targa_app(alpha_channel: bool, to: &str) -> (Prompted, Writes) {
    let mut app = app_with_docs(1);
    if alpha_channel {
        app.run("select.all", json!({})).unwrap();
        app.run("select.saveSelection", json!({})).unwrap();
    }
    let writes: Writes = Rc::default();
    let written = writes.clone();
    app.services.file_dialog = Some(crate::file_dialog::fake(vec![Some(FileDialogAnswer::SaveTo(to.into()))]).0);
    app.services.export = Some(Box::new(|_, _, settings| Ok((asked(settings.tga_bits.map(TgaBits::bits)), Vec::new()))));
    app.services.write = Some(Box::new(move |path, bytes| {
        written.borrow_mut().push((path.to_string(), bytes.to_vec()));
        Ok(())
    }));
    let h = egui_kittest::Harness::builder().with_size(egui::vec2(800.0, 600.0)).build_ui_state(
        |ui, app| {
            show(app, ui.ctx());
            targa_options_ui::show(app, ui.ctx());
            app.poll_file_dialog(ui.ctx(), None);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.ctx.set_os(egui::os::OperatingSystem::Windows);
    (h, writes)
}

/// Save As to a Targa asks first: 32 bits per pixel when the document has an alpha channel,
/// else 24 (as Photoshop's Targa Options starts), and writes only once answered.
#[test]
fn save_as_to_a_targa_asks_for_its_bits_per_pixel() {
    for (alpha_channel, start) in [(false, TgaBits::Bits24), (true, TgaBits::Bits32)] {
        let (mut h, writes) = targa_app(alpha_channel, "unused.tga");
        h.state_mut().save_as(Some("t.tga".into())).unwrap();
        h.run_steps(2);
        assert_eq!(h.state().targa_options.as_ref().map(|p| (p.bits, p.copy)), Some((start, false)));
        assert!(writes.borrow().is_empty(), "nothing is written before the answer");
        assert!(h.state_mut().save_as(Some("other.tga".into())).is_err(), "one save at a time");
        h.key_press(Key::Enter);
        h.run_steps(3);
        assert_eq!(writes.borrow().as_slice(), &[("t.tga".to_string(), asked(Some(start.bits())))]);
        assert!(h.state().targa_options.is_none());
        assert_eq!(h.state().session.active().unwrap().path.as_deref(), Some("t.tga"));
    }
}

/// The choice made in the prompt is the one written; Cancel writes nothing.
#[test]
fn targa_options_writes_the_choice_or_cancels() {
    let (mut h, writes) = targa_app(true, "unused.tga");
    h.state_mut().save_as(Some("t.tga".into())).unwrap();
    h.run_steps(2);
    h.get_by_label("24 bits/pixel").click();
    h.run_steps(2);
    h.get_by_label("OK").click();
    h.run_steps(3);
    assert_eq!(writes.borrow().as_slice(), &[("t.tga".to_string(), asked(Some(24)))]);

    h.state_mut().save_as(Some("c.tga".into())).unwrap();
    h.run_steps(2);
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(h.state().targa_options.is_none());
    assert_eq!(writes.borrow().len(), 1, "cancelled: nothing more written");
    assert_eq!(h.state().ui.status, "Save cancelled");
    assert_eq!(h.state().session.active().unwrap().path.as_deref(), Some("t.tga"));
}

/// Scripts choose with `tgaBits` and are never asked; a grayscale Targa (8-bit, nothing to choose)
/// and other formats aren't asked either.
#[test]
fn scripted_bits_grayscale_and_other_formats_skip_the_prompt() {
    let (mut h, writes) = targa_app(true, "unused.tga");
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, "file.saveAs", json!({"path": "s.tga", "tgaBits": 24})).unwrap();
    assert!(h.state().targa_options.is_none());
    assert_eq!(writes.borrow().as_slice(), &[("s.tga".to_string(), asked(Some(24)))]);
    for bad in [json!(16), json!("32")] {
        assert!(crate::menus::invoke(h.state_mut(), &ctx, "file.saveAs", json!({"path": "s.tga", "tgaBits": bad})).is_err(), "{bad}");
    }
    assert_eq!(writes.borrow().len(), 1, "a bad tgaBits writes nothing");
    crate::menus::invoke(h.state_mut(), &ctx, "file.saveAs", json!({"path": "p.png"})).unwrap();
    h.state_mut().run("image.mode.grayscale", json!({})).unwrap();
    crate::menus::invoke(h.state_mut(), &ctx, "file.saveAs", json!({"path": "g.tga"})).unwrap();
    assert!(h.state().targa_options.is_none());
    assert_eq!(writes.borrow()[1..], [("p.png".to_string(), asked(None)), ("g.tga".to_string(), asked(None))]);
}

/// Save a Copy to a Targa asks too, and the copy leaves the document's path and unsaved state.
#[test]
fn save_a_copy_to_a_targa_asks_and_keeps_the_document_unsaved() {
    let (mut h, writes) = targa_app(true, "copy.tga");
    make_dirty(h.state_mut(), 0);
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, "file.saveACopy", json!({})).unwrap();
    h.run_steps(3);
    assert_eq!(h.state().targa_options.as_ref().map(|p| (p.bits, p.copy)), Some((TgaBits::Bits32, true)));
    assert!(targa_options_ui::park(h.state_mut(), "other.tga".into(), false, "file.saveAs").is_err(), "the waiting prompt is kept");
    assert_eq!(h.state().targa_options.as_ref().map(|p| p.path.as_str()), Some("copy.tga"));
    h.key_press(Key::Enter);
    h.run_steps(3);
    assert_eq!(writes.borrow().as_slice(), &[("copy.tga".to_string(), asked(Some(32)))]);
    let st = h.state().session.active().unwrap();
    assert!(st.is_dirty());
    assert_eq!(st.path, None);
}

/// Save As of a document a flat Targa can't hold (here, two layers) is a copy through Targa
/// Options too, as without it: the document keeps its path and its unsaved changes (#2550).
#[test]
fn targa_options_keeps_a_layered_save_as_a_copy() {
    let (mut h, writes) = targa_app(false, "unused.tga");
    make_dirty(h.state_mut(), 0);
    h.state_mut().save_as(Some("t.tga".into())).unwrap();
    h.run_steps(2);
    assert_eq!(h.state().targa_options.as_ref().map(|p| p.copy), Some(true));
    h.key_press(Key::Enter);
    h.run_steps(3);
    assert_eq!(writes.borrow().as_slice(), &[("t.tga".to_string(), asked(Some(24)))]);
    let st = h.state().session.active().unwrap();
    assert!(st.is_dirty());
    assert_eq!(st.path, None);
}

/// An action being recorded keeps a Targa Save As or Save a Copy with its bits, answered in Targa
/// Options or scripted (#2032, #4373), and plays them back without asking.
#[test]
fn recorded_targa_save_as_keeps_its_bits_and_replays_without_asking() {
    let (mut h, writes) = targa_app(true, "copy.tga");
    let ctx = h.ctx.clone();
    h.state_mut().run("actions.record", json!({})).unwrap();
    h.state_mut().save_as(Some("t.tga".into())).unwrap();
    h.run_steps(2);
    h.get_by_label("24 bits/pixel").click();
    h.run_steps(2);
    h.get_by_label("OK").click();
    h.run_steps(3);
    crate::menus::invoke(h.state_mut(), &ctx, "file.saveACopy", json!({})).unwrap();
    h.run_steps(3);
    h.key_press(Key::Enter);
    h.run_steps(3);
    crate::menus::invoke(h.state_mut(), &ctx, "file.saveAs", json!({"path": "s.tga", "tgaBits": 32})).unwrap();
    h.state_mut().run("actions.stop", json!({})).unwrap();
    assert_eq!(
        writes.borrow().as_slice(),
        &[("t.tga".to_string(), asked(Some(24))), ("copy.tga".to_string(), asked(Some(32))), ("s.tga".to_string(), asked(Some(32)))]
    );
    let steps = &h.state().session.actions.list[0].steps;
    assert_eq!(
        steps,
        &[
            ("file.saveAs".into(), json!({"path": "t.tga", "tgaBits": 24})),
            ("file.saveACopy".into(), json!({"path": "copy.tga", "tgaBits": 32})),
            ("file.saveAs".into(), json!({"path": "s.tga", "tgaBits": 32}))
        ]
    );
    writes.borrow_mut().clear();
    let played = h.state_mut().run("actions.play", json!({"action": 0})).unwrap();
    assert!(played.get("failed").is_none(), "{played}");
    assert!(h.state().targa_options.is_none(), "playback doesn't ask");
    assert_eq!(
        writes.borrow().as_slice(),
        &[("t.tga".to_string(), asked(Some(24))), ("copy.tga".to_string(), asked(Some(32))), ("s.tga".to_string(), asked(Some(32)))]
    );
}

/// Closing an unsaved document whose save goes to a Targa waits behind Targa Options, then closes.
#[test]
fn closing_after_a_targa_save_waits_for_targa_options() {
    let (mut h, writes) = targa_app(false, "kept.tga");
    // An edit that keeps the lone Background, so the Targa holds the document and isn't a copy.
    h.state_mut().run("edit.fill", json!({"color": "#000000"})).unwrap();
    assert!(intercept(h.state_mut(), "file.close", &Value::Null));
    h.run_steps(2);
    h.key_press(Key::Y);
    h.run_steps(3);
    assert!(h.state().targa_options.is_some());
    assert_eq!(h.state().session.documents().len(), 1, "nothing closes before Targa Options saves");
    assert!(h.query_by_label("(Y)es").is_none(), "the close prompt waits behind Targa Options");
    h.key_press(Key::Enter);
    h.run_steps(3);
    assert_eq!(writes.borrow().as_slice(), &[("kept.tga".to_string(), asked(Some(24)))]);
    assert!(h.state().discard.is_none());
    assert!(h.state().session.documents().is_empty());
}
