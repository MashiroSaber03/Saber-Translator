//! The two-modal save/close workflow, including the writer's success/failure boundary.
use super::*;
use crate::{FileDialogAnswer, tiff_options_ui};
use egui_kittest::kittest::Queryable;
use std::{cell::RefCell, rc::Rc};

type Writes = Rc<RefCell<Vec<(String, Vec<u8>)>>>;

fn tiff_prompt(command: &str) -> (Prompted, Writes, DocId, DocId) {
    let mut app = app_with_docs(2);
    make_dirty(&mut app, 0);
    let (saved, other) = (doc_id(&app, 0), doc_id(&app, 1));
    app.session.active_mut().unwrap().path = Some("original.tif".into());
    let writes: Writes = Rc::default();
    let written = writes.clone();
    app.services.file_dialog = Some(crate::file_dialog::fake(vec![Some(FileDialogAnswer::SaveTo("kept.tif".into()))]).0);
    app.services.export = Some(Box::new(move |doc, _, settings| Ok((serde_json::to_vec(&(doc.id, settings.tiff_layers)).unwrap(), Vec::new()))));
    app.services.write = Some(Box::new(move |path, bytes| {
        written.borrow_mut().push((path.to_string(), bytes.to_vec()));
        Ok(())
    }));
    let builder = egui_kittest::Harness::builder().with_size(egui::vec2(800.0, 600.0));
    let builder = if std::env::var_os("PHOTOCRAFT_TIFF_SNAPSHOTS").is_some() { builder.wgpu() } else { builder };
    let mut h = builder.build_ui_state(
        |ui, app| {
            show(app, ui.ctx());
            tiff_options_ui::show(app, ui.ctx());
            app.poll_file_dialog(ui.ctx(), None);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.ctx.set_os(egui::os::OperatingSystem::Windows);
    assert!(intercept(h.state_mut(), command, &Value::Null));
    h.run_steps(2);
    (h, writes, saved, other)
}

fn capture(h: &mut Prompted, name: &str) {
    if let Some(dir) = std::env::var_os("PHOTOCRAFT_TIFF_SNAPSHOTS") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        h.render().unwrap().save(dir.join(format!("{name}.png"))).unwrap();
    }
}

#[test]
fn closing_after_tiff_save_waits_for_options_and_writes_the_requested_document() {
    for command in ["file.close", "file.closeAll", EXIT] {
        let (mut h, writes, saved, other) = tiff_prompt(command);
        capture(&mut h, "01-unsaved");
        h.key_press(Key::Y);
        h.run_steps(3);
        capture(&mut h, "02-tiff-options");
        assert!(h.state().tiff_options.is_some());
        assert_eq!(h.state().session.documents().len(), 2, "nothing closes before TIFF Options saves");
        assert_eq!(docs_left(&h), Some(1));
        assert!(!h.state().allow_close);
        assert!(writes.borrow().is_empty());
        assert!(h.query_by_label("(Y)es").is_none(), "the close prompt waits behind TIFF Options");
        // Another window or a control request can change the active tab while the modal is up.
        h.state_mut().refocus(other).unwrap();
        h.key_press(Key::Enter);
        h.run_steps(3);
        assert_eq!(writes.borrow().as_slice(), &[("kept.tif".into(), serde_json::to_vec(&(saved, true)).unwrap())]);
        assert!(h.state().discard.is_none());
        assert!(h.state().tiff_options.is_none());
        match command {
            EXIT => assert!(h.state().allow_close),
            "file.closeAll" => assert!(h.state().session.documents().is_empty()),
            _ => {
                assert_eq!(h.state().session.documents().len(), 1);
                assert_eq!(doc_id(h.state(), 0), other);
            }
        }
    }
}

#[test]
fn cancelling_failing_or_copying_a_tiff_keeps_the_original_unsaved() {
    for outcome in ["cancel", "export error", "write error", "copy"] {
        let (mut h, writes, saved, _) = tiff_prompt("file.close");
        h.key_press(Key::Y);
        h.run_steps(3);
        match outcome {
            "export error" => h.state_mut().services.export = Some(Box::new(|_, _, _| Err("cannot encode".into()))),
            "write error" => h.state_mut().services.write = Some(Box::new(|_, _| Err("disk full".into()))),
            "copy" => {
                h.get_by_label("Discard Layers and Save a Copy").click();
                h.run_steps(2);
            }
            _ => {}
        }
        h.key_press(if outcome == "cancel" { Key::Escape } else { Key::Enter });
        h.run_steps(3);
        if outcome == "cancel" {
            capture(&mut h, "03-cancelled");
        }
        assert!(h.state().tiff_options.is_none());
        assert_eq!(h.state().session.documents().len(), 2, "{outcome}");
        assert_eq!(docs_left(&h), Some(1), "{outcome}");
        assert!(h.query_by_label("(Y)es").is_some());
        let original = &h.state().session.documents()[0];
        assert!(original.is_dirty());
        assert_eq!(original.path.as_deref(), Some("original.tif"));
        if outcome == "copy" {
            assert_eq!(writes.borrow().as_slice(), &[("kept.tif".into(), serde_json::to_vec(&(saved, false)).unwrap())]);
        } else {
            assert!(writes.borrow().is_empty());
        }
        if outcome.ends_with("error") {
            assert!(h.state().ui.status_error);
        }
    }
}

#[test]
fn tiff_options_cannot_be_replaced_or_retarget_a_closed_document() {
    let (mut h, writes, _, other) = tiff_prompt("file.close");
    h.key_press(Key::Y);
    h.run_steps(3);
    assert!(h.state_mut().save_as(Some("replacement.tif".into())).is_err());
    assert_eq!(h.state().tiff_options.as_ref().unwrap().path, "kept.tif");
    // A direct engine command bypasses the UI close guard. Never fall through to the next tab.
    h.state_mut().run("file.close", json!({"document": 0})).unwrap();
    h.key_press(Key::Enter);
    h.run_steps(3);
    assert!(writes.borrow().is_empty());
    assert_eq!(h.state().session.documents().len(), 1);
    assert_eq!(doc_id(h.state(), 0), other);
    assert!(h.state().ui.status_error);
}

#[test]
fn close_all_waits_for_each_layered_tiff_write() {
    let (mut h, writes, first, second) = tiff_prompt("file.closeAll");
    make_dirty(h.state_mut(), 1);
    // Rebuild the prompt after making the second document dirty.
    h.state_mut().discard = None;
    assert!(intercept(h.state_mut(), "file.closeAll", &Value::Null));
    h.state_mut().services.file_dialog =
        Some(crate::file_dialog::fake(vec![Some(FileDialogAnswer::SaveTo("first.tif".into())), Some(FileDialogAnswer::SaveTo("second.tif".into()))]).0);
    for (i, doc) in [first, second].into_iter().enumerate() {
        h.key_press(Key::Y);
        h.run_steps(3);
        assert_eq!(h.state().session.documents().len(), 2 - i, "each saved document closes once written");
        assert_eq!(writes.borrow().len(), i);
        assert_eq!(docs_left(&h), Some(2 - i));
        h.key_press(Key::Enter);
        h.run_steps(3);
        assert_eq!(writes.borrow()[i].1, serde_json::to_vec(&(doc, true)).unwrap());
    }
    assert!(h.state().session.documents().is_empty());
    assert!(h.state().discard.is_none());
}
