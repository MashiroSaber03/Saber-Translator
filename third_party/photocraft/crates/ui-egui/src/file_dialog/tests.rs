use super::*;
use crate::{Services, menus};
use std::cell::RefCell;
use std::rc::Rc;

/// The requests the fake dialog service was given, with their replies: dialogs the user is still in.
type Open = Rc<RefCell<Vec<(FileDialogRequest, FileDialogReply)>>>;

/// An app with an untitled document, whose file dialogs stay open until the test answers them, and
/// whose writer records the paths it wrote.
fn app() -> (PhotocraftApp, Open, Rc<RefCell<Vec<String>>>) {
    let (open, written): (Open, Rc<RefCell<Vec<String>>>) = Default::default();
    let (held, w) = (open.clone(), written.clone());
    let services = Services {
        export: Some(Box::new(|_: &photocraft_doc::Document, _: &str, _: &crate::ExportSettings| Ok((b"out".to_vec(), Vec::new())))),
        write: Some(Box::new(move |path: &str, _: &[u8]| {
            w.borrow_mut().push(path.to_string());
            Ok(())
        })),
        file_dialog: Some(Box::new(move |request, _parent, reply| held.borrow_mut().push((request, reply)))),
        ..Default::default()
    };
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
    app.run("file.new", json!({"width": 4, "height": 4})).unwrap();
    (app, open, written)
}

#[test]
fn embedded_place_and_drop_add_layers_to_the_current_page() {
    let (mut app, open, _) = app();
    let bytes = photocraft_io::export(&app.session.active().unwrap().doc, "material.png", &Default::default()).unwrap().bytes;
    app.services.host_save = Some(Box::new(|| {}));
    app.services.import = Some(Box::new(|name, bytes, depth| {
        photocraft_io::import_with_svg_group_depth(name, bytes, depth).map(|v| (v.document, v.warnings)).map_err(|e| e.to_string())
    }));
    let original = app.session.active().unwrap().doc.clone();
    let ctx = egui::Context::default();
    assert!(menus::is_enabled(&app, "file.placeEmbedded"));
    menus::invoke(&mut app, &ctx, "file.placeEmbedded", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::Contents("material.png".into(), bytes.clone())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(app.session.documents().len(), 1);
    let doc = &app.session.active().unwrap().doc;
    assert_eq!(doc.id, original.id);
    assert_eq!(doc.size, original.size);
    assert_eq!(doc.layer_count(), original.layer_count() + 1);
    app.run("edit.undo", json!({})).unwrap();
    assert_eq!(app.session.active().unwrap().doc.layer_count(), original.layer_count());
    app.open_bytes("dropped.png", &bytes).unwrap();
    assert_eq!(app.session.documents().len(), 1);
    assert_eq!(app.session.active().unwrap().doc.id, original.id);
    assert_eq!(app.session.active().unwrap().doc.layer_count(), original.layer_count() + 1);
}

#[test]
fn embedded_cancelled_or_invalid_material_keeps_the_page() {
    let (mut app, open, _) = app();
    app.services.host_save = Some(Box::new(|| {}));
    let original = app.session.active().unwrap().doc.clone();
    let ctx = egui::Context::default();
    for reply in [None, Some(FileDialogAnswer::Contents("invalid.png".into(), vec![0; 16]))] {
        menus::invoke(&mut app, &ctx, "file.placeEmbedded", json!({})).unwrap();
        app.poll_file_dialog(&ctx, None);
        answer(&open, reply);
        app.poll_file_dialog(&ctx, None);
        assert_eq!(app.session.documents().len(), 1);
        let doc = &app.session.active().unwrap().doc;
        assert_eq!(doc.id, original.id);
        assert_eq!(doc.size, original.size);
        assert_eq!(doc.layer_count(), original.layer_count());
    }
}

#[test]
fn pdn_save_requests_a_native_copy_without_overwriting_the_source() {
    let (mut app, open, written) = app();
    let st = app.session.active_mut().unwrap();
    st.path = Some("/pics/layers.PDN".into());
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "/pics/layers.pcraft"));
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/layers.pcraft".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/layers.pcraft"]);
    assert_eq!(app.session.active().unwrap().path.as_deref(), Some("/pics/layers.pcraft"));
}

/// A real flat JPEG import, optionally edited to contain another layer. A filename without a
/// path is the web/download case; desktop opens also retain their source location.
fn jpeg_app(path: Option<&str>, layered: bool) -> (PhotocraftApp, Open, Rc<RefCell<Vec<String>>>) {
    let (mut app, open, written) = app();
    let bytes = photocraft_io::export(&app.session.active().unwrap().doc, "My image.JPEG", &Default::default()).unwrap().bytes;
    let doc = photocraft_io::import("My image.JPEG", &bytes).unwrap().document;
    app.session.add_document(doc, path.map(str::to_string));
    if layered {
        app.run("layer.new.layer", json!({})).unwrap();
    }
    (app, open, written)
}

#[test]
fn layered_flat_source_save_defaults_to_psd_without_overwriting_source() {
    let ctx = egui::Context::default();
    for command in ["file.save", "file.saveAs"] {
        let (mut app, open, written) = jpeg_app(Some("/pics/My image.JPEG"), true);
        menus::invoke(&mut app, &ctx, command, json!({})).unwrap();
        app.poll_file_dialog(&ctx, None);
        assert!(
            matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "/pics/My image.psd"),
            "{command}: {:?}",
            open.borrow().first().map(|(r, _)| r.clone())
        );
        assert!(written.borrow().is_empty(), "asking where to save must not overwrite the JPEG");
        assert_eq!(app.session.active().unwrap().path.as_deref(), Some("/pics/My image.JPEG"));
        answer(&open, Some(FileDialogAnswer::SaveTo("/pics/My image.psd".into())));
        app.poll_file_dialog(&ctx, None);
        assert_eq!(*written.borrow(), ["/pics/My image.psd"]);
        assert_eq!(app.session.active().unwrap().path.as_deref(), Some("/pics/My image.psd"));
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 2);
    }
}

#[test]
fn layered_filename_only_save_defaults_to_psd() {
    let (mut app, open, written) = jpeg_app(None, true);
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "My image.psd"));
    answer(&open, Some(FileDialogAnswer::SaveTo("My image.psd".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["My image.psd"]);
}

#[test]
fn save_dialog_preserves_flat_and_existing_layered_format_defaults() {
    let ctx = egui::Context::default();
    for (path, layered) in [
        ("/pics/My image.JPEG", false),
        ("/pics/My image.PSD", true),
        ("/pics/My image.psb", true),
        ("/pics/My image.pcraft", true),
        ("/pics/My image.TIF", true),
        ("/pics/My image.tiff", true),
    ] {
        let (mut app, open, written) = jpeg_app(Some(path), layered);
        menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
        app.poll_file_dialog(&ctx, None);
        assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == path), "{path}");
        answer(&open, None);
        app.poll_file_dialog(&ctx, None);
        assert!(written.borrow().is_empty());
    }
}

#[test]
fn transparent_png_keeps_its_format_until_its_lone_layer_becomes_smart() {
    let (mut app, open, written) = app();
    std::sync::Arc::make_mut(&mut app.session.active_mut().unwrap().doc).layers[0].surface_mut().unwrap().write_pixel(0, 0, &[1.0, 0.0, 0.0, 0.0]);
    let bytes = photocraft_io::export(&app.session.active().unwrap().doc, "transparent.PNG", &Default::default()).unwrap().bytes;
    let doc = photocraft_io::import("transparent.PNG", &bytes).unwrap().document;
    assert_eq!(doc.layers.len(), 1);
    assert!(!doc.layers[0].locks.transparency, "transparent PNG imports as an unlocked raster layer");
    assert_eq!(doc.layers[0].surface().unwrap().pixel(0, 0)[3], 0.0);
    app.session.add_document(doc, Some("/pics/transparent.PNG".into()));
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "/pics/transparent.PNG"));
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    app.run("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    assert_eq!(app.session.active().unwrap().doc.layers.len(), 1);
    assert!(matches!(app.session.active().unwrap().doc.layers[0].content, photocraft_doc::LayerContent::Smart(_)));
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "/pics/transparent.psd"));
    assert!(written.borrow().is_empty());
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
}

#[test]
fn layered_save_respects_explicit_jpeg_choices() {
    let ctx = egui::Context::default();
    let (mut app, open, written) = jpeg_app(Some("/pics/My image.JPEG"), true);
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::SaveTo("/exports/chosen.JPEG".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/exports/chosen.jpeg"]);
    // The JPEG can't hold the layers, so it is a copy and the document keeps its file (#2550).
    assert_eq!(app.session.active().unwrap().path.as_deref(), Some("/pics/My image.JPEG"));
    // Explicit command paths carry the same choice and bypass the picker.
    for command in ["file.save", "file.saveAs"] {
        let (mut app, open, written) = jpeg_app(Some("/pics/My image.JPEG"), true);
        menus::invoke(&mut app, &ctx, command, json!({"path": "/exports/chosen.jpg"})).unwrap();
        assert!(!app.file_dialog_open() && open.borrow().is_empty());
        assert_eq!(*written.borrow(), ["/exports/chosen.jpg"]);
    }
}

/// The user answers the open dialog, from the dialog's own thread.
fn answer(open: &Open, answer: Option<FileDialogAnswer>) {
    let (_, reply) = open.borrow_mut().pop().expect("a dialog is open");
    std::thread::spawn(move || reply.send(answer)).join().unwrap();
}

#[test]
fn open_starts_in_the_last_used_folder() {
    let (mut app, open, _) = app();
    let ctx = egui::Context::default();
    app.ui.recent_files = vec!["/pics/cat.psd".into(), "/old/dog.png".into()];
    menus::invoke(&mut app, &ctx, "file.open", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(
        matches!(open.borrow().as_slice(), [(FileDialogRequest::Open { multiple: true, initial_dir: Some(dir), extensions: None }, _)] if dir == "/pics"),
        "{:?}",
        open.borrow().first().map(|(r, _)| r.clone())
    );
}

#[test]
fn ask_folder_uses_the_async_dialog_and_returns_the_selected_path() {
    let (mut app, open, _) = app();
    let ctx = egui::Context::default();
    app.ui.recent_files = vec!["/pics/cat.psd".into()];
    let selected = Rc::new(RefCell::new(None));
    let result = selected.clone();

    assert_eq!(
        app.ask_folder(move |_, path| {
            *result.borrow_mut() = Some(path);
            Ok(Value::Null)
        })
        .unwrap(),
        json!({"fileDialog": "folder"})
    );
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Folder { initial_dir: Some(dir) }, _)] if dir == "/pics"));

    answer(&open, Some(FileDialogAnswer::FolderTo("/pics/frames".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(selected.borrow().as_deref(), Some("/pics/frames"));
}

#[test]
fn filtered_picker_requests_only_the_given_extensions() {
    let (mut app, open, _) = app();
    app.pick_file_bytes_filtered(&["cube", "3dl", "look"], |_, _, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&egui::Context::default(), None);
    assert!(matches!(
        open.borrow().first().map(|(r, _)| r),
        Some(FileDialogRequest::Open { multiple: false, extensions: Some(exts), .. })
            if exts.iter().map(String::as_str).collect::<Vec<_>>() == ["cube", "3dl", "look"]
    ));
}

#[test]
fn scripts_browse_lists_only_script_files() {
    let (mut app, open, _) = app();
    menus::invoke(&mut app, &egui::Context::default(), "file.scripts.browse", json!({})).unwrap();
    app.poll_file_dialog(&egui::Context::default(), None);
    assert!(matches!(
        open.borrow().first().map(|(r, _)| r),
        Some(FileDialogRequest::Open { multiple: false, extensions: Some(exts), .. })
            if exts.iter().map(String::as_str).collect::<Vec<_>>() == ["psjs", "ccjs", "jsx", "js", "jsxbin", "json", "txt"]
    ));
}

#[test]
fn last_used_dir_takes_the_most_recent_parent() {
    assert_eq!(last_used_dir(&[]), None);
    assert_eq!(last_used_dir(&["/pics/cat.psd".into()]), Some("/pics".into()));
    assert_eq!(last_used_dir(&["/pics/cat.psd".into(), "/old/dog.png".into()]), Some("/pics".into()));
    assert_eq!(last_used_dir(&["cat.psd".into()]), None, "no folder to start in");
    assert_eq!(last_used_dir(&["/".into()]), None, "the root has no parent");
}

#[test]
fn the_app_keeps_running_while_a_dialog_is_open() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    assert_eq!(menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap(), json!({"fileDialog": "save"}));
    assert!(open.borrow().is_empty(), "shown at the end of the frame, which has the window to parent it to");
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested.ends_with(".psd")));
    // Frames go on while the user is in the dialog: nothing is written, commands still run, and
    // a second dialog is refused.
    for _ in 0..3 {
        app.poll_file_dialog(&ctx, None);
    }
    assert!(written.borrow().is_empty());
    assert!(app.file_dialog_open());
    app.run("layer.new.layer", json!({})).unwrap();
    assert_eq!(menus::invoke(&mut app, &ctx, "file.open", json!({})).unwrap_err(), "a file dialog is already open");
    app.poll_file_dialog(&ctx, None);
    assert_eq!(open.borrow().len(), 1);
    // The answer wakes the app, and the save goes on from where it asked.
    for _ in 0..4 {
        ctx.run_ui(egui::RawInput::default(), |_| {}).textures_delta.clear();
    }
    assert!(!ctx.has_requested_repaint(), "idle");
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/a.psd".into())));
    assert!(ctx.has_requested_repaint());
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/a.psd"]);
    let st = app.session.active().unwrap();
    assert_eq!(st.path.as_deref(), Some("/pics/a.psd"));
    assert_eq!(st.doc.name, "a.psd", "a successful Save As adopts the file name");
    assert!(!st.is_dirty());
    assert!(!app.file_dialog_open());
}

/// Preferences ▸ File Handling ▸ Lowercase Extension (default on): the chosen path's extension
/// is lowercased on the way to the writer; unchecked, the user's spelling is kept.
#[test]
fn save_paths_get_a_lowercase_extension_when_the_preference_is_on() {
    let ctx = egui::Context::default();
    {
        let (mut app, open, written) = app();
        menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
        app.poll_file_dialog(&ctx, None);
        answer(&open, Some(FileDialogAnswer::SaveTo("/pics/CAT.PSD".into())));
        app.poll_file_dialog(&ctx, None);
        assert_eq!(*written.borrow(), ["/pics/CAT.psd"], "only the extension is lowercased");
        assert_eq!(app.session.active().unwrap().path.as_deref(), Some("/pics/CAT.psd"));
    }
    let (mut app, open, written) = app();
    app.run("prefs.set", json!({"path": "fileHandling.lowercaseExtension", "value": false})).unwrap();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/CAT.PSD".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/CAT.PSD"], "off keeps the user's spelling");
}

/// #5039: a name typed without an extension (where file extensions are hidden) gets the
/// suggested file's extension, so the save picks its format instead of failing.
#[test]
fn save_paths_without_an_extension_get_the_suggested_one() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/cat".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/cat.psd"]);
    assert_eq!(app.session.active().unwrap().path.as_deref(), Some("/pics/cat.psd"));
    // A name that already has an extension is left alone.
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/dog.png".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/cat.psd", "/pics/dog.png"]);
}

#[test]
fn save_and_export_dialogs_start_beside_the_document() {
    let (mut app, open, _) = app();
    let ctx = egui::Context::default();
    let dir = std::env::temp_dir().join("PhotoCraft projects").join("图像");
    let source = dir.join("original.psd");
    app.session.active_mut().unwrap().path = Some(source.to_string_lossy().into_owned());
    for command in ["file.saveAs", "file.export.exportAs", "file.export.saveForWebLegacy"] {
        let r = menus::invoke(&mut app, &ctx, command, json!({})).unwrap();
        if let Some(id) = r["dialog"].as_u64() {
            crate::dialogs::confirm(&mut app, id).unwrap();
        }
        app.poll_file_dialog(&ctx, None);
        {
            let held = open.borrow();
            let (FileDialogRequest::Save { suggested }, _) = &held[0] else { panic!("save dialog expected") };
            assert_eq!(std::path::Path::new(suggested).parent(), Some(dir.as_path()), "{command}");
            if command == "file.saveAs" {
                assert_eq!(std::path::Path::new(suggested), source);
            }
        }
        answer(&open, None);
        app.poll_file_dialog(&ctx, None);
    }
}

/// The path `pick_save` suggests for `name` in `dir` on this platform: it joins with `Path::join`,
/// which uses `\` on Windows, so the expected values must be built the same way (#2745).
fn joined(dir: &str, name: &str) -> String {
    std::path::Path::new(dir).join(name).to_string_lossy().into_owned()
}

/// #1826: after the user navigates a save or export dialog elsewhere, the document's next dialog
/// starts there. Save As of a saved document still suggests its own path; other documents and
/// closed documents are unaffected.
#[test]
fn save_dialogs_remember_where_each_document_was_last_saved() {
    let (mut app, open, _) = app();
    let ctx = egui::Context::default();
    let first = app.session.active().unwrap().doc.id;
    app.session.active_mut().unwrap().path = Some("/proj/original.psd".into());
    let suggested = |open: &Open| match &open.borrow()[0].0 {
        FileDialogRequest::Save { suggested } => suggested.clone(),
        other => panic!("save dialog expected, got {other:?}"),
    };
    // An export starts beside the document; the user saves it under /renders instead.
    app.pick_save("result.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), joined("/proj", "result.png"));
    answer(&open, Some(FileDialogAnswer::SaveTo("/renders/result.png".into())));
    app.poll_file_dialog(&ctx, None);
    // The next export starts in /renders; Save As still offers the document's own path.
    app.pick_save("again.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), joined("/renders", "again.png"));
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), "/proj/original.psd");
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    // A cancelled dialog and an explicit directory change nothing.
    app.pick_save("/elsewhere/x.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), "/elsewhere/x.png");
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    app.pick_save("third.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), joined("/renders", "third.png"));
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    // Another document has its own memory: untitled, it starts wherever the shell defaults.
    app.run("file.new", json!({"width": 4, "height": 4})).unwrap();
    let second = app.session.active().unwrap().doc.id;
    assert_ne!(first, second);
    app.pick_save("fresh.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), "fresh.png");
    answer(&open, Some(FileDialogAnswer::SaveTo("/other/fresh.png".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(app.save_dirs.get(&second).map(|p| p.to_string_lossy().into_owned()).as_deref(), Some("/other"));
    assert_eq!(app.save_dirs.get(&first).map(|p| p.to_string_lossy().into_owned()).as_deref(), Some("/renders"));
    // The folder is remembered for the document the dialog was asked for, not the active one.
    app.session.set_active(0);
    app.pick_save("swap.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), joined("/renders", "swap.png"));
    app.session.set_active(1);
    answer(&open, Some(FileDialogAnswer::SaveTo("/moved/swap.png".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(app.save_dirs.get(&first).map(|p| p.to_string_lossy().into_owned()).as_deref(), Some("/moved"));
    assert_eq!(app.save_dirs.get(&second).map(|p| p.to_string_lossy().into_owned()).as_deref(), Some("/other"));
    // Closing a document forgets its folder at the next dialog.
    app.run("file.close", json!({"document": 1})).unwrap();
    app.pick_save("last.png", |_, _| Ok(Value::Null)).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(suggested(&open), joined("/moved", "last.png"));
    assert!(!app.save_dirs.contains_key(&second));
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
}

#[test]
fn save_suggestions_preserve_explicit_directories_and_untitled_names() {
    let (mut app, open, _) = app();
    let ctx = egui::Context::default();
    for (document_path, suggested) in
        [(None, "Untitled.png"), (Some("project/source.psd"), "exports/result.png"), (Some("project/source.psd"), "/other/result.png")]
    {
        app.session.active_mut().unwrap().path = document_path.map(str::to_string);
        app.pick_save(suggested, |_, _| Ok(Value::Null)).unwrap();
        app.poll_file_dialog(&ctx, None);
        assert!(matches!(&open.borrow()[0].0, FileDialogRequest::Save { suggested: actual } if actual == suggested));
        answer(&open, None);
        app.poll_file_dialog(&ctx, None);
    }
}

#[test]
fn the_answer_goes_to_the_document_it_was_asked_for() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    app.run("file.new", json!({"width": 4, "height": 4})).unwrap();
    app.session.set_active(0);
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    app.session.set_active(1);
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/first.psd".into())));
    app.poll_file_dialog(&ctx, None);
    let paths: Vec<_> = app.session.documents().iter().map(|d| d.path.clone()).collect();
    assert_eq!(paths, [Some("/pics/first.psd".to_string()), None]);
    assert_eq!(app.session.active_index(), Some(1), "a completed save preserves the selected tab");
    assert_eq!(app.session.documents()[0].doc.name, "first.psd");
    assert_ne!(app.session.documents()[1].doc.name, "first.psd");
    // Once that document is closed, the answer has nothing to save.
    app.session.set_active(0);
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    app.run("file.close", json!({"document": 0})).unwrap();
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/again.psd".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/first.psd"]);
    assert!(app.ui.status_error && app.session.documents()[0].path.is_none());
}

#[test]
fn cancel_and_a_dialog_that_could_not_show_end_quietly() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    app.run("layer.new.layer", json!({})).unwrap();
    let before = app.session.active().unwrap().clone();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    assert!(!app.file_dialog_open() && !app.ui.status_error);
    // A reply dropped unanswered (the dialog failed to show) is a Cancel, not a dialog left open forever.
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    open.borrow_mut().clear();
    app.poll_file_dialog(&ctx, None);
    assert!(!app.file_dialog_open() && !app.ui.status_error);
    assert!(written.borrow().is_empty());
    let after = app.session.active().unwrap();
    assert_eq!(after.doc.name, before.doc.name);
    assert_eq!(after.path, before.path);
    assert_eq!((after.revision, after.saved_revision), (before.revision, before.saved_revision));
    // Without a dialog service, asking is a Cancel straight away.
    app.services.file_dialog = None;
    assert_eq!(app.open_dialog_file().unwrap_err(), CANCELLED);
    assert!(!app.file_dialog_open());
}

#[test]
fn opening_another_document_before_the_save_answer_keeps_its_identity() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    let first_id = app.active_doc_id().unwrap();
    app.save_as(None).unwrap();
    app.poll_file_dialog(&ctx, None);
    app.run("file.new", json!({"width": 4, "height": 4, "name": "Untitled-2"})).unwrap();
    let second_id = app.active_doc_id().unwrap();
    app.run("document.move", json!({"document": 0, "to": 1})).unwrap();
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/保存 first.psd".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/pics/保存 first.psd"]);
    let first = app.session.documents().iter().find(|s| s.doc.id == first_id).unwrap();
    assert_eq!(first.doc.name, "保存 first.psd");
    assert_eq!(first.path.as_deref(), Some("/pics/保存 first.psd"));
    assert_eq!(app.active_doc_id().unwrap(), second_id);
    assert_eq!(app.session.active().unwrap().doc.name, "Untitled-2");
}

#[test]
fn failed_deferred_save_preserves_both_identities_and_the_selected_tab() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    app.run("layer.new.layer", json!({})).unwrap();
    app.save_as(None).unwrap();
    app.poll_file_dialog(&ctx, None);
    app.run("file.new", json!({"width": 4, "height": 4, "name": "second"})).unwrap();
    let before: Vec<_> = app.session.documents().iter().map(|d| (d.doc.name.clone(), d.path.clone(), d.revision, d.saved_revision)).collect();
    app.services.write = Some(Box::new(|_, _| Err("disk full".into())));
    answer(&open, Some(FileDialogAnswer::SaveTo("failed.psd".into())));
    app.poll_file_dialog(&ctx, None);
    let after: Vec<_> = app.session.documents().iter().map(|d| (d.doc.name.clone(), d.path.clone(), d.revision, d.saved_revision)).collect();
    assert_eq!(after, before);
    assert_eq!(app.session.active_index(), Some(1));
    assert!(app.ui.status_error && app.ui.status == "disk full");
    assert!(written.borrow().is_empty());
}

#[test]
fn an_answer_of_the_wrong_kind_is_an_error_not_a_crash() {
    let (mut app, open, written) = app();
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::Paths(vec!["/pics/a.psd".into()])));
    app.poll_file_dialog(&ctx, None);
    assert!(app.ui.status_error && app.ui.status == UNEXPECTED);
    assert!(written.borrow().is_empty());
    app.open_dialog_file().unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/a.psd".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(app.session.documents().len(), 1);
}

/// The app with a painted layer converted to a smart object and a duplicate instance of it.
fn smart_app() -> (PhotocraftApp, Open, Rc<RefCell<Vec<String>>>, [u64; 2]) {
    let (mut app, open, written) = app();
    app.run("layer.new.layer", json!({})).unwrap();
    app.session
        .edit("paint", |doc, active| {
            doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[0.0, 0.0, 1.0, 1.0]);
            Ok(())
        })
        .unwrap();
    let a = app.run("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].as_u64().unwrap();
    let b = app.run("layer.duplicate", json!({"layer": a})).unwrap()["layer"].as_u64().unwrap();
    app.session.select_layer(photocraft_doc::LayerId(a)).unwrap();
    (app, open, written, [a, b])
}

fn smart_source(app: &PhotocraftApp, id: u64) -> photocraft_doc::SmartSource {
    match &app.session.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
        photocraft_doc::LayerContent::Smart(sm) => sm.source.clone(),
        other => panic!("not a smart object: {}", other.kind_name()),
    }
}

#[test]
fn replace_contents_picks_a_file_for_every_instance() {
    let (mut app, open, _, ids) = smart_app();
    let ctx = egui::Context::default();
    let before = ids.map(|id| smart_source(&app, id));
    // Cancelling changes nothing.
    menus::invoke(&mut app, &ctx, "layer.smartObjects.replaceContents", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Open { multiple: false, .. }, _)]));
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    assert_eq!(ids.map(|id| smart_source(&app, id)), before);
    // A picked file (by contents, as on the web) replaces the contents of both instances.
    let png = photocraft_io::export(&app.session.active().unwrap().doc, "png", &Default::default()).unwrap().bytes;
    menus::invoke(&mut app, &ctx, "layer.smartObjects.replaceContents", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::Contents("art.png".into(), png)));
    app.poll_file_dialog(&ctx, None);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    for id in ids {
        assert!(matches!(smart_source(&app, id), photocraft_doc::SmartSource::Embedded { ref file_name, .. } if file_name == "art.png"));
    }
    // A file that isn't an image is an error, and the contents stay.
    menus::invoke(&mut app, &ctx, "layer.smartObjects.replaceContents", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::Contents("notes.png".into(), b"not an image".to_vec())));
    app.poll_file_dialog(&ctx, None);
    assert!(app.ui.status_error);
    assert!(matches!(smart_source(&app, ids[1]), photocraft_doc::SmartSource::Embedded { ref file_name, .. } if file_name == "art.png"));
}

#[test]
fn export_contents_suggests_the_contents_file_name() {
    let (mut app, open, written, _) = smart_app();
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "layer.smartObjects.exportContents", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(
        matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "Layer 1.pcraft"),
        "{:?}",
        open.borrow().first().map(|(r, _)| r.clone())
    );
    answer(&open, Some(FileDialogAnswer::SaveTo("/out/contents.pcraft".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(*written.borrow(), ["/out/contents.pcraft"]);
}

#[test]
fn convert_to_linked_asks_where_to_write_the_contents() {
    let (mut app, open, _, ids) = smart_app();
    let ctx = egui::Context::default();
    let dir = std::env::temp_dir().join(format!("pcraft-convert-linked-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("linked.pcraft").to_string_lossy().into_owned();
    menus::invoke(&mut app, &ctx, "layer.smartObjects.convertToLinked", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { .. }, _)]));
    answer(&open, Some(FileDialogAnswer::SaveTo(path.clone())));
    app.poll_file_dialog(&ctx, None);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert!(std::fs::metadata(&path).is_ok_and(|m| m.len() > 0));
    std::fs::remove_dir_all(&dir).ok();
    for id in ids {
        assert_eq!(smart_source(&app, id), photocraft_doc::SmartSource::Linked { path: path.clone() });
    }
}

#[test]
fn quick_export_selected_layers_writes_separate_pngs_with_one_prompt() {
    let (mut app, open, written) = app();
    app.run("layer.new.layer", json!({"name": "Same/name"})).unwrap();
    app.run("paint.bucket", json!({"x": 0, "y": 0, "color": "#ff0000"})).unwrap();
    app.run("layer.new.layer", json!({"name": "Same/name"})).unwrap();
    app.run("paint.bucket", json!({"x": 0, "y": 0, "color": "#00ff00"})).unwrap();
    app.run("select.allLayers", json!({})).unwrap();
    let selected = app.session.active().unwrap().selected_layers();
    let before = app.session.active().unwrap().doc.clone();
    let sources = Rc::new(RefCell::new(Vec::new()));
    let capture = sources.clone();
    app.services.export = Some(Box::new(move |doc, path, _| {
        assert_eq!(doc.layers.len(), 1, "each PNG must contain just its source layer");
        let png = photocraft_io::export(doc, path, &Default::default()).unwrap();
        assert!(png.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        capture.borrow_mut().push(doc.name.clone());
        Ok((png.bytes, Vec::new()))
    }));
    let ctx = egui::Context::default();
    assert!(!crate::layer_menu_ui::single_layer_only("layer.quickExportAsPng"));
    menus::invoke(&mut app, &ctx, "layer.quickExportAsPng", Value::Null).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(open.borrow().len(), 1, "one destination for the entire selection");
    // Selection changes while choosing a destination must not change the captured sources.
    app.run("layer.select", json!({"layer": selected[0].0})).unwrap();
    answer(&open, Some(FileDialogAnswer::SaveTo("selected.png".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(written.borrow().len(), selected.len());
    assert_eq!(sources.borrow().len(), selected.len());
    assert_eq!(written.borrow()[0], "selected.png");
    assert_eq!(written.borrow()[1], "selected-2-Same_name.png");
    assert_eq!(selected.len(), 2, "Select All Layers leaves the locked Background out");
    assert_eq!(app.session.active().unwrap().doc, before);
    assert!(!app.ui.status_error, "{}", app.ui.status);
}

#[test]
fn cancelling_multi_layer_quick_export_writes_nothing() {
    let (mut app, open, written) = app();
    for color in ["#ff0000", "#00ff00"] {
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("paint.bucket", json!({"x": 0, "y": 0, "color": color})).unwrap();
    }
    app.run("select.allLayers", json!({})).unwrap();
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "layer.quickExportAsPng", Value::Null).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    assert!(written.borrow().is_empty());
}

/// #3354: a system with no dialog backend answers `Unavailable`; the message reaches the status
/// bar as an error, where a cancelled dialog stays silent.
#[test]
fn an_unavailable_dialog_is_reported_and_a_cancelled_one_is_not() {
    let ctx = egui::Context::default();
    let (mut app, open, _) = app();
    menus::invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, None);
    app.poll_file_dialog(&ctx, None);
    assert!(!app.ui.status_error, "Cancel is the user's choice");
    assert!(!app.file_dialog_open());

    menus::invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    answer(&open, Some(FileDialogAnswer::Unavailable("No file dialog can open".into())));
    app.poll_file_dialog(&ctx, None);
    assert!(app.ui.status_error);
    assert_eq!(app.ui.status, "No file dialog can open");
    assert!(!app.file_dialog_open(), "the pending dialog is over");
}

/// #3629: Layer › Export As with several artboards selected exports one file per board, at each
/// board's size and with the dialog's settings, after one destination prompt.
#[test]
fn export_as_with_several_artboards_selected_writes_one_file_each() {
    let (mut app, open, written) = app();
    let a = app.run("layer.new.artboard", json!({"rect": [0, 0, 40, 30], "name": "Header"})).unwrap()["layer"].as_u64().unwrap();
    let b = app.run("layer.new.artboard", json!({"rect": [60, 0, 20, 50], "name": "Capsule"})).unwrap()["layer"].as_u64().unwrap();
    app.run("layer.select", json!({"layer": a})).unwrap();
    app.run("layer.select", json!({"layer": b, "mode": "add"})).unwrap();
    assert_eq!(app.session.active().unwrap().selected_layers().len(), 2);
    let sizes = Rc::new(RefCell::new(Vec::new()));
    let capture = sizes.clone();
    app.services.export = Some(Box::new(move |doc: &photocraft_doc::Document, _: &str, _: &crate::ExportSettings| {
        capture.borrow_mut().push((doc.name.clone(), doc.size.width, doc.size.height));
        Ok((b"out".to_vec(), Vec::new()))
    }));
    assert!(!crate::layer_menu_ui::single_layer_only("layer.exportAs"), "the menu offers it for a selection");
    let ctx = egui::Context::default();
    let r = menus::invoke(&mut app, &ctx, "layer.exportAs", Value::Null).unwrap();
    let id = r["dialog"].as_u64().unwrap();
    let mut fields = app.ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.clone();
    assert_eq!(fields["__label"], "Export As: 2 layers");
    // The dialog's settings apply to every file: JPG at 50 %.
    fields.insert("format".into(), json!("jpg"));
    fields.insert("scale".into(), json!(50));
    crate::export_dialog::confirm(&mut app, &fields).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(open.borrow().len(), 1, "one destination for every board");
    answer(&open, Some(FileDialogAnswer::SaveTo("/out/boards.jpg".into())));
    app.poll_file_dialog(&ctx, None);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(written.borrow().len(), 2);
    assert_eq!(written.borrow()[0], "/out/boards.jpg");
    // The sibling path is built with `Path::join`, so Windows turns the separator into `\`.
    let second = written.borrow()[1].replace('\\', "/");
    assert!(second.starts_with("/out/boards-2-") && second.ends_with(".jpg"), "{:?}", written.borrow());
    let mut got = sizes.borrow().iter().map(|(_, w, h)| (*w, *h)).collect::<Vec<_>>();
    got.sort_unstable();
    assert_eq!(got, [(10, 25), (20, 15)], "each board at its own size, scaled to 50 %");
}

/// #3806: File › Save on an OpenRaster file that can't hold the document (here an adjustment
/// layer) doesn't overwrite it: Save As opens on a PSD with a notice saying what the .ora would
/// drop, and the PSD becomes the document's file. A plain document still saves back to its .ora.
#[test]
fn openraster_save_never_overwrites_what_it_cannot_hold() {
    let (mut app, open, written) = app();
    app.run("layer.new.layer", json!({})).unwrap();
    app.session.active_mut().unwrap().path = Some("/pics/art.ora".into());
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
    assert_eq!(*written.borrow(), ["/pics/art.ora"], "a plain document saves in place");
    app.run("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    let notices = app.ui.notices.len();
    menus::invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
    app.poll_file_dialog(&ctx, None);
    assert_eq!(written.borrow().len(), 1, "the .ora is not overwritten");
    assert!(matches!(open.borrow().as_slice(), [(FileDialogRequest::Save { suggested }, _)] if suggested == "/pics/art.psd"), "Save As offers PSD");
    assert_eq!(app.ui.notices.len(), notices + 1);
    assert!(app.ui.notices.last().unwrap().lines.iter().any(|l| l.contains("adjustment layers")), "{:?}", app.ui.notices.last().unwrap().lines);
    assert!(app.session.active().unwrap().is_dirty(), "the edits are still unsaved");
    answer(&open, Some(FileDialogAnswer::SaveTo("/pics/art.psd".into())));
    app.poll_file_dialog(&ctx, None);
    assert_eq!(written.borrow().last().map(String::as_str), Some("/pics/art.psd"));
    let st = app.session.active().unwrap();
    assert_eq!(st.path.as_deref(), Some("/pics/art.psd"));
    assert!(!st.is_dirty());
}

/// Save As to OpenRaster of a document it can't hold writes a copy, as Photoshop does for formats
/// that can't keep everything: the document keeps its file and its unsaved state. An automation
/// save without a path won't overwrite such an .ora either.
#[test]
fn save_as_openraster_of_what_it_cannot_hold_is_a_copy() {
    let (mut app, _open, written) = app();
    app.run("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    app.session.active_mut().unwrap().path = Some("/pics/art.psd".into());
    let ctx = egui::Context::default();
    menus::invoke(&mut app, &ctx, "file.saveAs", json!({"path": "/pics/copy.ora"})).unwrap();
    assert_eq!(*written.borrow(), ["/pics/copy.ora"]);
    let st = app.session.active().unwrap();
    assert_eq!(st.path.as_deref(), Some("/pics/art.psd"), "the document keeps its file");
    assert!(st.is_dirty());
    app.session.active_mut().unwrap().path = Some("/pics/art.ora".into());
    let err = app.save_automation(None).unwrap_err();
    assert!(err.contains("OpenRaster"), "{err}");
    assert_eq!(written.borrow().len(), 1);
}
