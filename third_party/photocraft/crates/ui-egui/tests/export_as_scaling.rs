//! #1353: Export As must scale the height even when the rounded width stays unchanged.
//! Drive the real Scale field and Export button, then decode the PNG the writer received.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use egui_kittest::kittest::Queryable;
use photocraft_ui_egui::{FileDialogAnswer, FileDialogRequest, PhotocraftApp, Services, menus};
use serde_json::json;

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;
type Writes = Rc<RefCell<Vec<(String, Vec<u8>)>>>;

fn harness() -> Option<(Harness, Writes)> {
    let built = std::panic::catch_unwind(|| {
        let writes = Writes::default();
        let captured = writes.clone();
        let h = egui_kittest::Harness::builder().with_size(egui::vec2(1100.0, 760.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(
            move |cc| {
                PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                let services = Services {
                    // Save dialogs answer with the suggested name.
                    file_dialog: Some(Box::new(|request, _parent, reply| {
                        reply.send(match request {
                            FileDialogRequest::Save { suggested } => Some(FileDialogAnswer::SaveTo(suggested)),
                            FileDialogRequest::Folder { .. } => None,
                            FileDialogRequest::Open { .. } => None,
                        })
                    })),
                    export: Some(Box::new(|doc, path, settings| {
                        let opts = photocraft_io::ExportOptions {
                            xmp: if settings.xmp_all { photocraft_io::XmpEmbed::All } else { photocraft_io::XmpEmbed::None },
                            ..Default::default()
                        };
                        photocraft_io::export(doc, path, &opts).map(|r| (r.bytes, r.warnings)).map_err(|e| e.to_string())
                    })),
                    write: Some(Box::new(move |path, bytes| {
                        captured.borrow_mut().push((path.to_string(), bytes.to_vec()));
                        Ok(())
                    })),
                    ..Default::default()
                };
                let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
                app.session.prefs.edit(|p| p.interface.language = "en".into());
                if let Some(rs) = cc.wgpu_render_state.as_ref() {
                    app.set_wgpu(rs.clone());
                }
                app
            },
        );
        (h, writes)
    });
    match built {
        Ok(h) => Some(h),
        Err(_) => {
            eprintln!("skipping: no GPU adapter");
            None
        }
    }
}

fn evidence(h: &mut Harness, depth: u8, stage: &str) {
    if let Some(dir) = std::env::var_os("PHOTOCRAFT_EXPORT_EVIDENCE_DIR") {
        std::fs::create_dir_all(&dir).expect("evidence directory");
        h.render().expect("render UI").save(std::path::Path::new(&dir).join(format!("{stage}-{depth}bit.png"))).expect("save UI evidence");
    }
}

#[test]
fn export_as_scales_a_narrow_image_without_editing_the_source() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some((mut h, writes)) = harness() else { return };
    h.run_steps(4);
    for depth in [8, 16, 32] {
        writes.borrow_mut().clear();
        {
            let app = h.state_mut();
            while app.session.active().is_some() {
                app.run("file.close", json!({"discard": true})).expect("close");
            }
            app.run("file.new", json!({"width": 1, "height": 100, "depth": depth, "background": "white"})).expect("new narrow document");
            app.sync_views();
        }
        let st = h.state().session.active().expect("document");
        let source = st.doc.clone();
        let history = st.history.entries();
        let revision = st.revision;

        let ctx = h.ctx.clone();
        menus::invoke(h.state_mut(), &ctx, "file.export.exportAs", json!({})).expect("open Export As");
        h.run_steps(6);
        let label = h.get_by_label("Scale").rect();
        let field = h
            .query_all_by_role(egui::accesskit::Role::SpinButton)
            .find(|n| {
                let r = n.rect();
                r.left() >= label.right() && r.left() < label.right() + 100.0 && (r.center().y - label.center().y).abs() < 3.0
            })
            .expect("Scale numeric field");
        field.click();
        h.run_steps(2);
        // Clicking a numeric field selects its current value; type through egui's text input.
        h.event(egui::Event::Text("50".into()));
        h.run_steps(3);
        // Click away to commit the field without Enter confirming the dialog.
        h.get_by_label("Image Size").click();
        h.run_steps(3);
        assert!(h.query_by_label("1 × 50 px").is_some(), "{depth}-bit: dialog shows the requested output size");
        evidence(&mut h, depth, "dialog");

        h.get_by_label("Export").click();
        h.run_steps(6);
        evidence(&mut h, depth, "result");
        let written = writes.borrow();
        assert_eq!(written.len(), 1, "{depth}-bit: Export writes one file");
        let (path, bytes) = &written[0];
        assert!(path.ends_with(".png"), "PNG export path: {path}");
        if let Some(dir) = std::env::var_os("PHOTOCRAFT_EXPORT_EVIDENCE_DIR") {
            std::fs::write(std::path::Path::new(&dir).join(format!("exported-{depth}bit.png")), bytes).expect("save exported PNG evidence");
        }
        let png = photocraft_codecs::decode_as(photocraft_codecs::Format::Png, bytes).expect("decode the written PNG");
        assert_eq!(png.dimensions(), (1, 50), "{depth}-bit: written dimensions match the dialog");

        let st = h.state().session.active().expect("source remains open");
        assert!(Arc::ptr_eq(&st.doc, &source), "{depth}-bit: source document is unchanged");
        assert_eq!((st.doc.size.width, st.doc.size.height), (1, 100));
        assert_eq!(st.history.entries(), history, "export must not add source history");
        assert_eq!(st.revision, revision, "export must not change source revision");
    }
}
