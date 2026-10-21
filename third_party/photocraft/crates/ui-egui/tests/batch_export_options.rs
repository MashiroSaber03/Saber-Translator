//! #2399: batch conversion exposes the formats and quality its engine already supports.
//! The menu controls are exercised separately from the existing engine's decoded outputs.

use egui_kittest::kittest::Queryable;
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image};
use photocraft_ui_egui::{PhotocraftApp, menus};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

fn harness() -> Harness {
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1100.0, 760.0)).with_max_steps(64).wgpu().build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.prefs.edit(|p| p.interface.language = "en".into());
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            app.set_wgpu(rs.clone());
        }
        app.run("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
        app.run("actions.record", json!({"name": "Invert"})).unwrap();
        app.run("image.adjustments.invert", json!({})).unwrap();
        app.run("actions.stop", json!({})).unwrap();
        app
    });
    h.run_steps(4);
    h
}

fn open(h: &mut Harness, command: &str) {
    let ctx = h.ctx.clone();
    menus::invoke(h.state_mut(), &ctx, command, json!({})).expect("open the real menu dialog");
    h.run_steps(5);
}

fn evidence(h: &mut Harness, name: &str) {
    if let Some(dir) = std::env::var_os("PHOTOCRAFT_BATCH_EVIDENCE_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        h.render().expect("offscreen UI render").save(Path::new(&dir).join(format!("{name}.png"))).unwrap();
    }
}

fn open_format_choices(h: &mut Harness) {
    let label = h.get_by_label("Format").rect();
    // The full app also draws the workspace and layer-blend dropdowns behind this modal.
    // Target the actual form field beside Format, like the existing numeric-field harnesses.
    h.query_all_by_role(egui::accesskit::Role::ComboBox)
        .find(|n| {
            let r = n.rect();
            r.left() >= label.right() && (r.center().y - label.center().y).abs() < 3.0
        })
        .expect("format dropdown beside its label")
        .click();
    h.run_steps(3);
}

fn choose_webp(h: &mut Harness) {
    open_format_choices(h);
    h.query_by_label("WebP").or_else(|| h.query_by_label("Webp")).expect("WebP output choice").click();
    h.run_steps(3);
}

fn replace_text(h: &mut Harness, value: &str) {
    h.event(egui::Event::Key { key: egui::Key::A, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::COMMAND });
    h.event(egui::Event::Key { key: egui::Key::A, physical_key: None, pressed: false, repeat: false, modifiers: egui::Modifiers::COMMAND });
    h.event(egui::Event::Text(value.into()));
    h.run_steps(3);
}

fn edit_path(h: &mut Harness, name: &str, value: &Path) {
    let label = h.get_by_label(name).rect();
    h.query_all_by_role(egui::accesskit::Role::TextInput)
        .find(|n| {
            let r = n.rect();
            r.left() >= label.right() && (r.center().y - label.center().y).abs() < 3.0
        })
        .expect("path text field beside its label")
        .click();
    h.run_steps(2);
    replace_text(h, &value.to_string_lossy());
    h.get_by_label(name).click();
    h.run_steps(2);
}

fn edit_quality(h: &mut Harness, value: &str) {
    let label = h.get_by_label("Quality").rect();
    h.query_all_by_role(egui::accesskit::Role::SpinButton)
        .find(|n| {
            let r = n.rect();
            r.left() >= label.right() && (r.center().y - label.center().y).abs() < 3.0
        })
        .expect("quality numeric field beside its label")
        .click();
    h.run_steps(2);
    replace_text(h, value);
    h.get_by_label("Quality").click();
    h.run_steps(3);
}

#[test]
fn batch_menu_offers_webp() {
    let _gpu = photocraft_testkit::gpu_lock();
    let mut h = harness();
    open(&mut h, "file.automate.batch");
    open_format_choices(&mut h);
    evidence(&mut h, "batch-formats");
    assert!(h.query_by_label("WebP").or_else(|| h.query_by_label("Webp")).is_some(), "Batch must offer the supported WebP output format");
}

#[test]
fn batch_menu_offers_quality() {
    let _gpu = photocraft_testkit::gpu_lock();
    let mut h = harness();
    open(&mut h, "file.automate.batch");
    evidence(&mut h, "batch-quality");
    assert!(h.query_by_label("Quality").is_some(), "Batch must expose its supported output quality");
}

#[test]
fn image_processor_menu_offers_webp() {
    let _gpu = photocraft_testkit::gpu_lock();
    let mut h = harness();
    open(&mut h, "file.scripts.imageProcessor");
    open_format_choices(&mut h);
    evidence(&mut h, "image-processor-formats");
    assert!(h.query_by_label("WebP").or_else(|| h.query_by_label("Webp")).is_some(), "Image Processor must offer the supported WebP output format");
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("photocraft-batch-options-{}-{now}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pattern() -> Image {
    let mut data = Vec::new();
    for y in 0..48u32 {
        for x in 0..64u32 {
            data.extend([(x * 13 + y * 3) as u8, (y * 17 + x * 5) as u8, ((x ^ y) * 11) as u8]);
        }
    }
    Image::from_u8(64, 48, ChannelLayout::Rgb, data).unwrap()
}

#[test]
fn existing_engine_batch_and_image_processor_support_webp_quality() {
    let dir = Scratch::new();
    let input = dir.0.join("owned-pattern.png");
    let source = pattern();
    std::fs::write(&input, photocraft_codecs::encode(&source, Format::Png, &EncodeOptions::default()).unwrap()).unwrap();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
    let doc = app.session.active().unwrap().doc.clone();
    let revision = app.session.active().unwrap().revision;
    let history = app.session.active().unwrap().history.entries();
    let prefs = serde_json::to_value(app.session.prefs()).unwrap();

    for (command, prefix) in [("file.automate.batch", "batch"), ("file.scripts.imageProcessor", "processor")] {
        let expected: Vec<u8> = source.data().iter().map(|v| if prefix == "batch" { 255 - v } else { *v }).collect();
        let mut outputs = Vec::new();
        for quality in [Value::Null, json!(0), json!(12)] {
            let out = dir.0.join(format!("{prefix}-{quality}"));
            std::fs::create_dir(&out).unwrap();
            let result = app
                .run(command, json!({"input": [input], "output": out, "steps": [["image.adjustments.invert", {}]], "format": "webp", "quality": quality}))
                .unwrap();
            assert_eq!(result["errors"], json!([]), "{command}: {result}");
            let path = result["files"][0].as_str().unwrap();
            assert!(path.ends_with(".webp"));
            let bytes = std::fs::read(path).unwrap();
            let decoded = photocraft_codecs::decode_as(Format::WebP, &bytes).unwrap();
            assert_eq!(decoded.dimensions(), (64, 48));
            assert_eq!(decoded.layout(), ChannelLayout::Rgb);
            if quality.is_null() && prefix == "batch" {
                assert_eq!(decoded.data(), expected, "default Batch WebP is lossless, and the action was applied");
            }
            outputs.push(bytes);
        }
        assert_ne!(outputs[1], outputs[2], "{command}: low and high quality must affect the encoder");
        assert!(outputs[1].len() < outputs[2].len(), "{command}: this synthetic pattern grows at higher quality");
    }
    let active = app.session.active().unwrap();
    assert!(Arc::ptr_eq(&active.doc, &doc), "batch commands leave the active document untouched");
    assert_eq!(active.revision, revision);
    assert_eq!(active.history.entries(), history);
    assert_eq!(serde_json::to_value(app.session.prefs()).unwrap(), prefs);
}

#[test]
fn batch_custom_quality_is_bounded_dialog_local_and_reversible() {
    let _gpu = photocraft_testkit::gpu_lock();
    let mut h = harness();
    open(&mut h, "file.automate.batch");
    assert!(h.state().ui.dialogs.last().unwrap().fields["quality"].is_null(), "default command quality remains unset");
    h.get_by_label("Custom").click();
    h.run_steps(3);
    edit_quality(&mut h, "-5");
    assert_eq!(h.state().ui.dialogs.last().unwrap().fields["quality"].as_f64(), Some(0.0));
    edit_quality(&mut h, "99");
    assert_eq!(h.state().ui.dialogs.last().unwrap().fields["quality"].as_f64(), Some(12.0));
    edit_quality(&mut h, "5");
    h.get_by_label("Custom").click();
    h.run_steps(3);
    assert!(h.state().ui.dialogs.last().unwrap().fields["quality"].is_null());
    h.get_by_label("Custom").click();
    h.run_steps(3);
    assert_eq!(h.state().ui.dialogs.last().unwrap().fields["quality"].as_f64(), Some(5.0), "switching defaults does not lose the dialog's custom value");
    h.get_by_label("Cancel").click();
    h.run_steps(3);
    open(&mut h, "file.automate.batch");
    assert!(h.state().ui.dialogs.last().unwrap().fields["quality"].is_null(), "reopening returns to the existing default");
}

#[test]
fn batch_and_processor_real_controls_write_the_chosen_webp_quality() {
    let _gpu = photocraft_testkit::gpu_lock();
    let dir = Scratch::new();
    let input = dir.0.join("input");
    std::fs::create_dir(&input).unwrap();
    let original = pattern();
    std::fs::write(input.join("owned-pattern.png"), photocraft_codecs::encode(&original, Format::Png, &EncodeOptions::default()).unwrap()).unwrap();
    let mut h = harness();
    let doc = h.state().session.active().unwrap().doc.clone();
    let revision = h.state().session.active().unwrap().revision;
    let history = h.state().session.active().unwrap().history.entries();
    let prefs = serde_json::to_value(h.state().session.prefs()).unwrap();
    let actions = serde_json::to_value(&h.state().session.actions.list).unwrap();
    for (command, prefix) in [("file.automate.batch", "batch"), ("file.scripts.imageProcessor", "processor")] {
        let mut outputs = Vec::new();
        for quality in [0, 12] {
            let output = dir.0.join(format!("{prefix}-{quality}"));
            open(&mut h, command);
            if prefix == "batch" {
                assert!(h.state().ui.dialogs.last().unwrap().fields["quality"].is_null());
                h.get_by_label("Custom").click();
                h.run_steps(3);
            } else {
                let fields = &h.state().ui.dialogs.last().unwrap().fields;
                assert_eq!(fields["quality"], json!(8));
                assert_eq!(fields["width"], json!(0));
                assert_eq!(fields["height"], json!(0));
                assert_eq!(fields["convertToSrgb"], json!(true));
            }
            choose_webp(&mut h);
            edit_path(&mut h, "Input", &input);
            edit_path(&mut h, "Output", &output);
            edit_quality(&mut h, &quality.to_string());
            evidence(&mut h, &format!("{prefix}-webp-quality-{quality}"));
            h.get_by_label("OK").click();
            h.run_steps(5);
            assert!(h.state().ui.dialogs.is_empty());
            let bytes = std::fs::read(output.join("owned-pattern.webp")).expect("the actual dialog confirmation writes the selected WebP");
            let decoded = photocraft_codecs::decode_as(Format::WebP, &bytes).unwrap();
            assert_eq!(decoded.dimensions(), (64, 48));
            assert_eq!(decoded.layout(), ChannelLayout::Rgb);
            let (_, params) = h.state().session.journal.last().unwrap();
            assert_eq!(params["format"], json!("webp"));
            assert_eq!(params["quality"].as_f64(), Some(f64::from(quality)));
            assert!(params.as_object().unwrap().keys().all(|k| !k.starts_with("__")), "private shell values never become command params");
            outputs.push(bytes);
        }
        assert_ne!(outputs[0], outputs[1], "{command}: actual chosen quality reaches the encoder");
        assert!(outputs[0].len() < outputs[1].len());
    }
    let active = h.state().session.active().unwrap();
    assert!(Arc::ptr_eq(&active.doc, &doc));
    assert_eq!((active.revision, active.history.entries()), (revision, history));
    assert_eq!(serde_json::to_value(h.state().session.prefs()).unwrap(), prefs);
    assert_eq!(serde_json::to_value(&h.state().session.actions.list).unwrap(), actions);
}

#[test]
fn batch_naming_controls_export_the_requested_names_and_reset_on_cancel() {
    let _gpu = photocraft_testkit::gpu_lock();
    let dir = Scratch::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    std::fs::create_dir(&input).unwrap();
    let source = pattern();
    let bytes = photocraft_codecs::encode(&source, Format::Png, &EncodeOptions::default()).unwrap();
    for name in ["first.png", "second.png"] {
        std::fs::write(input.join(name), &bytes).unwrap();
    }
    let mut h = harness();
    open(&mut h, "file.automate.batch");
    h.get_by_role_and_label(egui::accesskit::Role::TextInput, "File Naming").click();
    h.run_steps(2);
    replace_text(&mut h, "cancelled_{serial}");
    h.get_by_label("Cancel").click();
    h.run_steps(3);
    assert!(!output.exists());
    open(&mut h, "file.automate.batch");
    let fields = &h.state().ui.dialogs.last().unwrap().fields;
    assert_eq!(fields["naming"], "");
    assert_eq!(fields["startingSerial"], 1);
    edit_path(&mut h, "Input", &input);
    edit_path(&mut h, "Output", &output);
    h.get_by_role_and_label(egui::accesskit::Role::TextInput, "File Naming").click();
    h.run_steps(2);
    replace_text(&mut h, "{serial}_{name}");
    let label = h.get_by_label("Starting serial").rect();
    h.query_all_by_role(egui::accesskit::Role::SpinButton)
        .find(|n| {
            let r = n.rect();
            r.left() >= label.right() && r.left() < label.right() + 100.0 && (r.center().y - label.center().y).abs() < 3.0
        })
        .expect("starting serial number control")
        .click();
    h.run_steps(2);
    h.event(egui::Event::Text("7".into()));
    h.run_steps(3);
    h.get_by_label("Starting serial").click();
    h.run_steps(3);
    assert_eq!(h.state().ui.dialogs.last().unwrap().fields["startingSerial"], 7, "typing changes the starting serial");
    evidence(&mut h, "batch-naming");
    h.get_by_label("OK").click();
    h.run_steps(5);
    assert!(h.state().ui.dialogs.is_empty());
    let expected: Vec<_> = source.data().iter().map(|v| 255 - v).collect();
    for name in ["007_first.png", "008_second.png"] {
        let bytes = std::fs::read(output.join(name)).expect("dialog naming reaches the exporter");
        let decoded = photocraft_codecs::decode_as(Format::Png, &bytes).unwrap();
        assert_eq!(decoded.data(), expected, "the recorded Invert action still runs");
    }
    assert!(!output.join("first.png").exists());
    assert!(!output.join("second.png").exists());
}

#[test]
fn batch_default_quality_keeps_lossless_webp_and_existing_jpeg_bytes() {
    let _gpu = photocraft_testkit::gpu_lock();
    let dir = Scratch::new();
    let input = dir.0.join("input");
    std::fs::create_dir(&input).unwrap();
    let original = pattern();
    std::fs::write(input.join("owned-pattern.png"), photocraft_codecs::encode(&original, Format::Png, &EncodeOptions::default()).unwrap()).unwrap();
    let mut h = harness();
    for format in ["webp", "jpg"] {
        let output = dir.0.join(format!("ui-default-{format}"));
        let reference = dir.0.join(format!("engine-existing-default-{format}"));
        open(&mut h, "file.automate.batch");
        if format == "webp" {
            choose_webp(&mut h);
        } else {
            open_format_choices(&mut h);
            h.get_by_label("Jpg").click();
            h.run_steps(3);
        }
        edit_path(&mut h, "Input", &input);
        edit_path(&mut h, "Output", &output);
        assert!(h.state().ui.dialogs.last().unwrap().fields["quality"].is_null());
        evidence(&mut h, &format!("batch-default-{format}"));
        h.get_by_label("OK").click();
        h.run_steps(5);
        let result = std::fs::read(output.join(format!("owned-pattern.{format}"))).unwrap();
        let mut session = photocraft_engine::Session::new();
        let r = session
            .execute("file.automate.batch", json!({"input": input, "output": reference, "steps": [["image.adjustments.invert", {}]], "format": format}))
            .unwrap();
        assert_eq!(r["errors"], json!([]));
        assert_eq!(
            result,
            std::fs::read(reference.join(format!("owned-pattern.{format}"))).unwrap(),
            "{format}: UI default equals the command's existing unset quality"
        );
        if format == "webp" {
            let decoded = photocraft_codecs::decode_as(Format::WebP, &result).unwrap();
            let expected: Vec<u8> = original.data().iter().map(|v| 255 - v).collect();
            assert_eq!(decoded.data(), expected, "default WebP remains lossless after the recorded action");
        }
    }
}
