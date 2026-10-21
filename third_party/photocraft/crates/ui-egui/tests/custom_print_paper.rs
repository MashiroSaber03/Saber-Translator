//! #3820: edit custom paper through the real Print dialog, preview it, and print safely to PDF.
use egui::{Event, Key, Modifiers, PointerButton, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_ui_egui::{PhotocraftApp, file_ui::open_print};
use serde_json::{Map, Value, json};

fn app() -> (PhotocraftApp, u64) {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), photocraft_ui_egui::Services::default());
    app.run("file.new", json!({"width": 120, "height": 80, "name": "Custom paper"})).unwrap();
    let id = open_print(&mut app);
    // Every UI path in these fixtures is safe even if an unexpected Enter confirms the dialog.
    app.ui.dialog_mut(id).unwrap().fields.insert("dryRun".into(), json!(true));
    // Exercise PDF paper controls even when native printers are available, on every OS.
    app.ui.dialog_mut(id).unwrap().fields.insert("__printers".into(), json!(["Test native printer"]));
    (app, id)
}

fn show(ui: &mut egui::Ui, app: &mut PhotocraftApp) {
    if ui.ctx().fonts(|fonts| fonts.families().contains(&egui::FontFamily::Name("medium".into()))) {
        photocraft_ui_egui::dialogs::show(app, ui.ctx());
    }
}

fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1000.0, 900.0)).with_max_steps(64).build_ui_state(show, app);
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h
}

fn click(h: &mut Harness<'_, PhotocraftApp>, label: &str) {
    if let Some(node) = h.query_by_label(label) {
        node.click();
    } else {
        h.get_by_value(label).click();
    }
    h.run_steps(3);
}

fn shown(h: &Harness<'_, PhotocraftApp>, text: &str) -> bool {
    h.query_by_label(text).is_some() || h.query_by_value(text).is_some()
}

fn enter_length(h: &mut Harness<'_, PhotocraftApp>, label: &str, value: &str) {
    // Anchor the pointer to the real label; the existing length widget immediately follows it.
    let pos = h.get_by_label(label).rect().right_center() + vec2(35.0, 0.0);
    h.event(Event::PointerMoved(pos));
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.key_press_modifiers(Modifiers::COMMAND, Key::A);
    h.event(Event::Text(value.into()));
    h.run_steps(1);
    h.key_press(Key::Tab);
    h.run_steps(3);
}

fn fields(h: &Harness<'_, PhotocraftApp>, id: u64) -> Map<String, Value> {
    h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.clone()
}

#[test]
fn custom_paper_ui_edits_reach_preview_and_pdf_and_restore_when_reopened() {
    let (mut app, id) = app();
    let output = std::env::temp_dir().join(format!("pc-custom-paper-{}.pdf", std::process::id()));
    let dialog = app.ui.dialog_mut(id).unwrap();
    dialog.fields.insert("output".into(), json!(output.to_string_lossy()));
    dialog.fields.insert("dryRun".into(), json!(true));
    let mut h = harness(app);
    click(&mut h, "letter");
    assert!(shown(&h, "a4"), "the real preset menu must be open");
    assert!(shown(&h, "Custom"), "the paper menu must offer editable custom dimensions");
    click(&mut h, "Custom");
    assert_eq!(fields(&h, id)["paper"], json!([612.0, 792.0]), "Custom starts at the current preset dimensions");
    click(&mut h, "Centimeters");
    click(&mut h, "Inches");
    enter_length(&mut h, "Paper Width", "7.25");
    enter_length(&mut h, "Paper Height", "9.25");
    let edited = fields(&h, id);
    assert_eq!(edited["paper"], json!([522.0, 666.0]), "the engine takes points, not the displayed units");
    click(&mut h, "Custom");
    click(&mut h, "a4");
    assert_eq!(fields(&h, id)["paper"], "a4");
    assert!(!shown(&h, "Paper Width"));
    click(&mut h, "a4");
    click(&mut h, "Custom");
    assert_eq!(fields(&h, id)["paper"], edited["paper"], "returning from a preset restores the custom dimensions");
    click(&mut h, "Inches");
    click(&mut h, "Millimeters");
    assert_eq!(fields(&h, id)["paper"], edited["paper"], "changing units must not change the physical paper");
    click(&mut h, "Millimeters");
    click(&mut h, "Inches");
    assert_eq!(fields(&h, id)["paper"], edited["paper"]);
    let preview = photocraft_engine::print_cmds::layout(&h.state().session.active().unwrap().doc, &Value::Object(edited.clone()), "file.print").unwrap();
    assert_eq!(preview.paper, (522.0, 666.0));
    click(&mut h, "Portrait");
    click(&mut h, "Landscape");
    let edited = fields(&h, id);
    assert_eq!(edited["paper"], json!([522.0, 666.0]), "orientation must not rewrite the unrotated custom dimensions");
    let preview = photocraft_engine::print_cmds::layout(&h.state().session.active().unwrap().doc, &Value::Object(edited), "file.print").unwrap();
    assert_eq!(preview.paper, (666.0, 522.0));
    click(&mut h, "Print");
    let pdf = std::fs::read(&output).unwrap();
    assert!(String::from_utf8_lossy(&pdf).contains("/MediaBox [0 0 666.00 522.00]"));
    assert_eq!(h.state().session.file_menu.last_print.as_ref().unwrap()["paper"], json!([522.0, 666.0]));
    let id = open_print(h.state_mut());
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("dryRun".into(), json!(true));
    h.run_steps(3);
    assert!(shown(&h, "Custom"), "reopening array-valued settings must show Custom");
    assert_eq!(fields(&h, id)["paper"], json!([522.0, 666.0]));
    std::fs::remove_file(output).unwrap();
}

#[test]
fn remembered_custom_paper_is_shown_without_becoming_a_preset() {
    let (mut app, id) = app();
    app.ui.dialog_mut(id).unwrap().fields.insert("paper".into(), json!([432.0, 576.0]));
    let h = harness(app);
    assert!(shown(&h, "Custom"), "an array paper size must not display as letter");
    assert!(shown(&h, "Paper Width"));
    assert!(shown(&h, "Paper Height"));
    assert_eq!(fields(&h, id)["paper"], json!([432.0, 576.0]));
}

#[test]
fn invalid_custom_paper_is_rejected_without_printing_or_changing_the_document() {
    let (mut app, id) = app();
    let output = std::env::temp_dir().join(format!("pc-invalid-custom-paper-{}.pdf", std::process::id()));
    app.ui.dialog_mut(id).unwrap().fields.insert("output".into(), json!(output.to_string_lossy()));
    let before = app.session.active().unwrap().doc.clone();
    let mut h = harness(app);
    click(&mut h, "letter");
    click(&mut h, "Custom");
    enter_length(&mut h, "Paper Width", "0");
    let edited = fields(&h, id);
    assert!(photocraft_engine::print_cmds::layout(&h.state().session.active().unwrap().doc, &Value::Object(edited), "file.print").is_err());
    click(&mut h, "Print");
    let error = &h.state().ui.status;
    assert!(error.contains("paper is too small"), "{error}");
    assert_eq!(h.state().session.active().unwrap().doc, before);
    assert!(h.state().session.file_menu.last_print.is_none());
    assert!(!output.exists());
}
