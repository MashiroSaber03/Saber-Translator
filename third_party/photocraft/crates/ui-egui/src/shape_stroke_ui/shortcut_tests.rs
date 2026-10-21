//! Stroke popups own plain keys before the application's tool shortcuts run.
use egui::{Key, Modifiers, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use serde_json::json;

use crate::{PhotocraftApp, state::Tool};

fn app(tool: Tool) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 80, "height": 60})).unwrap();
    app.ui.tool = tool;
    app.ui.tool_options.vector_mode = "shape".into();
    app.ui.tool_options.stroke_width = 3.0;
    app.sync_views();
    app
}

fn harness(app: PhotocraftApp, properties: bool) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1000.0, 750.0)).build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            let _language = crate::i18n::language_scope(crate::i18n::Lang::EN);
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            // Match PhotocraftApp::update: source validation and application shortcuts precede
            // drawing the controls, so their popup handlers have not seen this frame's keys.
            super::sync_editor(app, ui.ctx());
            crate::shortcuts::handle(app, ui.ctx());
            if properties {
                let layer = app.session.active().unwrap().active_layer.unwrap();
                ui.set_max_width(320.0);
                crate::vector_ui::shape_properties(app, ui, layer);
            } else {
                let tool = app.ui.tool;
                ui.horizontal(|ui| crate::vector_ui::options_bar(app, ui, tool));
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ProMedium);
    h.run_steps(4);
    h
}

fn open(h: &mut Harness<'_, PhotocraftApp>) {
    h.get_by_role_and_label(egui::accesskit::Role::Button, "Stroke Options").click();
    h.run_steps(3);
    assert!(h.state().ui.stroke_editor.is_some());
}

#[test]
fn escape_dismisses_stroke_child_then_parent_without_deselecting_the_document() {
    let mut app = app(Tool::Rectangle);
    app.run("select.rect", json!({"x": 8, "y": 8, "width": 24, "height": 20})).unwrap();
    let original = app.session.active().unwrap().doc.clone();
    let history = app.session.active().unwrap().history.past_len();
    let mut h = harness(app, false);
    open(&mut h);
    h.get_by_role_and_label(egui::accesskit::Role::ComboBox, "Alignment").click();
    h.run_steps(3);
    assert!(egui::Popup::is_any_open(&h.ctx));
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(!egui::Popup::is_any_open(&h.ctx));
    assert!(h.state().ui.stroke_editor.is_some(), "the first Escape closes only the child");
    assert_eq!(h.state().session.active().unwrap().doc, original);
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(h.state().ui.stroke_editor.is_none(), "the second Escape closes the parent");
    assert_eq!(h.state().session.active().unwrap().doc, original);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(h.state().session.active().unwrap().doc.selection.is_none(), "normal Escape resumes after the popup closes");
}

#[test]
fn enter_selects_a_stroke_style_and_escape_closes_its_popups_without_committing_pen() {
    let mut app = app(Tool::Pen);
    app.ui.pen = Some(crate::vector_ui::PenPath { knots: vec![[[12.0, 12.0]; 3], [[60.0, 40.0]; 3]], ..Default::default() });
    let pen = app.ui.pen.clone();
    let original = app.session.active().unwrap().doc.clone();
    let history = app.session.active().unwrap().history.past_len();
    let mut h = harness(app, false);
    open(&mut h);
    h.get_by_role_and_label(egui::accesskit::Role::RadioButton, "Dotted").focus();
    h.run_steps(2);
    h.key_press(Key::Enter);
    h.run_steps(3);
    assert_eq!(h.state().ui.tool_options.shape_stroke.dashes, [0.0, 2.0]);
    assert_eq!(h.state().ui.pen, pen, "Enter belongs to the focused style target");
    h.get_by_role_and_label(egui::accesskit::Role::ComboBox, "Caps").click();
    h.run_steps(3);
    assert!(egui::Popup::is_any_open(&h.ctx));
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(!egui::Popup::is_any_open(&h.ctx));
    assert!(h.state().ui.stroke_editor.is_some());
    assert_eq!(h.state().ui.pen, pen);
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(h.state().ui.stroke_editor.is_none());
    assert_eq!(h.state().ui.pen, pen, "dismissing Stroke Options leaves the Pen gesture intact");
    assert_eq!(h.state().session.active().unwrap().doc, original);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
}

#[test]
fn undo_shortcut_remains_available_and_invalidates_the_open_stroke_editor() {
    let mut app = app(Tool::Rectangle);
    app.run("shape.create", json!({"rect": [10, 10, 40, 30], "stroke": {"width": 4, "color": "#dd2244"}})).unwrap();
    let original = app.session.active().unwrap().doc.clone();
    let history = app.session.active().unwrap().history.past_len();
    let mut h = harness(app, true);
    open(&mut h);
    h.get_by_role_and_label(egui::accesskit::Role::RadioButton, "Dotted").click();
    h.run_steps(3);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
    assert!(h.state().ui.stroke_editor.is_some());
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run_steps(3);
    assert_eq!(h.state().session.active().unwrap().doc, original);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    assert!(h.state().ui.stroke_editor.is_none(), "Undo invalidates the editor's last applied style");
    assert!(!egui::Popup::is_any_open(&h.ctx));
}
