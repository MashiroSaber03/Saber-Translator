//! #2061: the scroll wheel over the options bar's Mode dropdown (Brush, Pencil, Gradient) steps
//! through the blend modes, as over the Layers panel's (#1747) and as in Photoshop.

use egui::accesskit::Role;
use egui::{Pos2, Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::BlendMode;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// The app with a new document and `tool` chosen.
fn harness(tool: Tool) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        app.ui.tool = tool;
        app
    });
    h.run_steps(8);
    h
}

/// The options bar's Mode dropdown: the highest one showing `Normal` (the Layers panel's is lower).
fn mode_dropdown(h: &Harness<'_, PhotocraftApp>) -> Rect {
    h.query_all_by_role(Role::ComboBox).filter(|n| n.value().as_deref() == Some("Normal")).map(|n| n.rect()).min_by(|a, b| a.top().total_cmp(&b.top())).unwrap()
}

/// `lines` wheel lines at `at` in one event: negative is down.
fn wheel(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, lines: f32) {
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Line,
        delta: vec2(0.0, lines),
        phase: egui::TouchPhase::Move,
        modifiers: Default::default(),
    });
    h.run_steps(3);
}

fn brush_mode(h: &Harness<'_, PhotocraftApp>) -> BlendMode {
    h.state().session.tools.brush.mode
}

#[test]
fn the_wheel_over_the_brush_mode_dropdown_steps_the_painting_mode() {
    let mut h = harness(Tool::Brush);
    let at = mode_dropdown(&h).center();
    assert_eq!(brush_mode(&h), BlendMode::Normal);
    wheel(&mut h, at, -1.0);
    assert_eq!(brush_mode(&h), BlendMode::Dissolve);
    wheel(&mut h, at, -1.0);
    assert!(h.state().ui.tool_options.brush_clear, "Clear follows Dissolve, as in Photoshop");
    wheel(&mut h, at, -1.0);
    assert!(!h.state().ui.tool_options.brush_clear);
    assert_eq!(brush_mode(&h), BlendMode::Darken);
    wheel(&mut h, at, 3.0);
    assert_eq!(brush_mode(&h), BlendMode::Normal, "up steps back and stops at the first mode");
}

#[test]
fn the_wheel_over_the_pencil_mode_dropdown_steps_the_painting_mode() {
    let mut h = harness(Tool::Pencil);
    let at = mode_dropdown(&h).center();
    wheel(&mut h, at, -1.0);
    assert_eq!(brush_mode(&h), BlendMode::Dissolve);
    wheel(&mut h, at, 1.0);
    assert_eq!(brush_mode(&h), BlendMode::Normal);
}

#[test]
fn the_wheel_over_the_gradient_mode_dropdown_steps_its_blend_mode() {
    let mut h = harness(Tool::Gradient);
    let at = mode_dropdown(&h).center();
    wheel(&mut h, at, -2.0);
    assert_eq!(h.state().ui.tool_options.gradient_blend_mode, BlendMode::Darken);
    wheel(&mut h, at, 1.0);
    assert_eq!(h.state().ui.tool_options.gradient_blend_mode, BlendMode::Dissolve);
}
