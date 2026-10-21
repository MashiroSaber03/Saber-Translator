//! Camera Raw through the real app shell and its existing agent entry point.
use egui::{Event, Modifiers, MouseWheelUnit, accesskit::Role, pos2, vec2};
use egui_kittest::{
    Harness,
    kittest::{NodeT, Queryable},
};
use photocraft_ui_egui::{PhotocraftApp, camera_raw_ui, control, theme::ThemeKind};
use serde_json::{Value, json};

fn invoke(h: &mut Harness<'_, PhotocraftApp>, params: Value) -> Value {
    let ctx = h.ctx.clone();
    let (request, _) = control::ControlRequest::new("ui.menu.invoke", json!({"id": "filter.cameraRaw", "params": params}));
    let outcome = control::handle(h.state_mut(), &ctx, &request);
    assert!(matches!(outcome, control::Outcome::Done(_)), "dialog control failed");
    control::inspect(h.state(), &ctx)["cameraRaw"].clone()
}

#[test]
fn header_stays_fixed_and_resize_hover_scroll_before_do_not_rebuild_the_proxy() {
    for theme in ThemeKind::ALL {
        let mut h = Harness::builder().with_size(vec2(1200.0, 800.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, theme);
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.set_theme(&cc.egui_ctx, theme);
            app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("edit.fill", json!({"color": "#708090"})).unwrap();
            app
        });
        h.run_steps(2);
        let original = invoke(&mut h, json!({}));
        h.run_steps(3);
        let label = photocraft_ui_egui::i18n::tr(photocraft_ui_egui::i18n::current(), "Tone Histogram");
        let header = h.query_by_label(label).expect("histogram graph").rect();
        h.event(Event::PointerMoved(pos2(1050.0, 400.0)));
        h.event(Event::MouseWheel { unit: MouseWheelUnit::Point, phase: egui::TouchPhase::Move, delta: vec2(0.0, -320.0), modifiers: Modifiers::NONE });
        h.run_steps(3);
        assert_eq!(h.query_by_label(label).expect("histogram graph").rect(), header, "{theme:?}: histogram scrolled");
        h.set_size(vec2(960.0, 600.0));
        h.set_pixels_per_point(2.0);
        h.run_steps(3);
        let before = invoke(&mut h, json!({"ui": {"before": true}}));
        h.run_steps(3);
        let inspected = control::inspect(h.state(), &h.ctx);
        assert_eq!(before["previewRevision"], original["previewRevision"]);
        assert_eq!(inspected["cameraRaw"]["previewRevision"], original["previewRevision"]);
        assert_eq!(before["histogram"]["red"], original["histogram"]["red"]);
        assert!(h.query_by_label(label).expect("histogram graph").rect().bottom() < 600.0);
    }
}

#[test]
fn histogram_matches_visible_preview_at_all_supported_depths_and_cancel_commit_are_reversible() {
    for mode in ["rgb", "grayscale"] {
        for depth in [8, 16, 32] {
            let ctx = egui::Context::default();
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width": 16, "height": 8, "mode": mode, "depth": depth})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("edit.fill", json!({"color": "#808080"})).unwrap();
            let original = app.session.active().unwrap().doc.clone();
            let steps = app.session.active().unwrap().history.past_len();
            let initial = camera_raw_ui::menu(&mut app, &ctx, "filter.cameraRaw", &json!({})).unwrap().unwrap();
            assert_eq!(initial["histogram"]["samples"], 128);
            let changed = camera_raw_ui::menu(&mut app, &ctx, "filter.cameraRaw", &json!({"ui": {"set": {"exposure": 1.0}}})).unwrap().unwrap();
            assert_ne!(changed["histogram"]["red"], initial["histogram"]["red"]);
            assert_eq!(app.session.active().unwrap().history.past_len(), steps);
            camera_raw_ui::menu(&mut app, &ctx, "filter.cameraRaw", &json!({"ui": {"cancel": true}})).unwrap().unwrap();
            assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
            camera_raw_ui::menu(&mut app, &ctx, "filter.cameraRaw", &json!({"ui": {"set": {"exposure": 1.0}, "commit": true}})).unwrap().unwrap();
            assert_eq!(app.session.active().unwrap().history.past_len(), steps + 1);
            app.run("edit.undo", json!({})).unwrap();
            assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
            app.run("edit.redo", json!({})).unwrap();
            let committed = camera_raw_ui::menu(&mut app, &ctx, "filter.cameraRaw", &json!({})).unwrap().unwrap();
            assert_eq!(committed["histogram"]["red"], changed["histogram"]["red"], "{mode}/{depth}: graph and committed pixels disagree");
        }
    }
}

#[test]
fn settings_sections_group_existing_controls_without_rendering_or_document_edits() {
    assert_eq!(photocraft_ui_egui::i18n::tr_ctx(photocraft_ui_egui::i18n::Lang::from_code("ru").unwrap(), "cameraRaw", "Light"), "Свет");
    for theme in ThemeKind::ALL {
        let mut h = Harness::builder().with_step_dt(1.0 / 60.0).with_size(vec2(1200.0, 900.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, theme);
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.set_theme(&cc.egui_ctx, theme);
            app.run("file.new", json!({"width":64,"height":48})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("edit.fill", json!({"color":"#708090"})).unwrap();
            app
        });
        let before = invoke(&mut h, json!({}));
        let doc = h.state().session.active().unwrap().doc.clone();
        h.run_steps(4);
        let lang = photocraft_ui_egui::i18n::current();
        let light = photocraft_ui_egui::i18n::tr_ctx(lang, "cameraRaw", "Light");
        // The dialog translates through the `cameraRaw` context; match it in any system language.
        let tr = |s| photocraft_ui_egui::i18n::tr_ctx(lang, "cameraRaw", s);
        let mut y = 0.0;
        for label in [light, tr("Color"), tr("Effects"), tr("Curve"), tr("Color Mixer"), tr("Color Grading"), tr("Detail")] {
            let next = h.get_by_role_and_label(Role::Button, label).rect().top();
            assert!(next > y, "section order: {label}");
            y = next;
        }
        assert!(h.query_by_label(tr("Basic")).is_none());
        assert!(h.query_by_label(tr("Temperature")).is_none());
        assert!(h.query_by_label(tr("Texture")).is_none());
        assert!(h.query_by_label(tr("Exposure")).is_some());
        h.get_by_role_and_label(Role::Button, light).click();
        h.run_steps(12);
        h.get_by_role_and_label(Role::Button, tr("Color")).click();
        h.run_steps(12);
        // The dock also has Saturation; query only the Camera Raw settings panel.
        let settings = std::iter::successors(Some(h.get_by_role_and_label(Role::Button, tr("Color"))), |node| node.parent())
            .find(|node| node.query_by_label(tr("Tone Histogram")).is_some())
            .expect("Camera Raw settings panel");
        for label in ["Temperature", "Tint", "Vibrance", "Saturation"] {
            assert!(settings.query_by_label(tr(label)).is_some(), "color control: {label}");
        }
        assert!(h.query_by_label(tr("Exposure")).is_none());
        h.get_by_role_and_label(Role::Button, tr("Color")).click();
        h.run_steps(12);
        h.get_by_role_and_label(Role::Button, tr("Effects")).click();
        h.run_steps(12);
        for label in ["Texture", "Clarity", "Dehaze", "Vignetting", "Grain"] {
            assert!(h.query_by_label(tr(label)).is_some(), "effects control: {label}");
        }
        let after = control::inspect(h.state(), &h.ctx);
        assert_eq!(after["cameraRaw"]["params"], before["params"]);
        assert_eq!(after["cameraRaw"]["previewRevision"], before["previewRevision"]);
        assert!(std::sync::Arc::ptr_eq(&doc, &h.state().session.active().unwrap().doc));
    }
}
