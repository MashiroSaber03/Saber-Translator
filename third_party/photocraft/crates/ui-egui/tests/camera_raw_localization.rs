//! Camera Raw labels and language changes through the real shared native/WASM shell.
use egui::{accesskit::Role, pos2, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_ui_egui::{PhotocraftApp, camera_raw_ui, control, i18n, theme::ThemeKind};
use serde_json::json;

#[test]
fn all_seven_languages_render_camera_raw_and_switch_without_resetting_view_or_filter() {
    let _gpu = photocraft_testkit::gpu_lock();
    for lang in i18n::Lang::all() {
        let gpu = std::env::var_os("PHOTOCRAFT_LOCALE_SNAPSHOTS").is_some();
        let builder = Harness::builder().with_step_dt(1.0 / 60.0).with_size(vec2(1440.0, 1000.0));
        let builder = if gpu { builder.wgpu() } else { builder };
        let mut h = builder.build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, ThemeKind::Pro);
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width":64,"height":48})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("edit.fill", json!({"color":"#80ff00"})).unwrap();
            camera_raw_ui::open(&mut app, &cc.egui_ctx).unwrap();
            app
        });
        h.run_steps(3);
        let ctx = h.ctx.clone();
        h.state_mut().set_theme(&ctx, ThemeKind::Pro);
        h.state_mut().session.prefs.edit(|p| p.interface.language = lang.code().into());
        h.run_steps(4);
        let tr = |s| i18n::tr_ctx(lang, "cameraRaw", s);
        let before = control::inspect(h.state(), &ctx)["cameraRaw"].clone();
        for section in ["Light", "Color", "Effects", "Curve", "Color Mixer", "Color Grading", "Detail"] {
            assert!(h.query_by_role_and_label(Role::Button, tr(section)).is_some(), "{}: {section}", lang.code());
        }
        assert!(h.query_by_label(tr("Blacks")).is_some());
        h.get_by_role_and_label(Role::Button, tr("Light")).click();
        h.run_steps(12);
        h.get_by_role_and_label(Role::Button, tr("Color")).click();
        h.run_steps(12);
        for label in ["White Balance: As Shot", "Temperature", "Tint", "Vibrance", "Saturation"] {
            assert!(h.query_by_label(tr(label)).is_some(), "{}: {label}", lang.code());
        }
        h.get_by_role_and_label(Role::Button, tr("Color")).click();
        h.run_steps(12);
        h.get_by_role_and_label(Role::Button, tr("Color Mixer")).click();
        h.run_steps(12);
        for tab in ["Hue", "Saturation", "Luminance"] {
            assert!(h.get_by_label(tr(tab)).rect().right() <= 1426.0, "{}: mixer tabs overflow the panel", lang.code());
        }
        for band in ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"] {
            assert!(h.query_by_label(tr(band)).is_some(), "{}: {band}", lang.code());
        }
        h.get_by_role_and_label(Role::Button, tr("Color Mixer")).click();
        h.run_steps(12);
        for (section, labels) in [
            ("Effects", vec!["Texture", "Clarity", "Dehaze", "Vignetting", "Grain", "Roughness"]),
            ("Curve", vec!["Highlights", "Lights", "Darks", "Shadows"]),
            ("Color Grading", vec!["Midtones", "Global", "Hue", "Saturation", "Luminance", "Blending", "Balance"]),
            ("Detail", vec!["Sharpening", "Masking", "Noise Reduction", "Luminance Detail", "Color Detail"]),
        ] {
            h.get_by_role_and_label(Role::Button, tr(section)).click();
            h.run_steps(12);
            for label in labels {
                assert!(h.query_all_by_label(tr(label)).next().is_some(), "{}: {section}/{label}", lang.code());
            }
            // Detail is also the sharpening slider label; target the section button.
            h.get_by_role_and_label(Role::Button, tr(section)).click();
            h.run_steps(12);
        }
        h.get_by_role_and_label(Role::Button, tr("Color Mixer")).click();
        h.run_steps(12);
        let graph = h.get_by_label(tr("Tone Histogram")).rect();
        h.event(egui::Event::PointerMoved(graph.center()));
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: graph.center(), button: egui::PointerButton::Secondary, pressed, modifiers: egui::Modifiers::NONE });
        }
        h.run_steps(3);
        for label in ["Show Lab Color Readouts", "Show Vectorscope"] {
            assert!(h.query_by_label(tr(label)).is_some(), "{}: scope menu/{label}", lang.code());
        }
        h.get_by_label(tr("Show Vectorscope")).click();
        h.run_steps(3);
        // The context menu stays open for its dependent scope options.
        for label in ["Show Red at 3 o'clock", "Show Skin Tone Indicator"] {
            assert!(h.query_by_label(tr(label)).is_some(), "{}: scope menu/{label}", lang.code());
        }
        let outside = pos2(100.0, 400.0);
        h.event(egui::Event::PointerMoved(outside));
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: outside, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
        }
        h.run_steps(3);
        let preview = control::inspect(h.state(), &ctx)["cameraRaw"]["previewRect"].clone();
        h.hover_at(pos2((preview[0].as_f64().unwrap() + preview[2].as_f64().unwrap()) as f32 * 0.5, 400.0));
        h.run_steps(3);
        if let Ok(dir) = std::env::var("PHOTOCRAFT_LOCALE_SNAPSHOTS") {
            std::fs::create_dir_all(&dir).unwrap();
            h.render().unwrap().save(std::path::Path::new(&dir).join(format!("{}.png", lang.code()))).unwrap();
        }
        // Changing labels must keep the same collapse IDs and open mixer, without an image edit.
        h.state_mut().session.prefs.edit(|p| p.interface.language = "en".into());
        h.run_steps(3);
        assert!(h.query_by_label("Aqua").is_some(), "{} → en: preserve the open section", lang.code());
        assert!(h.query_by_label("Exposure").is_none(), "{} → en: Light stays collapsed", lang.code());
        let after = control::inspect(h.state(), &ctx)["cameraRaw"].clone();
        assert_eq!(after["params"], before["params"]);
        assert_eq!(after["previewRevision"], before["previewRevision"]);
        assert_eq!(after["histogram"], before["histogram"]);
    }
}
