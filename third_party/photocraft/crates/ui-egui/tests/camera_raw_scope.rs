//! Real dialog/controller acceptance: interactions never make a second document edit path.
use egui::{Event, Modifiers, PointerButton, Pos2, Rect, pos2, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_ui_egui::{PhotocraftApp, camera_raw_ui, control, theme::ThemeKind};
use serde_json::{Value, json};

use photocraft_testkit::gpu_lock;

fn menu(app: &mut PhotocraftApp, ctx: &egui::Context, p: Value) -> Result<Value, String> {
    camera_raw_ui::menu(app, ctx, "filter.cameraRaw", &p).ok_or("missing handler")?
}
fn inspect(h: &Harness<'_, PhotocraftApp>) -> Value {
    control::inspect(h.state(), &h.ctx)["cameraRaw"].clone()
}
fn rect(h: &Harness<'_, PhotocraftApp>, field: &str) -> Rect {
    let v = inspect(h);
    let a = v[field].as_array().unwrap();
    Rect::from_min_max(pos2(a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32), pos2(a[2].as_f64().unwrap() as f32, a[3].as_f64().unwrap() as f32))
}
fn fixture(theme: ThemeKind) -> Harness<'static, PhotocraftApp> {
    fixture_render(theme, false)
}
fn fixture_render(theme: ThemeKind, gpu: bool) -> Harness<'static, PhotocraftApp> {
    let builder = Harness::builder().with_step_dt(1.0 / 60.0).with_size(vec2(1200.0, 800.0));
    let builder = if gpu { builder.wgpu() } else { builder };
    let mut h = builder.build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, theme);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.set_theme(&cc.egui_ctx, theme);
        app.run("file.new", json!({"width":64,"height":48})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color":"#80ff00"})).unwrap();
        camera_raw_ui::open(&mut app, &cc.egui_ctx).unwrap();
        app
    });
    h.run_steps(3);
    // Startup preferences can replace the initial theme; apply the requested test theme after it.
    let ctx = h.ctx.clone();
    h.state_mut().set_theme(&ctx, theme);
    h.run_steps(3);
    assert_eq!(h.state().ui.theme, theme);
    h
}
fn press(h: &mut Harness<'_, PhotocraftApp>, p: Pos2, pressed: bool, modifiers: Modifiers) {
    h.event(Event::ModifiersChanged(modifiers));
    h.event(Event::PointerMoved(p));
    h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers });
    h.run_steps(2);
}
fn drag(h: &mut Harness<'_, PhotocraftApp>, a: Pos2, b: Pos2) {
    press(h, a, true, Modifiers::NONE);
    for n in 1..=4 {
        h.event(Event::PointerMoved(a + (b - a) * n as f32 / 4.0));
        h.run_steps(2);
    }
    press(h, b, false, Modifiers::NONE);
}

#[test]
fn invalid_mixed_controller_requests_leave_dialog_preferences_and_document_unchanged() {
    let mut h = fixture(ThemeKind::Pro);
    let ctx = h.ctx.clone();
    let before = inspect(&h);
    let preferences = h.state().session.prefs().dialogs.clone();
    let document = h.state().session.active().unwrap().doc.clone();
    for request in [
        json!({"set":{"exposure":1.0},"before":true,"scope":{"sample":[-1,0]}}),
        json!({"set":{"exposure":1.0},"scope":{"floatingRect":[0,0,1,1]}}),
        json!({"set":{"exposure":1.0},"scope":{"unknown":true}}),
        json!({"set":{"unknown":1.0}}),
        json!({"before":"yes"}),
        json!({"commit":true,"cancel":true}),
        json!({"unknown":true}),
    ] {
        assert!(menu(h.state_mut(), &ctx, json!({"ui":request})).is_err());
        assert_eq!(inspect(&h), before);
        assert_eq!(h.state().session.prefs().dialogs, preferences);
        assert!(std::sync::Arc::ptr_eq(&h.state().session.active().unwrap().doc, &document));
    }
    let result = menu(h.state_mut(), &ctx, json!({"ui":{"set":{"exposure":1.0},"scope":{"tone":{"zone":"exposure","delta":0.5}}}})).unwrap();
    assert_eq!(result["params"]["exposure"], 1.5);
    assert_eq!(result["previewRevision"].as_u64().unwrap(), before["previewRevision"].as_u64().unwrap() + 1, "one completed preview per transaction");
}

#[test]
fn invalid_requests_do_not_open_a_dialog_or_replace_scope_preferences() {
    let mut h = fixture(ThemeKind::Pro);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"cancel":true}})).unwrap();
    let view = h.state().ui.camera_raw_scope.clone();
    let prefs = h.state().session.prefs().dialogs.clone();
    for request in
        [json!(null), json!([]), json!(1), json!({"ui":[]}), json!({"ui":{"set":{"pointCurve":vec![[0,0];17]}}}), json!({"ui":{"scope":{"unknown":true}}})]
    {
        assert!(menu(h.state_mut(), &ctx, request).is_err());
        assert!(inspect(&h).is_null());
        assert_eq!(h.state().ui.camera_raw_scope, view);
        assert_eq!(h.state().session.prefs().dialogs, prefs);
    }
}

#[test]
fn failed_commit_keeps_the_dialog_and_filter_settings_for_retry() {
    let mut h = fixture(ThemeKind::Pro);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"set":{"exposure":0.5}}})).unwrap();
    h.state_mut().run("layer.lockLayers", json!({"pixels":true})).unwrap();
    let document = h.state().session.active().unwrap().doc.clone();
    assert!(menu(h.state_mut(), &ctx, json!({"ui":{"commit":true}})).is_err());
    assert_eq!(inspect(&h)["params"]["exposure"], 0.5);
    assert!(std::sync::Arc::ptr_eq(&h.state().session.active().unwrap().doc, &document));
    h.state_mut().run("layer.lockLayers", json!({"pixels":false})).unwrap();
    menu(h.state_mut(), &ctx, json!({"ui":{"commit":true}})).unwrap();
    assert!(inspect(&h).is_null());
}

#[test]
fn malformed_curves_fail_in_both_preview_and_engine_without_an_edit() {
    let mut h = fixture(ThemeKind::Pro);
    let ctx = h.ctx.clone();
    let before = inspect(&h);
    let document = h.state().session.active().unwrap().doc.clone();
    for field in ["pointCurve", "pointCurveRed", "pointCurveGreen", "pointCurveBlue"] {
        for points in [json!([[255, 255], [0, 0]]), json!([[0, 0], [0, 128]]), json!([[0, -1], [255, 255]]), json!(vec![[0, 0]; 17])] {
            let settings = json!({field:points});
            assert!(menu(h.state_mut(), &ctx, json!({"ui":{"set":settings}})).is_err());
            assert!(h.state_mut().run("filter.cameraRaw", settings).is_err());
            assert_eq!(inspect(&h), before);
            assert!(std::sync::Arc::ptr_eq(&h.state().session.active().unwrap().doc, &document));
        }
    }
}

#[test]
fn exact_clipping_rgb_lab_capped_samplers_and_reversible_tone_controls() {
    for mode in ["rgb", "grayscale"] {
        for depth in [8, 16, 32] {
            let ctx = egui::Context::default();
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width":16,"height":8,"mode":mode,"depth":depth})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("edit.fill", json!({"color":"#808080"})).unwrap();
            let original = app.session.active().unwrap().doc.clone();
            let history = app.session.active().unwrap().history.past_len();
            let initial = menu(&mut app, &ctx, json!({})).unwrap();
            for (zone, amount) in [("blacks", 20.0), ("shadows", 20.0), ("exposure", 0.5), ("highlights", 20.0), ("whites", 20.0)] {
                let change = menu(&mut app, &ctx, json!({"ui":{"scope":{"tone":{"zone":zone,"delta":amount}}}})).unwrap();
                assert_eq!(change["params"][zone], amount);
                assert!(change["previewRevision"].as_u64().unwrap() > initial["previewRevision"].as_u64().unwrap());
                menu(&mut app, &ctx, json!({"ui":{"scope":{"tone":{"zone":zone,"reset":true}}}})).unwrap();
            }
            for i in 0..9 {
                menu(&mut app, &ctx, json!({"ui":{"scope":{"addSampler":[i as f32/9.0,0.5]}}})).unwrap();
            }
            let reading = menu(&mut app, &ctx, json!({"ui":{"scope":{"lab":true,"sample":[0.5,0.5],"vectorscope":true}}})).unwrap();
            assert_eq!(reading["pointerReadout"]["space"], "lab");
            assert_eq!(reading["samplerReadouts"].as_array().unwrap().len(), 9);
            assert!(reading["pointerReadout"]["values"][0].as_f64().unwrap() > 45.0);
            for bad in [
                json!({"addSampler":[0.5,0.5]}),
                json!({"sample":[-1,0]}),
                json!({"sample":[1e30,0]}),
                json!({"samplers":vec![[0.5,0.5];10]}),
                json!({"removeSampler":999}),
                json!({"tone":{"zone":"nonsense","delta":1}}),
                json!({"floatingRect":[0,0,1,1]}),
                json!({"unknown":true}),
            ] {
                assert!(menu(&mut app, &ctx, json!({"ui":{"scope":bad}})).is_err());
            }
            let no_op = menu(&mut app, &ctx, json!({"ui":{"scope":{"lab":true,"redRight":true,"hideSkinLine":true,"floating":true}}})).unwrap();
            assert_eq!(no_op["previewRevision"], reading["previewRevision"]);
            assert_eq!(no_op["scopeRevision"], reading["scopeRevision"]);
            assert_eq!(app.session.active().unwrap().history.past_len(), history);
            assert!(std::sync::Arc::ptr_eq(&original, &app.session.active().unwrap().doc));
            menu(&mut app, &ctx, json!({"ui":{"cancel":true}})).unwrap();
            assert!(std::sync::Arc::ptr_eq(&original, &app.session.active().unwrap().doc));
            menu(&mut app, &ctx, json!({"ui":{"scope":{"tone":{"zone":"exposure","delta":0.5}},"commit":true}})).unwrap();
            assert_eq!(app.session.active().unwrap().history.past_len(), history + 1);
            app.run("edit.undo", json!({})).unwrap();
            assert!(std::sync::Arc::ptr_eq(&original, &app.session.active().unwrap().doc));
        }
    }
}

#[test]
fn lab_readouts_respect_embedded_rgb_profile() {
    let ctx = egui::Context::default();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width":4,"height":4})).unwrap();
    app.run("layer.new.layer", json!({})).unwrap();
    app.run("edit.fill", json!({"color":"#ff0000"})).unwrap();
    let srgb = menu(&mut app, &ctx, json!({"ui":{"scope":{"sample":[0.5,0.5],"lab":true}}})).unwrap();
    menu(&mut app, &ctx, json!({"ui":{"cancel":true}})).unwrap();
    app.run("edit.assignProfile", json!({"profile":"display-p3"})).unwrap();
    let p3 = menu(&mut app, &ctx, json!({"ui":{"scope":{"sample":[0.5,0.5],"lab":true}}})).unwrap();
    let a = srgb["pointerReadout"]["values"][1].as_f64().unwrap();
    let b = p3["pointerReadout"]["values"][1].as_f64().unwrap();
    assert!((a - b).abs() > 5.0, "Lab must use document ICC: {a} vs {b}");
}

#[test]
fn real_drag_hover_warnings_sampler_motion_and_floating_scope_across_themes() {
    for theme in ThemeKind::ALL {
        let mut h = fixture(theme);
        let initial = inspect(&h);
        let history = h.state().session.active().unwrap().history.past_len();
        let graph = rect(&h, "scopeRect");
        for fraction in [0.05, 0.18, 0.5, 0.82, 0.95] {
            let start = pos2(graph.left() + graph.width() * fraction, graph.center().y);
            drag(&mut h, start, start + vec2(18.0, 0.0));
        }
        let after = inspect(&h);
        for zone in ["blacks", "shadows", "exposure", "highlights", "whites"] {
            assert!(after["params"][zone].as_f64().unwrap() > 0.0, "{theme:?}/{zone}");
        }
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
        let ctx = h.ctx.clone();
        menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"samplerTool":true,"vectorscope":true,"shadows":true,"highlights":true}}})).unwrap();
        h.run_steps(3);
        let revision = inspect(&h)["previewRevision"].clone();
        let scope_revision = inspect(&h)["scopeRevision"].clone();
        let preview = rect(&h, "previewRect");
        let a = preview.center();
        let b = a + vec2(70.0, 30.0);
        press(&mut h, a, true, Modifiers::NONE);
        press(&mut h, a, false, Modifiers::NONE);
        assert_eq!(inspect(&h)["scope"]["samplers"].as_array().unwrap().len(), 1, "{theme:?}: sample click");
        drag(&mut h, a, b);
        assert!(inspect(&h)["scope"]["samplers"][0][0].as_f64().unwrap() > 0.5);
        h.event(Event::PointerMoved(preview.center()));
        h.run_steps(3);
        assert!(!inspect(&h)["pointerReadout"].is_null());
        assert_eq!(inspect(&h)["previewRevision"], revision);
        assert_eq!(inspect(&h)["scopeRevision"], scope_revision);
        press(&mut h, b, true, Modifiers { alt: true, ..Modifiers::NONE });
        press(&mut h, b, false, Modifiers { alt: true, ..Modifiers::NONE });
        assert!(inspect(&h)["scope"]["samplers"].as_array().unwrap().is_empty());
        menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"floating":true,"floatingRect":[50,70,400,400]}}})).unwrap();
        h.run_steps(4);
        assert_eq!(inspect(&h)["scope"]["floating"], true);
        h.set_size(vec2(960.0, 600.0));
        h.set_pixels_per_point(2.0);
        h.run_steps(3);
        assert_eq!(inspect(&h)["previewRevision"], revision);
        assert_eq!(inspect(&h)["scopeRevision"], scope_revision);
        menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"floating":false}}})).unwrap();
        h.run_steps(3);
        assert_eq!(inspect(&h)["scope"]["floating"], false);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
        assert_eq!(initial["histogram"]["shadows"][2], 64 * 48);
        assert_eq!(initial["histogram"]["highlights"][1], 64 * 48);
    }
}

#[test]
fn real_shortcuts_triangles_menu_and_scope_preferences_roundtrip() {
    let mut h = fixture(ThemeKind::Pro);
    let revision = inspect(&h)["previewRevision"].clone();
    for (key, field) in [(egui::Key::U, "shadows"), (egui::Key::O, "highlights"), (egui::Key::S, "samplerTool")] {
        h.key_press(key);
        h.run_steps(2);
        assert_eq!(inspect(&h)["scope"][field], true, "{field} shortcut");
        h.key_press(key);
        h.run_steps(2);
        assert_eq!(inspect(&h)["scope"][field], false);
    }
    let graph = rect(&h, "scopeRect");
    for (p, field) in [(graph.min + vec2(9.0, 9.0), "shadows"), (pos2(graph.right() - 9.0, graph.top() + 9.0), "highlights")] {
        press(&mut h, p, true, Modifiers::NONE);
        press(&mut h, p, false, Modifiers::NONE);
        assert_eq!(inspect(&h)["scope"][field], true, "{field} triangle");
    }
    h.event(Event::PointerMoved(graph.center()));
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: graph.center(), button: PointerButton::Secondary, pressed, modifiers: Modifiers::NONE });
    }
    h.run_steps(3);
    let label = photocraft_ui_egui::i18n::tr(photocraft_ui_egui::i18n::current(), "Show Lab Color Readouts");
    h.get_by_label(label).click();
    h.run_steps(3);
    assert_eq!(inspect(&h)["scope"]["lab"], true);
    assert_eq!(inspect(&h)["previewRevision"], revision);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"floating":true,"floatingRect":[60,90,400,420],"vectorscope":true}}})).unwrap();
    h.run_steps(4);
    let saved = h.state().session.prefs().dialogs.get("filter.cameraRaw.scope").unwrap().clone();
    assert_eq!(saved["floating"], true);
    assert_eq!(saved["samplers"], json!([]));
    assert_eq!(saved["vectorscope"], false);
    let prefs = serde_json::to_string(h.state().session.prefs()).unwrap();
    let mut fresh = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    fresh.session.load_prefs_json(&prefs).unwrap();
    fresh.session.prefs.edit(|p| p.dialogs.get_mut("filter.cameraRaw.scope").unwrap()["vectorscope"] = json!(true));
    fresh.run("file.new", json!({"width":16,"height":16})).unwrap();
    fresh.run("layer.new.layer", json!({})).unwrap();
    menu(&mut fresh, &egui::Context::default(), json!({})).unwrap();
    assert!(!fresh.ui.camera_raw_scope.vectorscope, "vectorscope stays opt-in when reopening Camera Raw");
    assert!(fresh.ui.camera_raw_scope.floating);
    assert!(fresh.ui.camera_raw_scope.lab);
    assert_eq!(fresh.ui.camera_raw_scope.floating_rect, h.state().ui.camera_raw_scope.floating_rect);
}

#[test]
fn selected_region_scope_and_preview_share_the_engine_selection_boundary() {
    let ctx = egui::Context::default();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width":16,"height":8})).unwrap();
    app.run("layer.new.layer", json!({})).unwrap();
    app.run("edit.fill", json!({"color":"#808080"})).unwrap();
    app.run("select.rect", json!({"x":0,"y":0,"width":8,"height":8})).unwrap();
    let original = app.session.active().unwrap().doc.clone();
    menu(&mut app, &ctx, json!({})).unwrap();
    let full = menu(&mut app, &ctx, json!({"ui":{"set":{"exposure":1.0},"scope":{"vectorscope":true}}})).unwrap();
    assert_eq!(full["vectorscope"]["samples"], 128);
    let selected = menu(&mut app, &ctx, json!({"ui":{"scope":{"selectedRegion":true}}})).unwrap();
    assert_eq!(selected["vectorscope"]["samples"], 64);
    assert_eq!(selected["previewRevision"], full["previewRevision"]);
    assert_ne!(selected["scopeRevision"], full["scopeRevision"]);
    let before = menu(&mut app, &ctx, json!({"ui":{"before":true}})).unwrap();
    assert_eq!(before["vectorscope"]["before"], true);
    let after = menu(&mut app, &ctx, json!({"ui":{"before":false}})).unwrap();
    assert!(std::sync::Arc::ptr_eq(&original, &app.session.active().unwrap().doc));
    // Atomic validation: a bad property must not apply the valid tone delta or probe.
    assert!(menu(&mut app, &ctx, json!({"ui":{"scope":{"tone":{"zone":"exposure","delta":1},"sample":[0.5,0.5],"zzz":true}}})).is_err());
    let unchanged = control::inspect(&app, &ctx)["cameraRaw"].clone();
    assert_eq!(unchanged["params"], after["params"]);
    assert_eq!(unchanged["scope"], after["scope"]);
    assert_eq!(unchanged["pointerReadout"], after["pointerReadout"]);
    menu(&mut app, &ctx, json!({"ui":{"commit":true}})).unwrap();
    let committed = menu(&mut app, &ctx, json!({})).unwrap();
    assert_eq!(committed["histogram"], after["histogram"]);
}

#[test]
fn floating_window_is_above_the_dialog_and_can_be_dragged_back_to_the_dock() {
    let mut h = fixture(ThemeKind::Pro);
    // No heading, grab control or undock/sample actions in the context menu.
    assert!(h.query_by_label("Histogram").is_none());
    let graph = rect(&h, "scopeRect");
    for pressed in [true, false] {
        h.event(Event::PointerMoved(graph.center()));
        h.event(Event::PointerButton { pos: graph.center(), button: PointerButton::Secondary, pressed, modifiers: Modifiers::NONE });
    }
    h.run_steps(3);
    for source in ["Undock Scope", "Dock Scope", "Color Sampler (S)", "Clear Samplers"] {
        let label = photocraft_ui_egui::i18n::tr(photocraft_ui_egui::i18n::current(), source);
        assert!(h.query_by_label(label).is_none(), "removed action must not appear: {source}");
    }
    assert!(!h.state().ui.camera_raw_scope.vectorscope);
    let label = photocraft_ui_egui::i18n::tr(photocraft_ui_egui::i18n::current(), "Show Vectorscope");
    h.get_by_label(label).click();
    h.run_steps(3);
    assert!(h.state().ui.camera_raw_scope.vectorscope, "context menu still enables the optional vectorscope");
    press(&mut h, pos2(200.0, 400.0), true, Modifiers::NONE);
    press(&mut h, pos2(200.0, 400.0), false, Modifiers::NONE);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"floating":true,"floatingRect":[60,90,400,420]}}})).unwrap();
    h.run_steps(5);
    let r = inspect(&h)["scope"]["floatingRect"].clone();
    let title = pos2(r[0].as_f64().unwrap() as f32 + 70.0, r[1].as_f64().unwrap() as f32 + 12.0);
    let revision = inspect(&h)["previewRevision"].clone();
    drag(&mut h, title, title + vec2(120.0, 50.0));
    let moved = inspect(&h)["scope"]["floatingRect"].clone();
    assert!(moved[0].as_f64().unwrap() > r[0].as_f64().unwrap() + 80.0, "window must receive title drag above the full-screen dialog");
    assert_eq!(inspect(&h)["scope"]["floating"], true);
    // Resize the real floating window; geometry-only work keeps the image cache.
    let corner = pos2(moved[2].as_f64().unwrap() as f32 - 1.0, moved[3].as_f64().unwrap() as f32 - 1.0);
    drag(&mut h, corner, corner + vec2(50.0, 40.0));
    let resized = inspect(&h)["scope"]["floatingRect"].clone();
    assert!(resized[2].as_f64().unwrap() > moved[2].as_f64().unwrap() + 20.0);
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"floatingRect":[180,100,530,440]}}})).unwrap();
    h.run_steps(3);
    let moved = inspect(&h)["scope"]["floatingRect"].clone();
    assert!((moved[0].as_f64().unwrap() - 180.0).abs() < 2.0, "controller must reposition an already open panel: {moved}");
    assert!((moved[2].as_f64().unwrap() - 530.0).abs() < 4.0, "controller must resize an already open panel");
    let title = pos2(moved[0].as_f64().unwrap() as f32 + 70.0, moved[1].as_f64().unwrap() as f32 + 12.0);
    drag(&mut h, title, pos2(1000.0, 130.0));
    assert_eq!(inspect(&h)["scope"]["floating"], false, "release over dock");
    assert_eq!(inspect(&h)["previewRevision"], revision);
}

#[test]
fn alt_tone_diagnostics_and_double_click_reset_follow_the_same_proxy_revision() {
    let mut h = fixture(ThemeKind::Pro);
    let history = h.state().session.active().unwrap().history.past_len();
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"set":{"exposure":0.5}}})).unwrap();
    h.run_steps(3);
    assert!(inspect(&h)["overlayRevision"].is_null());
    let graph = rect(&h, "scopeRect");
    let p = graph.center();
    let alt = Modifiers { alt: true, ..Modifiers::NONE };
    press(&mut h, p, true, alt);
    h.event(Event::PointerMoved(p + vec2(15.0, 0.0)));
    h.run_steps(4);
    assert_eq!(inspect(&h)["overlayRevision"], inspect(&h)["previewRevision"], "Alt drag generates clipping diagnostics for the current image");
    press(&mut h, p + vec2(15.0, 0.0), false, Modifiers::NONE);
    // Reset through actual double-clicks, after waiting beyond the preceding gesture.
    h.run_steps(30);
    for _ in 0..2 {
        press(&mut h, p, true, Modifiers::NONE);
        press(&mut h, p, false, Modifiers::NONE);
    }
    assert_eq!(inspect(&h)["params"]["exposure"], 0.0);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
}

#[test]
fn both_clipping_warnings_leave_unclipped_preview_pixels_visible_on_gpu() {
    let _gpu = gpu_lock();
    let mut h = Harness::builder().with_step_dt(1.0 / 60.0).with_size(vec2(800.0, 600.0)).wgpu().build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, ThemeKind::Pro);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.set_theme(&cc.egui_ctx, ThemeKind::Pro);
        app.run("file.new", json!({"width":16,"height":16})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color":"#808080"})).unwrap();
        camera_raw_ui::open(&mut app, &cc.egui_ctx).unwrap();
        app
    });
    h.run_steps(5);
    let center = rect(&h, "previewRect").center();
    let pixel = (center.x as u32, center.y as u32);
    let original = h.render().unwrap().get_pixel(pixel.0, pixel.1).0;
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"shadows":true,"highlights":true}}})).unwrap();
    h.run_steps(4);
    let with_warnings = h.render().unwrap().get_pixel(pixel.0, pixel.1).0;
    assert_eq!(with_warnings, original, "both warnings must preserve an unclipped image pixel");
}

#[test]
fn curve_drag_moves_the_existing_point_without_creating_another() {
    let mut h = fixture(ThemeKind::Pro);
    let graph = open_curve(&mut h);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"set":{"pointCurve":[[0,0],[128,128],[255,255]]}}})).unwrap();
    h.run_steps(3);
    let a = pos2(graph.left() + 128.0 / 255.0 * 260.0, graph.bottom() - 128.0 / 255.0 * 260.0);
    let b = a + vec2(40.0, -32.0);
    drag(&mut h, a, b);
    let points = inspect(&h)["params"]["pointCurve"].clone();
    assert_eq!(points.as_array().unwrap().len(), 3, "drag must keep the original point count: {points}");
    assert!(points[1][0].as_f64().unwrap() > 155.0, "existing point must move horizontally: {points}");
    assert!(points[1][1].as_f64().unwrap() > 150.0, "existing point must move vertically: {points}");
}

fn open_curve(h: &mut Harness<'_, PhotocraftApp>) -> Rect {
    let lang = photocraft_ui_egui::i18n::current();
    h.get_by_label(photocraft_ui_egui::i18n::tr_ctx(lang, "cameraRaw", "Light")).click();
    h.run_steps(16);
    h.get_by_label(photocraft_ui_egui::i18n::tr(lang, "Curve")).click();
    h.run_steps(16);
    rect(h, "curveRect")
}
fn curve_at(graph: Rect, point: [f32; 2]) -> Pos2 {
    pos2(graph.left() + point[0] / 255.0 * graph.width(), graph.bottom() - point[1] / 255.0 * graph.height())
}
fn set_curve(h: &mut Harness<'_, PhotocraftApp>, points: Value) {
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"set":{"pointCurve":points}}})).unwrap();
    h.run_steps(3);
}
fn curve_points(h: &Harness<'_, PhotocraftApp>) -> Value {
    inspect(h)["params"]["pointCurve"].clone()
}

#[test]
fn controller_click_delivered_in_one_raw_frame_selects_an_existing_curve_point() {
    let mut h = fixture(ThemeKind::Pro);
    let graph = open_curve(&mut h);
    set_curve(&mut h, json!([[0.0, 0.0], [128.0, 128.0], [255.0, 255.0]]));
    let at = curve_at(graph, [128.0, 128.0]);
    let ctx = h.ctx.clone();
    let (request, _) = control::ControlRequest::new("ui.click", json!({"x":at.x,"y":at.y}));
    assert!(matches!(control::handle(h.state_mut(), &ctx, &request), control::Outcome::AfterInput));
    // Harness::event splits a press/release pair across frames. Feed the controller's actual
    // raw-input batch to egui to cover the native rapid-click boundary instead.
    let input = egui::RawInput { screen_rect: Some(ctx.content_rect()), events: h.state_mut().take_synthetic_step(), ..Default::default() };
    let mut output = ctx.run_ui(input, |ui| camera_raw_ui::show(h.state_mut(), ui.ctx()));
    // This CPU interaction assertion intentionally does not present a frame.
    output.textures_delta.clear();
    assert_eq!(inspect(&h)["curveState"]["selected"], 1);
    assert_eq!(curve_points(&h).as_array().unwrap().len(), 3);
}

#[test]
fn curve_capture_handles_fast_and_slow_motion_without_jumps_in_every_theme() {
    for theme in ThemeKind::ALL {
        let mut h = fixture(theme);
        let graph = open_curve(&mut h);
        let history = h.state().session.active().unwrap().history.past_len();
        for steps in [1, 16] {
            set_curve(&mut h, json!([[0.0, 0.0], [128.0, 128.0], [255.0, 255.0]]));
            let a = curve_at(graph, [128.0, 128.0]) + vec2(4.0, 3.0);
            let b = a + vec2(40.0, -32.0);
            let revision = inspect(&h)["previewRevision"].clone();
            press(&mut h, a, true, Modifiers::NONE);
            assert_eq!(curve_points(&h), json!([[0.0, 0.0], [128.0, 128.0], [255.0, 255.0]]), "press must preserve the grab offset");
            assert_eq!(inspect(&h)["previewRevision"], revision, "selection alone does not rebuild the image");
            for i in 1..=steps {
                h.event(Event::PointerMoved(a + (b - a) * i as f32 / steps as f32));
                h.run_steps(1);
                assert_eq!(curve_points(&h).as_array().unwrap().len(), 3);
                assert_eq!(inspect(&h)["curveState"]["drag"]["index"], 1, "capture survives every frame");
            }
            press(&mut h, b, false, Modifiers::NONE);
            let points = curve_points(&h);
            assert!((points[1][0].as_f64().unwrap() - (128.0 + 40.0 / graph.width() as f64 * 255.0)).abs() < 1.0);
            assert!((points[1][1].as_f64().unwrap() - (128.0 + 32.0 / graph.height() as f64 * 255.0)).abs() < 1.0);
            assert!(inspect(&h)["curveState"]["drag"].is_null());
        }
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
        let ctx = h.ctx.clone();
        menu(h.state_mut(), &ctx, json!({"ui":{"commit":true}})).unwrap();
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
        h.state_mut().run("edit.undo", json!({})).unwrap();
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    }
}

#[test]
fn curve_deletion_reentry_keyboard_limits_and_controller_replacement_share_capture() {
    let mut h = fixture(ThemeKind::Pro);
    let graph = open_curve(&mut h);
    let original = h.state().session.active().unwrap().doc.clone();
    set_curve(&mut h, json!([[0.0, 0.0], [128.0, 128.0], [255.0, 255.0]]));
    let a = curve_at(graph, [128.0, 128.0]);
    let outside = graph.right_center() + vec2(30.0, 0.0);
    press(&mut h, a, true, Modifiers::NONE);
    h.event(Event::PointerMoved(outside));
    h.run_steps(2);
    assert_eq!(curve_points(&h).as_array().unwrap().len(), 2);
    let b = curve_at(graph, [150.0, 160.0]);
    h.event(Event::PointerMoved(b));
    h.run_steps(2);
    assert_eq!(curve_points(&h), json!([[0.0, 0.0], [150.0, 160.0], [255.0, 255.0]]));
    press(&mut h, b, false, Modifiers::NONE);
    h.key_press(egui::Key::ArrowUp);
    h.run_steps(3);
    assert_eq!(curve_points(&h)[1][1].as_f64(), Some(161.0));
    h.key_press(egui::Key::Delete);
    h.run_steps(3);
    assert_eq!(curve_points(&h).as_array().unwrap().len(), 2);
    // Same-frame native/controller clicks also insert; right-click deletes independently.
    for button in [PointerButton::Primary, PointerButton::Secondary] {
        h.event(Event::PointerMoved(a));
        h.event(Event::PointerButton { pos: a, button, pressed: true, modifiers: Modifiers::NONE });
        h.event(Event::PointerButton { pos: a, button, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(3);
        assert_eq!(curve_points(&h).as_array().unwrap().len(), if button == PointerButton::Primary { 3 } else { 2 });
    }
    set_curve(&mut h, json!([[0.0, 0.0], [128.0, 128.0], [255.0, 255.0]]));
    press(&mut h, a, true, Modifiers::NONE);
    // A controller replacing points must cancel the old capture even with the same length.
    set_curve(&mut h, json!([[0.0, 0.0], [80.0, 170.0], [255.0, 255.0]]));
    assert!(inspect(&h)["curveState"]["drag"].is_null());
    press(&mut h, b, false, Modifiers::NONE);
    assert_eq!(curve_points(&h), json!([[0.0, 0.0], [80.0, 170.0], [255.0, 255.0]]));
    set_curve(&mut h, json!([[0.0, 0.0], [100.0, 120.0], [200.0, 180.0], [255.0, 255.0]]));
    drag(&mut h, curve_at(graph, [100.0, 120.0]), curve_at(graph, [240.0, 220.0]));
    assert_eq!(curve_points(&h)[1][0].as_f64(), Some(199.0), "point cannot pass its neighbour");
    drag(&mut h, graph.left_bottom(), graph.left_top() + vec2(-30.0, -30.0));
    assert_eq!(curve_points(&h).as_array().unwrap().len(), 4, "endpoint cannot be removed by dragging off");
    let full: Vec<_> = (0..16).map(|i| [i * 17, i * 17]).collect();
    set_curve(&mut h, json!(full));
    let p = curve_at(graph, [100.0, 180.0]);
    press(&mut h, p, true, Modifiers::NONE);
    press(&mut h, p, false, Modifiers::NONE);
    assert_eq!(curve_points(&h).as_array().unwrap().len(), 16);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"cancel":true}})).unwrap();
    assert!(std::sync::Arc::ptr_eq(&original, &h.state().session.active().unwrap().doc));
}

#[test]
fn curve_gpu_paints_the_moved_handle_in_every_theme() {
    let _gpu = gpu_lock();
    for theme in ThemeKind::ALL {
        let mut h = fixture_render(theme, true);
        let graph = open_curve(&mut h);
        set_curve(&mut h, json!([[0, 0], [128, 128], [255, 255]]));
        let before = h.render().unwrap();
        let a = curve_at(graph, [128.0, 128.0]);
        drag(&mut h, a, a + vec2(40.0, -32.0));
        assert_eq!(curve_points(&h).as_array().unwrap().len(), 3);
        let points = curve_points(&h);
        let b = curve_at(graph, [points[1][0].as_f64().unwrap() as f32, points[1][1].as_f64().unwrap() as f32]);
        let after = h.render().unwrap();
        assert_ne!(
            before.get_pixel(b.x.round() as u32, b.y.round() as u32),
            after.get_pixel(b.x.round() as u32, b.y.round() as u32),
            "moved handle must appear in the native GPU frame"
        );
        if let Ok(dir) = std::env::var("PHOTOCRAFT_CURVE_SNAPSHOTS") {
            std::fs::create_dir_all(&dir).unwrap();
            after.save(std::path::Path::new(&dir).join(format!("{theme:?}.png"))).unwrap();
        }
    }
}

#[test]
fn pixel_hover_tracks_displayed_rgb_before_after_and_keeps_analysis_cached() {
    let mut h = fixture(ThemeKind::Pro);
    let ctx = h.ctx.clone();
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"vectorscope":true,"lab":true}}})).unwrap();
    h.run_steps(3);
    let preview = rect(&h, "previewRect");
    let initial = inspect(&h);
    assert!(initial["hoverSample"].is_null());
    h.hover_at(preview.center());
    h.run_steps(3);
    let sample = inspect(&h);
    assert!(!sample["hoverSample"].is_null(), "ordinary hover works with sampler tool disabled");
    assert_eq!(sample["scope"]["samplerTool"], false);
    let rgb = &sample["hoverSample"]["rgb"];
    assert!((rgb[0].as_f64().unwrap() - 128.0 / 255.0).abs() < 0.0001);
    assert_eq!(rgb[1].as_f64(), Some(1.0));
    assert_eq!(rgb[2].as_f64(), Some(0.0));
    assert_eq!(sample["pointerReadout"]["space"], "lab", "badge RGB stays independent of Lab probe preferences");
    assert!((sample["hoverSample"]["hueSaturation"][0].as_f64().unwrap() - 0.25).abs() < 0.003);
    assert_eq!(sample["hoverSample"]["hueSaturation"][1].as_f64(), Some(1.0));
    for n in 1..=8 {
        h.hover_at(preview.center() + vec2(n as f32, n as f32));
        h.run_steps(1);
    }
    let moved = inspect(&h);
    assert_eq!(moved["previewRevision"], initial["previewRevision"]);
    assert_eq!(moved["scopeRevision"], initial["scopeRevision"]);
    assert_eq!(moved["histogram"], initial["histogram"]);
    assert_eq!(moved["vectorscope"], initial["vectorscope"]);
    let history = h.state().session.active().unwrap().history.past_len();
    menu(h.state_mut(), &ctx, json!({"ui":{"set":{"exposure":0.5}}})).unwrap();
    h.run_steps(3);
    assert_ne!(inspect(&h)["hoverSample"]["rgb"], sample["hoverSample"]["rgb"]);
    menu(h.state_mut(), &ctx, json!({"ui":{"before":true}})).unwrap();
    h.run_steps(3);
    assert_eq!(inspect(&h)["hoverSample"]["rgb"], sample["hoverSample"]["rgb"]);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    h.hover_at(pos2(500.0, 20.0));
    h.run_steps(3);
    assert!(inspect(&h)["hoverSample"].is_null(), "leaving the image clears overlays instead of retaining a stale pixel");
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"vectorscope":false}}})).unwrap();
    h.run_steps(3);
    assert!(inspect(&h)["vectorscopeRect"].is_null());
}

#[test]
fn pixel_hover_gpu_badge_single_band_and_crosshair_appear_and_clear_in_all_themes() {
    let _gpu = gpu_lock();
    for theme in ThemeKind::ALL {
        let mut h = fixture_render(theme, true);
        let ctx = h.ctx.clone();
        menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"vectorscope":true}}})).unwrap();
        h.run_steps(4);
        let histogram = rect(&h, "scopeRect");
        let scope = rect(&h, "vectorscopeRect");
        let image = rect(&h, "previewRect");
        let no_hover = h.render().unwrap();
        h.hover_at(image.center());
        h.run_steps(3);
        let hovered = h.render().unwrap();
        // RGB text/badge occupies the top centre; one tone marker crosses the histogram.
        let area_changed = |r: Rect| {
            (r.top().ceil() as u32..r.bottom().floor() as u32)
                .any(|y| (r.left().ceil() as u32..r.right().floor() as u32).any(|x| no_hover.get_pixel(x, y) != hovered.get_pixel(x, y)))
        };
        let badge = Rect::from_center_size(pos2(histogram.center().x, histogram.top() + 13.0), vec2(140.0, 18.0));
        let bands = Rect::from_min_max(pos2(histogram.left() + 4.0, histogram.center().y), pos2(histogram.right() - 4.0, histogram.bottom() - 4.0));
        assert!(area_changed(badge), "{theme:?}: RGB badge");
        assert!(area_changed(bands), "{theme:?}: translucent tone band");
        let y = histogram.center().y as u32;
        let changed: Vec<_> =
            (histogram.left().ceil() as u32..histogram.right().floor() as u32).filter(|&x| no_hover.get_pixel(x, y) != hovered.get_pixel(x, y)).collect();
        let first = *changed.first().expect("visible tone marker");
        let last = *changed.last().unwrap();
        assert_eq!(last - first, 1, "{theme:?}: exactly one two-pixel band, without a wide or blurred halo: {changed:?}");
        assert_eq!(changed.len(), (last - first + 1) as usize, "{theme:?}: one connected band");
        let rgb = inspect(&h)["hoverSample"]["rgb"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
        let tone = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let plot = histogram.shrink(4.0);
        let expected = plot.left() as f64 + tone * plot.width() as f64;
        assert!((f64::from(first + last) * 0.5 - expected).abs() <= 1.0, "{theme:?}: marker follows the pixel tone");
        assert!(area_changed(scope), "{theme:?}: vectorscope target");
        let bright_target = (scope.top().ceil() as u32..scope.bottom().floor() as u32).any(|y| {
            (scope.left().ceil() as u32..scope.right().floor() as u32).any(|x| {
                let before = no_hover.get_pixel(x, y).0;
                let after = hovered.get_pixel(x, y).0;
                let luminance = |p: [u8; 4]| p.iter().take(3).map(|c| u16::from(*c)).sum::<u16>();
                luminance(after) > 600 && luminance(before) < 450
            })
        });
        assert!(bright_target, "{theme:?}: target must contrast against the dark scope even in light themes");
        if let Ok(dir) = std::env::var("PHOTOCRAFT_HOVER_SNAPSHOTS") {
            std::fs::create_dir_all(&dir).unwrap();
            hovered.save(std::path::Path::new(&dir).join(format!("{theme:?}.png"))).unwrap();
        }
        h.hover_at(pos2(500.0, 20.0));
        h.run_steps(3);
        let cleared = h.render().unwrap();
        for r in [histogram, scope] {
            for y in r.top().ceil() as u32..r.bottom().floor() as u32 {
                for x in r.left().ceil() as u32..r.right().floor() as u32 {
                    assert_eq!(no_hover.get_pixel(x, y), cleared.get_pixel(x, y), "overlay must clear: {theme:?}");
                }
            }
        }
    }
}

#[test]
fn sampler_list_replacement_and_operations_cannot_silently_cancel_each_other() {
    let mut h = fixture(ThemeKind::Pro);
    let ctx = h.ctx.clone();
    let combined = json!({"ui":{"scope":{"samplers":[[0.1,0.1]],"addSampler":[0.5,0.5]}}});
    assert!(menu(h.state_mut(), &ctx, combined).is_err(), "replacing and editing the list in one request is ambiguous");
    menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"samplers":[[0.1,0.1],[0.2,0.2]]}}})).unwrap();
    // Fixed order: clear, then remove, then add.
    let r = menu(h.state_mut(), &ctx, json!({"ui":{"scope":{"clearSamplers":true,"addSampler":[0.5,0.5]}}})).unwrap();
    assert_eq!(r["scope"]["samplers"], json!([[0.5, 0.5]]));
}

/// Alt-drags the slider under the only visible `label` and returns the inspected dialog.
fn alt_drag_slider(h: &mut Harness<'_, PhotocraftApp>, label: &str) -> Value {
    // The label sits above its 18 pt slider track.
    let text = h.get_by_label(label).rect();
    let at = pos2(text.left() + 80.0, text.bottom() + 12.0);
    let alt = Modifiers { alt: true, ..Modifiers::NONE };
    press(h, at, true, alt);
    for n in 1..=3 {
        h.event(Event::PointerMoved(at + vec2(10.0 * n as f32, 0.0)));
        h.run_steps(2);
    }
    let during = inspect(h);
    press(h, at + vec2(30.0, 0.0), false, Modifiers::NONE);
    during
}

#[test]
fn alt_drag_diagnostics_follow_light_tone_sliders_not_same_named_ones() {
    let tr = |s| photocraft_ui_egui::i18n::tr_ctx(photocraft_ui_egui::i18n::current(), "cameraRaw", s);
    let mut h = fixture(ThemeKind::Pro);
    let during = alt_drag_slider(&mut h, tr("Highlights"));
    assert_ne!(during["params"]["highlights"], 0.0, "the drag hit the Light slider");
    assert_eq!(during["overlayRevision"], during["previewRevision"], "Light tone sliders drive Alt diagnostics");

    let mut h = fixture(ThemeKind::Pro);
    h.get_by_label(tr("Light")).click();
    h.run_steps(12);
    h.get_by_label(tr("Curve")).click();
    h.run_steps(12);
    // The Curve section's Highlights shares its label with the Light tone slider.
    let during = alt_drag_slider(&mut h, tr("Highlights"));
    assert_ne!(during["params"]["curveHighlights"], 0.0, "the drag hit the Curve slider");
    assert!(during["overlayRevision"].is_null(), "a same-named non-tone slider must not drive Alt diagnostics");
}
