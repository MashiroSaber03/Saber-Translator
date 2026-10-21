use super::*;
use egui::{Context, Modifiers, Pos2, RawInput, Rect};
use egui_kittest::{
    Harness,
    kittest::{NodeT, Queryable},
};
use photocraft_engine::Session;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(Session::new(), crate::Services::default());
    app.run("file.new", json!({"width":100,"height":80})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Brush;

    app.ui.views[0].zoom = 1.0;
    app.last_canvas_rect = Rect::from_min_size(Pos2::ZERO, vec2(500.0, 400.0));
    app
}
fn down(app: &mut PhotocraftApp, p: [f64; 2]) {
    crate::canvas::tool_event(app, ToolEvent::Down { x: p[0], y: p[1], pressure: 1.0 }, Modifiers::NONE);
}
fn move_to(app: &mut PhotocraftApp, p: [f64; 2], mods: Modifiers) {
    crate::canvas::tool_event(app, ToolEvent::Move { x: p[0], y: p[1], pressure: 1.0 }, mods);
}
fn up(app: &mut PhotocraftApp, p: [f64; 2]) {
    crate::canvas::tool_event(app, ToolEvent::Up { x: p[0], y: p[1] }, Modifiers::NONE);
}
fn key(app: &mut PhotocraftApp, key: Key) {
    let ctx = Context::default();
    let input = RawInput {
        focused: true,
        events: vec![egui::Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers: Modifiers::NONE }],
        ..Default::default()
    };
    ctx.run_ui(input, |ui| track(app, ui.ctx())).textures_delta.clear();
}
#[test]
fn transform_moves_rotates_snaps_and_commits_without_painting_or_history() {
    let mut app = app();
    app.run("paint.setSymmetry", json!({"mode":"dual"})).unwrap();
    let before = app.session.active().unwrap().doc.clone();
    let history = app.session.active().unwrap().history.past_len();
    crate::menus::invoke(&mut app, &Context::default(), "ui.symmetryTransform", json!({})).unwrap();
    down(&mut app, [50.0, 40.0]);
    move_to(&mut app, [60.0, 45.0], Modifiers::NONE);
    up(&mut app, [60.0, 45.0]);
    let p = app.ui.symmetry_transform.as_ref().unwrap().preview;
    assert_eq!(p.center, [60.0, 45.0]);
    assert_eq!(preset(&app).unwrap().center, [50.0, 40.0]);
    let handle = rotation_handle(p, [100.0, 80.0]);
    down(&mut app, handle);
    let angle = 112_f64.to_radians();
    move_to(&mut app, [60.0 + 28.0 * angle.cos(), 45.0 + 28.0 * angle.sin()], Modifiers { shift: true, ..Modifiers::NONE });
    up(&mut app, [0.0, 0.0]);
    assert!((app.ui.symmetry_transform.as_ref().unwrap().preview.rotation - 15.0).abs() < 1e-8);
    key(&mut app, Key::Enter);
    assert!(app.ui.symmetry_transform.is_none());
    assert_eq!(preset(&app).unwrap().center, [60.0, 45.0]);
    assert_eq!(app.session.active().unwrap().doc, before);
    assert_eq!(app.session.active().unwrap().history.past_len(), history);
    assert_eq!(app.session.journal.last().unwrap().0, "paint.setSymmetry");
}
#[test]
fn cancel_competing_edits_tool_document_and_focus_changes_drop_only_the_preview() {
    let mut app = app();
    app.run("paint.setSymmetry", json!({"mode":"vertical"})).unwrap();
    begin(&mut app).unwrap();
    down(&mut app, [50.0, 40.0]);
    move_to(&mut app, [10.0, 20.0], Modifiers::NONE);
    key(&mut app, Key::Escape);
    assert_eq!(preset(&app).unwrap().center, [50.0, 40.0]);
    assert!(app.ui.symmetry_transform.is_none());
    begin(&mut app).unwrap();
    app.run("document.inspect", json!({})).unwrap();
    assert!(app.ui.symmetry_transform.is_some());
    app.run("paint.setSymmetry", json!({"mode":"horizontal"})).unwrap();
    assert!(app.ui.symmetry_transform.is_none());
    for change in 0..4 {
        begin(&mut app).unwrap();
        match change {
            0 => app.ui.tool = Tool::Pencil,
            1 => {
                app.session.execute("file.new", json!({"width":10,"height":10})).unwrap();
            }
            2 => {
                app.session.execute("layer.new.layer", json!({})).unwrap();
            }
            _ => {}
        }
        let ctx = Context::default();
        ctx.run_ui(RawInput { focused: change != 3, ..Default::default() }, |ui| track(&mut app, ui.ctx())).textures_delta.clear();
        assert!(app.ui.symmetry_transform.is_none());
        if change == 1 {
            assert!(app.session.set_active(0));
        }
    }
    begin(&mut app).unwrap();
    let ctx = Context::default();
    let input = RawInput {
        focused: true,
        events: vec![egui::Event::Key { key: Key::V, physical_key: Some(Key::V), pressed: true, repeat: false, modifiers: Modifiers::NONE }],
        ..Default::default()
    };
    ctx.run_ui(input, |ui| {
        crate::shortcuts::handle(&mut app, ui.ctx());
        track(&mut app, ui.ctx());
    })
    .textures_delta
    .clear();
    assert_eq!(app.ui.tool, Tool::Move);
    assert!(app.ui.symmetry_transform.is_none());
}
#[test]
fn guides_never_consume_normal_paint_and_transform_background_clicks_never_paint() {
    let mut app = app();
    app.run("paint.setSymmetry", json!({"mode":"vertical"})).unwrap();
    let before = app.session.active().unwrap().history.past_len();
    down(&mut app, [20.0, 20.0]);
    up(&mut app, [20.0, 20.0]);
    assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
    begin(&mut app).unwrap();
    down(&mut app, [10.0, 10.0]);
    move_to(&mut app, [30.0, 30.0], Modifiers::NONE);
    up(&mut app, [30.0, 30.0]);
    assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
}
#[test]
fn menu_selects_all_modes_resets_positions_and_preserves_work_path_choice() {
    let mut app = app();
    app.session.edit_prefs(|p| p.interface.language = "en".into());
    app.run("path.set", json!({"name":"work","path":{"subpaths":[{"knots":[[20,0],[20,80]]}]}})).unwrap();
    let mut h = Harness::builder().with_size(vec2(500.0, 450.0)).build_ui_state(
        |ctx, app| {
            crate::i18n::with_language(crate::i18n::Lang::EN, || menu(app, ctx));
        },
        app,
    );
    for mode in SymmetryMode::ALL {
        h.get_by_label("Set painting symmetry options").click();
        h.run_steps(3);
        h.get_by_label(label(mode)).click();
        h.run_steps(3);
        assert_eq!(preset(h.state()).unwrap().mode, mode);
        assert_eq!(preset(h.state()).unwrap().center, [50.0, 40.0]);
        h.get_by_label("Set painting symmetry options").click();
        h.run_steps(3);
        assert_eq!(h.get_by_label(label(mode)).accesskit_node().toggled(), Some(egui::accesskit::Toggled::True));
        h.get_by_label("Transform Symmetry").click();
        h.run_steps(3);
        assert!(h.state().ui.symmetry_transform.is_some());
        h.state_mut().ui.symmetry_transform = None;
    }
    h.get_by_label("Set painting symmetry options").click();
    h.run_steps(3);
    h.get_by_label("Work Path").click();
    h.run_steps(3);
    assert_eq!(h.state().session.active().unwrap().symmetry.as_ref().unwrap().path_source(), Some("work"));
    assert!(!can_transform(h.state()));
    h.get_by_label("Set painting symmetry options").click();
    h.run_steps(3);
    h.get_by_label("Symmetry Off").click();
    h.run_steps(3);
    assert!(preset(h.state()).is_none());
}
#[test]
fn overlays_follow_document_view_transforms_in_every_theme() {
    for theme in crate::theme::ThemeKind::ALL {
        let mut app = app();
        app.run("paint.setSymmetry", json!({"mode":"dual"})).unwrap();
        let ctx = Context::default();
        crate::theme::apply(&ctx, theme);
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(500.0, 400.0)), zoom: 1.7, center: [50.0, 40.0], rotation: 37.0, flip: true, aspect: 1.0 };
        let mut out = ctx.run_ui(RawInput { screen_rect: Some(xf.rect), ..Default::default() }, |ui| {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("symmetry")));
            draw(&app, &painter, &xf);
        });
        let lines: Vec<_> =
            out.shapes.iter().filter_map(|s| if let egui::Shape::LineSegment { points, .. } = &s.shape { Some(*points) } else { None }).collect();
        assert_eq!(lines.len(), 2);
        let c = xf.to_screen(50.0, 40.0);
        for [a, b] in lines {
            assert!((egui::pos2((a.x + b.x) / 2.0, (a.y + b.y) / 2.0) - c).length() < 0.01);
        }
        out.textures_delta.clear();
    }
    let de = crate::i18n::Lang::from_code("de").unwrap();
    assert_eq!(crate::i18n::tr(de, "Dual Axis"), "Duale Achse");
    assert_eq!(crate::i18n::tr(de, "Transform Symmetry"), "Symmetrie transformieren");
}
/// #4495: while the symmetry is transformed the options bar has ✓ and ⊘ (Enter and Esc were the
/// only ways out), and the canvas shows the move, rotate or arrow cursor instead of the brush.
#[test]
fn transform_has_commit_and_cancel_buttons_and_its_own_cursors() {
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(Session::new(), crate::Services::default())
    });
    h.state_mut().run("file.new", json!({"width":400,"height":300})).unwrap();
    h.state_mut().ui.tool = Tool::Brush;
    h.state_mut().run("paint.setSymmetry", json!({"mode":"dual"})).unwrap();
    h.run_steps(4);
    let cursor = |h: &Harness<'_, PhotocraftApp>| h.output().platform_output.cursor_icon;
    let begin_transform = |h: &mut Harness<'_, PhotocraftApp>| {
        let ctx = h.ctx.clone();
        crate::menus::invoke(h.state_mut(), &ctx, "ui.symmetryTransform", json!({})).unwrap();
        h.run_steps(3);
    };
    let xf = ViewXform::active(h.state()).expect("a canvas");
    let off_handles = xf.to_screen(40.0, 260.0);
    // Painting: the brush outline replaces the pointer.
    h.hover_at(off_handles);
    h.run_steps(2);
    assert_ne!(cursor(&h), egui::CursorIcon::Default, "the brush cursor while painting");
    assert!(h.query_by_label("Commit transform").is_none(), "no ✓ outside the transform");

    begin_transform(&mut h);
    h.hover_at(off_handles);
    h.run_steps(2);
    assert_eq!(cursor(&h), egui::CursorIcon::Default, "an arrow, not the brush, away from the handles");
    let p = h.state().ui.symmetry_transform.as_ref().unwrap().preview;
    h.hover_at(xf.to_screen(p.center[0] as f32, p.center[1] as f32));
    h.run_steps(2);
    assert_eq!(cursor(&h), egui::CursorIcon::Move, "the centre moves");
    let r = rotation_handle(p, [400.0, 300.0]);
    h.hover_at(xf.to_screen(r[0] as f32, r[1] as f32));
    h.run_steps(2);
    assert_eq!(cursor(&h), egui::CursorIcon::Alias, "the handle rotates");

    // ⊘ drops the preview; ✓ applies it.
    h.state_mut().ui.symmetry_transform.as_mut().unwrap().preview = PresetSymmetry::new(p.mode, [120.0, 90.0], 30.0).unwrap();
    h.get_by_label("Cancel transform").click();
    h.run_steps(3);
    assert!(h.state().ui.symmetry_transform.is_none());
    assert_eq!(preset(h.state()).unwrap().center, [200.0, 150.0], "cancel keeps the original");
    begin_transform(&mut h);
    h.state_mut().ui.symmetry_transform.as_mut().unwrap().preview = PresetSymmetry::new(p.mode, [120.0, 90.0], 30.0).unwrap();
    h.get_by_label("Commit transform").click();
    h.run_steps(3);
    assert!(h.state().ui.symmetry_transform.is_none());
    let done = preset(h.state()).unwrap();
    assert_eq!((done.center, done.rotation), ([120.0, 90.0], 30.0), "commit applies the preview");
}
