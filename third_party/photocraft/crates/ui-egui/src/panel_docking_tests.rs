use super::*;
use egui::{Event, Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::{Harness, kittest::Queryable};

fn app() -> PhotocraftApp {
    PhotocraftApp::new(photocraft_engine::Session::new(), Default::default())
}

#[test]
fn panel_commands_group_split_reorder_float_and_restore_without_duplicate_ownership() {
    let mut app = app();
    app.run("window.panel.move", json!({"panel":"swatches", "anchor":"layers", "zone":"top"})).unwrap();
    app.run("window.panel.move", json!({"panel":"properties", "anchor":"swatches", "before":"swatches"})).unwrap();
    let before_float = app.ui.docking.clone().unwrap();
    app.run("window.panel.float", json!({"panel":"properties", "x":50,"y":70,"width":330,"height":410})).unwrap();
    assert!(app.ui.docking.as_ref().unwrap().floating.iter().any(|g| g.panels == ["properties"]));
    app.run("window.panel.dock", json!({"panel":"properties"})).unwrap();
    assert_eq!(app.ui.docking.as_ref().unwrap(), &before_float);
    app.run("window.panel.close", json!({"panel":"properties"})).unwrap();
    assert!(!app.ui.docking.as_ref().unwrap().contains(&"properties".into()));
    app.run("window.panel.activate", json!({"panel":"properties"})).unwrap();
    assert_eq!(app.ui.docking.as_ref().unwrap(), &before_float);
    let layout = app.ui.docking.as_ref().unwrap();
    assert!(valid(layout));
    let unique: std::collections::HashSet<_> = layout.panels().into_iter().collect();
    assert_eq!(unique.len(), layout.panels().len());
    let saved = serde_json::to_value(&app.ui).unwrap();
    let restored: crate::state::UiState = serde_json::from_value(saved).unwrap();
    assert_eq!(restored.docking, app.ui.docking);
    assert_eq!(restored.docking_hidden, app.ui.docking_hidden);
}

#[test]
fn invalid_panel_requests_are_atomic_and_do_not_enter_a_custom_workspace() {
    let mut app = app();
    for (id, params) in [
        ("window.panel.move", json!({"panel":"layers","anchor":"unknown"})),
        ("window.panel.move", json!({"panel":"layers","anchor":"properties","before":"unknown"})),
        ("window.panel.move", json!({"panel":"layers","anchor":"properties","zone":42})),
        ("window.panel.float", json!({"panel":"layers","width":-10})),
        ("window.panel.float", json!({"panel":"unknown"})),
        ("window.panel.float", json!({"panel":"layers","x":"invalid"})),
    ] {
        let old = app.ui.docking.clone();
        let hidden = app.ui.docking_hidden.clone();
        assert!(app.run(id, params).is_err(), "{id}");
        assert_eq!(app.ui.docking, old);
        assert_eq!(app.ui.docking_hidden, hidden);
    }
}

fn harness(app: PhotocraftApp, size: egui::Vec2) -> Harness<'static, PhotocraftApp> {
    let mut ready = false;
    let mut harness = Harness::builder().with_size(size).with_step_dt(1.0 / 60.0).build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if !ready {
                PhotocraftApp::setup_context(ui.ctx(), crate::theme::ThemeKind::ProMedium);
                ready = true;
                return;
            }
            crate::panels::right_dock(app, ui);
            egui::CentralPanel::default().show(ui, |_| {});
        },
        app,
    );
    harness.run_steps(4);
    harness
}

fn drag(h: &mut Harness<'static, PhotocraftApp>, from: Pos2, to: Pos2) {
    h.event(Event::PointerMoved(from));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    for step in 1..=8 {
        h.event(Event::PointerMoved(from + (to - from) * (step as f32 / 8.0)));
        h.run_steps(1);
    }
    h.event(Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

#[test]
fn ordinary_panel_tab_drag_enters_shared_docking_and_can_redock_into_another_tab_group() {
    let mut h = harness(app(), vec2(1280.0, 800.0));
    let source = h.query_all_by_label("Layers").find(|node| node.rect().height() <= 45.0).unwrap().rect().center();
    drag(&mut h, source, egui::pos2(160.0, 140.0));
    let layout = h.state().ui.docking.as_ref().expect("ordinary tab drag starts shared docking");
    assert!(layout.floating.iter().any(|g| g.panels.contains(&"layers".into())), "{layout:?}");
    let source = h.ctx.read_response(egui::Id::new("photocraft-panel-docking").with(("tab", &"layers".to_string()))).unwrap().rect.center();
    let destination = h.ctx.read_response(egui::Id::new("photocraft-panel-docking").with(("tab", &"properties".to_string()))).unwrap().rect.center();
    drag(&mut h, source, destination);
    let layout = h.state().ui.docking.as_ref().unwrap();
    assert!(!layout.floating.iter().any(|g| g.panels.contains(&"layers".into())));
    assert!(valid(layout));
}

#[test]
fn saved_workspace_restores_the_custom_tree_and_hidden_panel_placement() {
    let mut app = app();
    let ctx = egui::Context::default();
    app.run("window.panel.move", json!({"panel":"swatches","anchor":"layers","zone":"top"})).unwrap();
    app.run("window.panel.float", json!({"panel":"properties","x":50,"y":80})).unwrap();
    app.run("window.panel.close", json!({"panel":"swatches"})).unwrap();
    let layout = app.ui.docking.clone();
    let hidden = app.ui.docking_hidden.clone();
    crate::menus::invoke(&mut app, &ctx, "window.workspace.newWorkspace", json!({"name":"Custom docking"})).unwrap();
    crate::menus::invoke(&mut app, &ctx, "window.workspace.essentials", json!({})).unwrap();
    assert!(app.ui.docking.is_none());
    crate::menus::invoke(&mut app, &ctx, "window.workspace.select", json!({"name":"Custom docking"})).unwrap();
    assert_eq!(app.ui.docking, layout);
    assert_eq!(app.ui.docking_hidden, hidden);
}

#[test]
fn shared_layout_keeps_panel_identity_across_themes_and_respects_workspace_lock() {
    let mut app = app();
    app.run("window.panel.move", json!({"panel":"color","anchor":"layers"})).unwrap();
    let layout = app.ui.docking.clone();
    app.ui.theme = crate::theme::ThemeKind::Studio;
    assert_eq!(app.ui.docking, layout);
    app.session.prefs.edit(|prefs| prefs.workspace_locked = true);
    assert!(app.run("window.panel.float", json!({"panel":"color"})).is_err());
    assert_eq!(app.ui.docking, layout);
    assert!(app.run("window.panel.activate", json!({"panel":"color"})).is_ok());
}

/// Explicit offscreen evidence; this creates no native windows.
#[test]
fn capture_custom_panel_docking_visual_fixtures() {
    let Some(directory) = std::env::var_os("CRAFT_UI_DOCKING_FIXTURES").map(std::path::PathBuf::from) else { return };
    std::fs::create_dir_all(&directory).unwrap();
    for theme in crate::theme::ThemeKind::ALL {
        for width in [800.0, 1280.0] {
            for scale in [1.0, 1.5, 2.0] {
                let mut app = app();
                app.run("file.new", json!({})).unwrap();
                app.run("window.panel.move", json!({"panel":"swatches", "anchor":"layers", "zone":"top"})).unwrap();
                app.run("window.panel.float", json!({"panel":"properties","x":30,"y":90,"width":300,"height":440})).unwrap();
                let mut ready = false;
                let mut harness = Harness::builder().with_size(vec2(width, 800.0)).with_pixels_per_point(scale).wgpu().build_ui_state(
                    move |ui, app: &mut PhotocraftApp| {
                        if !ready {
                            PhotocraftApp::setup_context(ui.ctx(), theme);
                            app.ui.theme = theme;
                            ready = true;
                            return;
                        }
                        crate::panels::right_dock(app, ui);
                        egui::CentralPanel::default().show(ui, |ui| {
                            ui.label("Custom panel workspace");
                        });
                    },
                    app,
                );
                harness.input_mut().max_texture_side = Some(8192);
                harness.run_steps(5);
                assert!(valid(harness.state().ui.docking.as_ref().unwrap()));
                harness.render().unwrap().save(directory.join(format!("photocraft-shared-docking-{}-{width}-{scale}x.png", theme.id()))).unwrap();
            }
        }
    }
}

#[test]
fn floating_group_titles_are_opaque_in_studio_themes() {
    // Studio's tab strip is transparent over the dock's card; a floating group must not show the canvas (#4509).
    let mut app = app();
    app.run("window.panel.float", json!({"panel":"swatches","x":40,"y":60,"width":300,"height":300})).unwrap();
    let mut ready = false;
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).wgpu().build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if !ready {
                PhotocraftApp::setup_context(ui.ctx(), crate::theme::ThemeKind::Studio);
                app.ui.theme = crate::theme::ThemeKind::Studio;
                ready = true;
                return;
            }
            crate::panels::right_dock(app, ui);
            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(255, 0, 255))).show(ui, |_| {});
        },
        app,
    );
    h.run_steps(5);
    let area = h.ctx.memory(|m| m.area_rect(egui::Id::new("photocraft-panel-docking").with(("floating", &"swatches".to_string())))).unwrap();
    let image = h.render().unwrap();
    let ppp = image.width() as f32 / 1280.0;
    let strip = egui::Rect::from_min_size(area.min, vec2(area.width(), 40.0)).shrink(2.0);
    let magenta = (strip.left() as u32..strip.right() as u32)
        .flat_map(|x| (strip.top() as u32..strip.bottom() as u32).map(move |y| (x, y)))
        .filter(|&(x, y)| image.get_pixel((x as f32 * ppp) as u32, (y as f32 * ppp) as u32).0[..3] == [255, 0, 255])
        .count();
    assert_eq!(magenta, 0, "canvas shows through the floating group's title strip {strip:?}");
}

#[test]
fn escape_cancels_the_first_drag_without_replacing_the_legacy_workspace() {
    let mut h = harness(app(), vec2(1280.0, 800.0));
    let source = h.query_all_by_label("Layers").find(|node| node.rect().height() <= 45.0).unwrap().rect().center();
    h.event(Event::PointerMoved(source));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(Event::PointerMoved(source - vec2(90.0, 0.0)));
    h.run_steps(2);
    assert!(h.state().ui.docking.is_some());
    h.event(Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
    assert!(h.state().ui.docking.is_none());
    h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert!(h.state().ui.docking.is_none());
}

#[test]
fn serialized_layout_actions_cover_geometry_reorder_accordion_and_atomic_rejection() {
    let mut app = app();
    app.run("window.panel.move", json!({"panel":"swatches","anchor":"layers","zone":"top"})).unwrap();
    app.run("window.panel.layout", json!({"action":{"ResizeSplit":{"path":[],"size":{"Ratio":0.35}}}})).unwrap();
    assert!(
        matches!(app.ui.docking.as_ref().unwrap().root, Some(Node::Split { size: craft_ui::layout::SplitSize::Ratio(value), .. }) if (value - 0.35).abs() < 0.001)
    );
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    app.run("window.panel.layout", json!({"action":{"MoveFloating":{"panel":"properties","rect":[44,66,360,450]}}})).unwrap();
    assert_eq!(
        app.ui.docking.as_ref().unwrap().floating.iter().find(|group| group.panels.iter().any(|p| p == "properties")).unwrap().rect,
        [44.0, 66.0, 360.0, 450.0]
    );
    app.run("window.panel.layout", json!({"action":{"Move":{"panel":"properties","anchor":"swatches","placement":{"Tab":{"before":"swatches"}}}}})).unwrap();
    assert_eq!(group_members_for_test(app.ui.docking.as_ref().unwrap(), "swatches"), ["properties", "swatches"]);
    let before = app.ui.docking.clone();
    let hidden = app.ui.docking_hidden.clone();
    for action in [
        json!({"MoveFloating":{"panel":"properties","rect":[0,0,-5,30]}}),
        json!({"ResizeSplit":{"path":[true,true,true],"size":{"Ratio":0.2}}}),
        json!({"Open":{"panel":"unknown","anchor":null}}),
        json!({"Move":{"panel":"properties","anchor":"swatches","placement":{"Tab":{"before":"unknown"}}}}),
    ] {
        assert!(app.run("window.panel.layout", json!({"action":action})).is_err());
        assert_eq!(app.ui.docking, before);
        assert_eq!(app.ui.docking_hidden, hidden);
    }
    app.ui.docking = Some(Layout {
        root: Some(Node::Stack { entries: vec![craft_ui::docking::StackEntry { panel: "layers".into(), open: true, height: Some(100.0) }] }),
        floating: Vec::new(),
    });
    app.run("window.panel.layout", json!({"action":{"SetStackOpen":{"panel":"layers","open":false}}})).unwrap();
    app.run("window.panel.layout", json!({"action":{"ResizeStack":{"panel":"layers","height":150}}})).unwrap();
    let Some(Node::Stack { entries }) = &app.ui.docking.as_ref().unwrap().root else { panic!("expected stack") };
    assert!(!entries[0].open);
    assert_eq!(entries[0].height, Some(150.0));
}

fn group_members_for_test(layout: &Layout<String>, panel: &str) -> Vec<String> {
    let mut pending: Vec<_> = layout.root.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Tabs { panels, .. } if panels.iter().any(|p| p == panel) => return panels.clone(),
            Node::Split { first, second, .. } => pending.extend([first.as_ref(), second.as_ref()]),
            _ => {}
        }
    }
    Vec::new()
}

#[test]
fn customization_retains_native_collapsed_groups_until_their_tab_is_activated() {
    let mut app = app();
    app.ui.dock.set_collapsed(Group::Color, true);
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    let layout = app.ui.docking.as_ref().unwrap();
    assert!(collapsed_members(&app, layout).contains("swatches"));
    let saved = serde_json::to_value(&app.ui).unwrap();
    let restored: crate::state::UiState = serde_json::from_value(saved).unwrap();
    assert!(restored.dock.is_collapsed(Group::Color));
    app.run("window.panel.layout", json!({"action":{"Activate":{"panel":"swatches"}}})).unwrap();
    assert!(!app.ui.dock.is_collapsed(Group::Color));
}

#[test]
fn splitting_a_short_column_keeps_the_layers_footer_visible_and_usable() {
    let mut app = app();
    app.run("file.new", json!({"width":480,"height":360})).unwrap();
    app.run("window.panel.move", json!({"panel":"swatches","anchor":"layers","zone":"top"})).unwrap();
    app.run("window.panel.float", json!({"panel":"properties","x":30,"y":90,"width":300,"height":440})).unwrap();
    let mut h = harness(app, vec2(800.0, 800.0));
    let button = h.get_by_label("Create a new layer").rect();
    assert!(button.bottom() <= 800.0, "Layers footer is clipped below the viewport: {button:?}");
    let before = h.state().session.active().unwrap().doc.layers.len();
    let at = button.center();
    h.hover_at(at);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), before + 1);
}

#[test]
fn populated_canvas_keeps_a_real_opacity_edit_after_floating_and_redocking() {
    let mut app = app();
    app.ui.theme = crate::theme::ThemeKind::Studio;
    app.run("file.new", json!({"width":480,"height":360})).unwrap();
    app.run("layer.new.layer", json!({"name":"Docking sample"})).unwrap();
    app.run("select.rect", json!({"x":80,"y":70,"width":250,"height":160})).unwrap();
    app.run("edit.fill", json!({"color":"#5297bb"})).unwrap();
    app.run("select.deselect", json!({})).unwrap();
    let layer = app.session.active().unwrap().active_layer.unwrap();
    app.run("window.panel.float", json!({"panel":"properties","x":40,"y":140,"width":310,"height":490})).unwrap();
    let directory = std::env::var_os("CRAFT_UI_DOCKING_FIXTURES").map(std::path::PathBuf::from);
    let builder = Harness::builder().with_size(vec2(1280.0, 800.0));
    let builder = if directory.is_some() { builder.wgpu() } else { builder };
    let mut ready = false;
    let mut h = builder.build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if !ready {
                PhotocraftApp::setup_context(ui.ctx(), crate::theme::ThemeKind::Studio);
                ready = true;
                return;
            }
            crate::panels::title_bar(app, ui);
            crate::panels::options_bar(app, ui);
            crate::panels::toolbar(app, ui);
            crate::panels::right_dock(app, ui);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    h.input_mut().max_texture_side = Some(8192);
    h.run_steps(5);
    let floating = h.ctx.memory(|m| m.area_rect(egui::Id::new("photocraft-panel-docking").with(("floating", &"properties".to_string())))).unwrap();
    let mut sliders: Vec<_> = h.ctx.viewport(|vp| {
        vp.prev_pass
            .widgets
            .layers()
            .flat_map(|(_, widgets)| widgets.iter())
            .filter(|widget| widget.sense.senses_drag() && widget.rect.height() == 18.0 && floating.contains_rect(widget.rect))
            .map(|widget| widget.rect)
            .collect()
    });
    sliders.sort_by(|a, b| a.top().total_cmp(&b.top()));
    let at = sliders.first().expect("Properties opacity slider").center();
    h.hover_at(at);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    h.run_steps(2);
    let opacity = h.state().session.active().unwrap().doc.layer(layer).unwrap().opacity;
    assert!((opacity - 0.5).abs() < 0.03, "actual opacity {opacity}");
    if let Some(directory) = &directory {
        std::fs::create_dir_all(directory).unwrap();
        h.render().unwrap().save(directory.join("photocraft-populated-opacity-floating.png")).unwrap();
    }
    let from = h.ctx.read_response(egui::Id::new("photocraft-panel-docking").with(("tab", &"properties".to_string()))).unwrap().rect.center();
    let to = h.ctx.read_response(egui::Id::new("photocraft-panel-docking").with(("tab", &"layers".to_string()))).unwrap().rect.center();
    drag(&mut h, from, to);
    assert!(h.state().ui.docking.as_ref().unwrap().floating.iter().all(|group| !group.panels.iter().any(|panel| panel == "properties")));
    assert_eq!(h.state().session.active().unwrap().doc.layer(layer).unwrap().opacity, opacity);
    if let Some(directory) = &directory {
        h.render().unwrap().save(directory.join("photocraft-populated-opacity-redocked.png")).unwrap();
    }
}

#[test]
fn changing_workspace_during_a_drag_cancels_the_stale_gesture() {
    let mut app = app();
    app.run("window.panel.move", json!({"panel":"swatches","anchor":"layers","zone":"top"})).unwrap();
    let destination = app.ui.docking.clone();
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut app, &ctx, "window.workspace.newWorkspace", json!({"name":"Destination"})).unwrap();
    crate::menus::invoke(&mut app, &ctx, "window.workspace.essentials", json!({})).unwrap();
    let mut h = harness(app, vec2(1280.0, 800.0));
    let source = h.query_all_by_label("Layers").find(|node| node.rect().height() <= 45.0).unwrap().rect().center();
    h.event(Event::PointerMoved(source));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(Event::PointerMoved(source - vec2(90.0, 0.0)));
    h.run_steps(2);
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, "window.workspace.select", json!({"name":"Destination"})).unwrap();
    h.event(Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
    assert_eq!(h.state().ui.docking, destination);
    h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert_eq!(h.state().ui.docking, destination);
}

#[test]
fn first_legacy_drag_workspace_switch_preserves_custom_and_legacy_destinations() {
    let layout_state = |app: &PhotocraftApp| json!({"panels":app.ui.panels,"dock":app.ui.dock,"tabs":app.ui.dock_tabs,"layout":app.ui.docking,"hidden":app.ui.docking_hidden,"collapsed":app.ui.docking_collapsed,"timeline":app.ui.timeline.open});
    for customized_destination in [true, false] {
        for cancellation in ["release", "escape", "focus-loss"] {
            let mut app = app();
            let ctx = egui::Context::default();
            if customized_destination {
                app.run("window.panel.move", json!({"panel":"swatches","anchor":"layers","zone":"top"})).unwrap();
                crate::menus::invoke(&mut app, &ctx, "window.workspace.newWorkspace", json!({"name":"Destination"})).unwrap();
                crate::menus::invoke(&mut app, &ctx, "window.workspace.essentials", json!({})).unwrap();
            }
            let mut h = harness(app, vec2(1280.0, 800.0));
            let source = h.query_all_by_label("Layers").find(|node| node.rect().height() <= 45.0).unwrap().rect().center();
            h.event(Event::PointerMoved(source));
            h.run_steps(1);
            h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
            h.run_steps(1);
            h.event(Event::PointerMoved(source - vec2(90.0, 0.0)));
            // One frame installs the first legacy drag; no shared dock has rendered yet.
            h.run_steps(1);
            assert!(h.state().ui.docking.is_some());
            let origin = egui::Id::new("photocraft-panel-docking").with("legacy-origin");
            assert!(h.ctx.data_mut(|data| data.get_temp::<LegacyDrag>(origin)).unwrap().layout.is_none());
            let ctx = h.ctx.clone();
            let command = if customized_destination { "window.workspace.select" } else { "window.workspace.painting" };
            crate::menus::invoke(h.state_mut(), &ctx, command, json!({"name":"Destination"})).unwrap();
            assert_eq!(h.state().ui.docking.is_some(), customized_destination);
            let destination = layout_state(h.state());
            let tabs = serde_json::to_value(h.state().ui.dock_tabs).unwrap();
            match cancellation {
                "escape" => h.event(Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }),
                "focus-loss" => h.event(Event::WindowFocused(false)),
                _ => {
                    h.event(Event::PointerButton { pos: source - vec2(90.0, 0.0), button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE })
                }
            }
            h.run_steps(1);
            assert_eq!(layout_state(h.state()), destination, "{customized_destination}/{cancellation}");
            if cancellation == "focus-loss" {
                let at = source - vec2(90.0, 0.0);
                h.event(Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed: true, modifiers: Modifiers::NONE });
                h.run_steps(1);
                assert_eq!(layout_state(h.state()), destination, "secondary press must not resume the stale primary drag");
                h.event(Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed: false, modifiers: Modifiers::NONE });
                h.run_steps(1);
            }
            assert_eq!(serde_json::to_value(h.state().ui.dock_tabs).unwrap(), tabs);
            assert!(h.ctx.data_mut(|data| data.get_temp::<LegacyDrag>(origin)).is_none(), "stale origin: {customized_destination}/{cancellation}");
            h.event(Event::WindowFocused(true));
            h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
            h.run_steps(2);
            assert_eq!(layout_state(h.state()), destination);
        }
    }
}

fn control_set(app: &mut PhotocraftApp, ctx: &egui::Context, params: Value) -> Value {
    let (request, _) = crate::control::ControlRequest::new("ui.set", params);
    match crate::control::handle(app, ctx, &request) {
        crate::control::Outcome::Done(value) => value,
        _ => panic!("ui.set must reply synchronously"),
    }
}

#[test]
fn current_main_hidden_tabs_migrate_by_name_and_selected_identity() {
    for theme in [crate::theme::ThemeKind::ProMedium, crate::theme::ThemeKind::Studio] {
        let mut app = app();
        app.ui.theme = theme;
        let pro = matches!(theme, crate::theme::ThemeKind::Pro | crate::theme::ThemeKind::ProMedium);
        let color = Group::Color.tabs(pro).iter().position(|name| *name == "Color").unwrap();
        let swatches = Group::Color.tabs(pro).iter().position(|name| *name == "Swatches").unwrap();
        app.ui.dock.hide_tab(Group::Color, swatches, pro);
        app.ui.dock.hide_tab(Group::Layers, 2, pro);
        app.ui.dock_tabs.color = color;
        let saved = json!({"panels":app.ui.panels,"dock":app.ui.dock,"dockTabs":app.ui.dock_tabs});
        assert_eq!(saved["dock"]["hidden_tabs"]["layers"], json!(["Paths"]));
        let mut restored = self::app();
        restored.ui.theme = theme;
        crate::dock::apply(&mut restored, &saved);
        restored.run("window.panel.float", json!({"panel":"properties"})).unwrap();
        let layout = restored.ui.docking.as_ref().unwrap();
        assert!(!layout.contains(&"swatches".into()) && !layout.contains(&"paths".into()));
        assert!(panel_active(layout, "color"));
        assert!(restored.ui.docking_hidden.contains_key("swatches") && restored.ui.docking_hidden.contains_key("paths"));
        restored.run("window.panel.swatches", json!({"show":true})).unwrap();
        assert!(panel_active(restored.ui.docking.as_ref().unwrap(), "swatches"));
    }
}

#[test]
fn legacy_control_updates_authoritative_custom_visibility_and_selection_atomically() {
    let mut app = app();
    app.ui.theme = crate::theme::ThemeKind::ProMedium;
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    let ctx = egui::Context::default();
    assert_eq!(control_set(&mut app, &ctx, json!({"dockTabs":{"layers":1}}))["ok"], true);
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "channels"));
    assert_eq!(control_set(&mut app, &ctx, json!({"panels":{"layers":false}}))["ok"], true);
    assert!(PANELS.iter().filter(|p| p.2 == Group::Layers).all(|p| !app.ui.docking.as_ref().unwrap().contains(&p.0.into())));
    assert!(!app.ui.panels.layers);
    assert_eq!(control_set(&mut app, &ctx, json!({"panels":{"layers":true},"dockTabs":{"layers":2}}))["ok"], true);
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "paths"));
    let before = serde_json::to_value(&app.ui).unwrap();
    for request in [
        json!({"panels":{"color":false},"dockTabs":{"layers":999}}),
        json!({"tool":"brush","dock":{"order":["layers","color"]}}),
        json!({"panels":{"layers":false},"dockTabs":{"layers":1}}),
    ] {
        assert_eq!(control_set(&mut app, &ctx, request)["ok"], false);
        assert_eq!(serde_json::to_value(&app.ui).unwrap(), before);
    }
}

#[test]
fn legacy_dock_width_is_consumed_by_custom_renderer_and_clamped_to_existing_range() {
    let mut app = app();
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    let mut h = harness(app, vec2(1280.0, 800.0));
    let ctx = h.ctx.clone();
    for (requested, expected) in [(410.0, 410.0), (900.0, 520.0), (20.0, 250.0)] {
        assert_eq!(control_set(h.state_mut(), &ctx, json!({"dockWidth":requested}))["ok"], true);
        h.run_steps(2);
        let state = egui::containers::panel::PanelState::load(&h.ctx, egui::Id::new("shared-panel-dock")).unwrap();
        assert!((state.outer_rect.width() - expected).abs() < 1.0, "{} / {expected}", state.outer_rect.width());
        assert!(h.ctx.data(|data| data.get_temp::<f32>(crate::panels::dock_width_id())).is_none());
    }
}

#[test]
fn explicit_reveal_activates_existing_tab_and_expands_custom_group() {
    let mut app = app();
    app.run("window.panel.move", json!({"panel":"properties","anchor":"layers"})).unwrap();
    app.run("window.panel.group", json!({"panel":"layers","operation":"collapse"})).unwrap();
    app.run("window.panel.layers", json!({"show":true})).unwrap();
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "layers"));
    assert!(!collapsed_members(&app, app.ui.docking.as_ref().unwrap()).contains("layers"));
    app.run("window.panel.layers", json!({"show":true})).unwrap();
    assert!(app.ui.docking.as_ref().unwrap().contains(&"layers".into()));
    let before = app.ui.docking.clone();
    assert!(app.run("window.panel.layers", json!({"show":"yes"})).is_err());
    assert_eq!(app.ui.docking, before);
}

// #4028: in a custom workspace too, a new adjustment layer brings Properties to the front.
#[test]
fn a_new_adjustment_layer_activates_properties_in_a_custom_workspace() {
    let mut app = app();
    app.run("window.panel.activate", json!({"panel":"adjustments"})).unwrap();
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "adjustments"));
    app.run("file.new", json!({"width":64,"height":64})).unwrap();
    app.run("layer.newAdjustmentLayer.curves", json!({})).unwrap();
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "properties"));
}

#[test]
fn custom_group_commands_preserve_members_and_workspace_timeline() {
    let mut app = app();
    app.run("window.panel.move", json!({"panel":"properties","anchor":"layers"})).unwrap();
    let members = group_members(app.ui.docking.as_ref().unwrap(), "layers");
    app.run("window.panel.group", json!({"panel":"layers","operation":"up"})).unwrap();
    assert_eq!(group_members(app.ui.docking.as_ref().unwrap(), "layers"), members);
    app.ui.timeline.open = true;
    app.run("window.panel.group", json!({"panel":"layers","operation":"collapse"})).unwrap();
    let saved = json!({"panels":app.ui.panels,"dock":app.ui.dock,"dockTabs":app.ui.dock_tabs,"docking":app.ui.docking,"dockingHidden":app.ui.docking_hidden,"dockingCollapsed":app.ui.docking_collapsed,"timelineOpen":true});
    let mut restored = self::app();
    crate::dock::apply(&mut restored, &saved);
    assert_eq!(restored.ui.docking, app.ui.docking);
    assert!(restored.ui.timeline.open);
    assert!(collapsed_members(&restored, restored.ui.docking.as_ref().unwrap()).contains("layers"));
    app.run("window.panel.group", json!({"panel":"layers","operation":"close"})).unwrap();
    assert!(members.iter().all(|id| !app.ui.docking.as_ref().unwrap().contains(id)));
    assert!(members.iter().all(|id| app.ui.docking_hidden.contains_key(id)));
    app.session.prefs.edit(|p| p.workspace_locked = true);
    let before = serde_json::to_value(&app.ui).unwrap();
    assert!(app.run("window.panel.group", json!({"panel":"color","operation":"close"})).is_err());
    assert_eq!(serde_json::to_value(&app.ui).unwrap(), before);
    restored.ui.timeline.playing = true;
    crate::dock::apply(&mut restored, &json!({"dock":{}}));
    assert!(!restored.ui.timeline.open && !restored.ui.timeline.playing);
}

#[test]
fn customized_panel_menus_and_rail_remain_reachable_after_float_and_restore() {
    let mut app = app();
    app.run("window.panel.float", json!({"panel":"swatches","x":20,"y":40})).unwrap();
    let mut h = harness(app, vec2(1280.0, 800.0));
    assert!(h.query_all_by_label("Panel menu").count() > 0);
    assert!(h.query_all_by_label("Color & Swatches").count() > 0);
    let floating = h.ctx.memory(|memory| memory.area_rect(egui::Id::new("photocraft-panel-docking").with(("floating", &"swatches".to_string())))).unwrap();
    let first = h.query_all_by_label("Panel menu").find(|node| floating.contains_rect(node.rect())).unwrap().rect().center();
    h.hover_at(first);
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: first, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    h.run_steps(2);
    assert!(h.query_all_by_label("Close Tab Group").count() > 0);
    assert!(h.query_all_by_label("New Swatch…").count() > 0);
    // Panel-specific menu body is shared by both its context and group menu seams.
    let mut ready = false;
    let mut menu = Harness::builder().build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if !ready {
                PhotocraftApp::setup_context(ui.ctx(), crate::theme::ThemeKind::ProMedium);
                ready = true;
            }
            panel_menu(app, ui, "swatches");
        },
        self::app(),
    );
    menu.run_steps(2);
    assert!(menu.query_all_by_label("New Swatch…").count() > 0);
}

#[test]
fn custom_workspace_saves_shared_layout_and_timeline_without_embedding_flags() {
    let mut app = app();
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    app.ui.timeline.open = true;
    app.ui.panels.rail = false;
    app.ui.panels.menu_bar = false;
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut app, &ctx, "window.workspace.newWorkspace", json!({"name":"Docked Motion"})).unwrap();
    let saved = app.session.prefs().workspaces.get("Docked Motion").unwrap().clone();
    assert!(saved["docking"].is_object() && saved["timelineOpen"] == true);
    assert!(saved["panels"].get("rail").is_none() && saved["panels"].get("menu_bar").is_none());
    let layout = app.ui.docking.clone();
    app.ui.workspace = "Motion".into();
    crate::menus::apply_workspace(&mut app);
    assert!(app.ui.docking.is_none() && app.ui.timeline.open);
    crate::dock::apply(&mut app, &saved);
    assert_eq!(app.ui.docking, layout);
    assert!(app.ui.timeline.open);
}

#[test]
fn closing_and_reopening_a_whole_group_retains_its_return_position_and_tab_order() {
    let mut app = app();
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    let before = app.ui.docking.clone().unwrap();
    let members = group_members(&before, "layers");
    app.run("window.panel.group", json!({"panel":"layers","operation":"close"})).unwrap();
    for panel in &members {
        app.run("window.panel.activate", json!({"panel":panel})).unwrap();
    }
    let after = app.ui.docking.as_ref().unwrap();
    assert_eq!(group_members(after, "layers"), members);
    assert_eq!(root_groups(after), root_groups(&before));
    assert!(valid(after));
}

#[test]
fn floating_group_grip_redocks_all_tabs_as_one_atomic_move() {
    let mut app = app();
    app.run("window.panel.float", json!({"panel":"swatches","x":30,"y":60})).unwrap();
    app.run("window.panel.move", json!({"panel":"patterns","anchor":"swatches"})).unwrap();
    let mut h = harness(app, vec2(1280.0, 800.0));
    let from = h.ctx.read_response(egui::Id::new("photocraft-panel-docking").with(("move-group", Some(&"swatches".to_string())))).unwrap().rect.center();
    let to = h.ctx.read_response(egui::Id::new("photocraft-panel-docking").with(("tab", &"layers".to_string()))).unwrap().rect.center();
    drag(&mut h, from, to);
    let layout = h.state().ui.docking.as_ref().unwrap();
    assert!(layout.floating.iter().all(|group| !group.panels.iter().any(|id| id == "swatches" || id == "patterns")));
    let members = group_members(layout, "layers");
    let swatches = members.iter().position(|id| id == "swatches").unwrap();
    assert_eq!(members.get(swatches + 1).map(String::as_str), Some("patterns"));
    assert!(valid(layout));
    let before = h.state().ui.docking.clone();
    assert!(h.state_mut().run("window.panel.group", json!({"panel":"swatches","operation":"move","anchor":"patterns"})).is_err());
    assert_eq!(h.state().ui.docking, before);
}

#[test]
fn mixed_theme_and_legacy_selection_projects_the_selected_panel_in_the_new_theme() {
    let mut app = app();
    app.ui.theme = crate::theme::ThemeKind::ProMedium;
    app.run("window.panel.float", json!({"panel":"properties"})).unwrap();
    let ctx = egui::Context::default();
    assert_eq!(control_set(&mut app, &ctx, json!({"theme":"studio","dockTabs":{"color":0}}))["ok"], true);
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "swatches"));
    assert_eq!(app.ui.dock_tabs.color, 0);
    assert_eq!(control_set(&mut app, &ctx, json!({"theme":"proMedium","dockTabs":{"color":0}}))["ok"], true);
    assert!(panel_active(app.ui.docking.as_ref().unwrap(), "color"));
    assert_eq!(app.ui.dock_tabs.color, 0);
}

#[test]
fn native_tab_reorder_stays_native_until_drag_crosses_the_strip() {
    let mut h = harness(app(), vec2(1280.0, 800.0));
    let source = h.query_all_by_label("Layers").find(|node| node.rect().height() <= 45.0).unwrap().rect().center();
    let target = h.query_all_by_label("Channels").find(|node| node.rect().height() <= 45.0).unwrap().rect().right_center() - vec2(1.0, 0.0);
    h.event(Event::PointerMoved(source));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: source, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    let near_edge = egui::pos2(target.x, source.y - 21.0);
    h.event(Event::PointerMoved(near_edge));
    h.run_steps(1);
    assert!(h.state().ui.docking.is_none(), "native near-edge reorder tolerance must not promote to custom docking");
    h.event(Event::PointerMoved(target));
    h.run_steps(1);
    assert!(h.state().ui.docking.is_none(), "native horizontal reorder must not promote to custom docking");
    h.event(Event::PointerButton { pos: target, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert!(h.state().ui.docking.is_none());
    let tabs = h.state().ui.dock.visible_tabs(Group::Layers, false);
    let layers = tabs.iter().position(|(_, label)| *label == "Layers").unwrap();
    let channels = tabs.iter().position(|(_, label)| *label == "Channels").unwrap();
    assert!(channels < layers, "upstream native reorder remains effective");
}
