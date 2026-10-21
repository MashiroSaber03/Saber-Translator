//! #1120: workspace selection/reset must apply useful layouts, including saved layouts.

use photocraft_engine::Session;
use photocraft_ui_egui::dock::{self, Group};
use photocraft_ui_egui::{PhotocraftApp, menus};
use serde_json::{Value, json};

const PRESETS: [(&str, &str, &[Group], bool); 6] = [
    ("essentials", "Essentials", &[Group::Color, Group::Properties, Group::Layers], false),
    ("photography", "Photography", &[Group::Properties, Group::Navigator, Group::History, Group::Layers], false),
    ("painting", "Painting", &[Group::Color, Group::Layers], false),
    ("graphicAndWeb", "Graphic and Web", &[Group::Properties, Group::Character, Group::Layers], false),
    ("pixelArt", "Pixel Art", &[Group::Color, Group::Navigator, Group::Layers], false),
    ("motion", "Motion", &[Group::Properties, Group::Layers], true),
];

fn app() -> PhotocraftApp {
    PhotocraftApp::new(Session::new(), Default::default())
}

fn invoke(app: &mut PhotocraftApp, id: &str, params: Value) {
    menus::invoke(app, &egui::Context::default(), id, params).unwrap();
}

fn select(app: &mut PhotocraftApp, suffix: &str) {
    invoke(app, &format!("window.workspace.{suffix}"), json!({}));
}

fn layout(app: &PhotocraftApp) -> Value {
    json!({"panels": app.ui.panels, "dockTabs": app.ui.dock_tabs, "dock": app.ui.dock, "timelineOpen": app.ui.timeline.open})
}

fn shown(app: &PhotocraftApp) -> Vec<Group> {
    app.ui.dock.order().into_iter().filter(|g| g.shown(&app.ui.panels)).collect()
}

#[test]
fn reported_presets_have_distinct_useful_layouts() {
    let mut app = app();
    let mut failures = Vec::new();
    for (suffix, name, groups, timeline) in PRESETS.into_iter().skip(3) {
        select(&mut app, suffix);
        if shown(&app) != groups || app.ui.timeline.open != timeline {
            failures.push(format!("{name}: {:?}, timeline={}", shown(&app), app.ui.timeline.open));
        }
    }
    assert!(failures.is_empty(), "incorrect layouts: {failures:?}");
}

#[test]
fn presets_select_expected_panels_and_reset_every_dock_group() {
    let mut app = app();
    app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
    let doc = app.session.active().unwrap().doc.clone();
    for (suffix, name, groups, timeline) in PRESETS {
        select(&mut app, suffix);
        assert_eq!(app.ui.workspace, name);
        assert_eq!(shown(&app), groups, "{name}");
        assert_eq!(app.ui.timeline.open, timeline, "{name}");
        let id = format!("window.workspace.{suffix}");
        assert_eq!(menus::menu_items(&app).iter().find(|item| item.id == id).unwrap().checked, Some(true));
        let expected = layout(&app);
        app.ui.panels.character = !app.ui.panels.character;
        app.ui.panels.navigator = !app.ui.panels.navigator;
        app.ui.timeline.open = !timeline;
        app.ui.dock.move_group(Group::Layers, Some(Group::Color));
        app.ui.dock.set_collapsed(Group::Layers, true);
        app.ui.dock.heights.insert(Group::Properties, 99.0);
        app.ui.dock_tabs.color = 2;
        app.ui.dock_tabs.layers = 2;
        invoke(&mut app, "window.workspace.resetWorkspace", json!({}));
        assert_eq!(layout(&app), expected, "reset {name}");
        assert_eq!(app.session.active().unwrap().doc, doc, "layout must not create/edit a timeline");
    }
}

#[test]
fn presets_are_deterministic_from_every_other_preset_and_select_route() {
    for (suffix, name, _, _) in PRESETS {
        let mut baseline = app();
        select(&mut baseline, suffix);
        let expected = layout(&baseline);
        for (prior, _, _, _) in PRESETS {
            let mut app = app();
            select(&mut app, prior);
            invoke(&mut app, "window.workspace.select", json!({"name": name}));
            assert_eq!(layout(&app), expected, "{prior} -> {name}");
        }
    }
}

#[test]
fn preset_layouts_round_trip_through_remembered_preferences() {
    for (suffix, name, _, _) in PRESETS {
        let mut app = app();
        select(&mut app, suffix);
        let expected = layout(&app);
        dock::persist(&mut app, &egui::Context::default());
        let mut session = Session::new();
        session.load_prefs_json(&app.session.prefs_to_json()).unwrap();
        let mut restored = PhotocraftApp::new(session, Default::default());
        dock::restore(&mut restored);
        assert_eq!(restored.ui.workspace, name);
        assert_eq!(layout(&restored), expected, "restore {name}");
    }
}

#[test]
fn open_timeline_is_saved_without_saving_playback() {
    let mut app = app();
    invoke(&mut app, "window.panel.timeline", json!({}));
    assert!(app.ui.timeline.open);
    app.ui.timeline.playing = true;
    invoke(&mut app, "window.workspace.newWorkspace", json!({"name": "Open timeline"}));
    dock::persist(&mut app, &egui::Context::default());
    let prefs = app.session.prefs_to_json();
    let mut session = Session::new();
    session.load_prefs_json(&prefs).unwrap();
    let mut restored = PhotocraftApp::new(session, Default::default());
    dock::restore(&mut restored);
    assert!(restored.ui.timeline.open);
    assert!(!restored.ui.timeline.playing);
    select(&mut restored, "essentials");
    invoke(&mut restored, "window.workspace.select", json!({"name": "Open timeline"}));
    assert!(restored.ui.timeline.open);
    assert!(!restored.ui.timeline.playing);
}

#[test]
fn custom_workspaces_restore_timeline_visibility_and_dock_after_presets() {
    let mut app = app();
    for (name, suffix) in [("Custom still", "pixelArt"), ("Custom motion", "motion")] {
        select(&mut app, suffix);
        app.ui.dock.move_group(Group::Navigator, Some(Group::Color));
        app.ui.dock_tabs.color = 1;
        app.ui.dock.set_collapsed(Group::Properties, true);
        let expected = layout(&app);
        invoke(&mut app, "window.workspace.newWorkspace", json!({"name": name}));
        let mut session = Session::new();
        session.load_prefs_json(&app.session.prefs_to_json()).unwrap();
        let mut restored = PhotocraftApp::new(session, Default::default());
        select(&mut restored, if suffix == "motion" { "essentials" } else { "motion" });
        invoke(&mut restored, "window.workspace.select", json!({"name": name}));
        assert_eq!(layout(&restored), expected, "custom select {name}");
        restored.ui.timeline.open = !restored.ui.timeline.open;
        restored.ui.panels.color = !restored.ui.panels.color;
        invoke(&mut restored, "window.workspace.resetWorkspace", json!({}));
        assert_eq!(layout(&restored), expected, "custom reset {name}");
    }
}

#[test]
fn legacy_workspace_without_timeline_visibility_still_loads() {
    let mut app = app();
    select(&mut app, "painting");
    let mut legacy = layout(&app);
    legacy.as_object_mut().unwrap().remove("timelineOpen");
    app.session.prefs.edit(|p| p.workspaces.insert("Legacy".into(), legacy));
    select(&mut app, "motion");
    invoke(&mut app, "window.workspace.select", json!({"name": "Legacy"}));
    assert_eq!(shown(&app), [Group::Color, Group::Layers]);
    assert!(!app.ui.timeline.open);
}

#[test]
fn presets_preserve_shell_preferences_and_workspace_lock() {
    let mut app = app();
    app.ui.panels.toolbar = false;
    app.ui.panels.toolbar_double = true;
    app.ui.panels.options_bar = false;
    app.ui.panels.status_bar = false;
    app.session.prefs.edit(|p| p.workspace_locked = true);
    for (suffix, _, _, _) in PRESETS {
        select(&mut app, suffix);
        assert!(!app.ui.panels.toolbar);
        assert!(app.ui.panels.toolbar_double);
        assert!(!app.ui.panels.options_bar);
        assert!(!app.ui.panels.status_bar);
        assert!(app.session.prefs().workspace_locked);
    }
}

#[test]
fn titlebar_picker_displays_every_selected_preset() {
    fn collect_text(shape: &egui::Shape, texts: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(text) => texts.push(text.galley.text().to_string()),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| collect_text(shape, texts)),
            _ => {}
        }
    }

    let mut missing = Vec::new();
    for (suffix, name, _, _) in PRESETS {
        let mut app = app();
        select(&mut app, suffix);
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, Default::default());
        let mut texts = Vec::new();
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 900.0))), ..Default::default() };
            let mut output = ctx.run_ui(input, |ui| photocraft_ui_egui::panels::title_bar(&mut app, ui));
            output.textures_delta.clear();
            texts.clear();
            for shape in output.shapes {
                collect_text(&shape.shape, &mut texts);
            }
        }
        if !texts.iter().any(|text| text == name) {
            missing.push(name);
        }
    }
    assert!(missing.is_empty(), "selected presets missing from titlebar picker: {missing:?}");
}
