use egui::vec2;
use egui_kittest::Harness;
use photocraft_engine::paint::{Control, GrayTile, MaskMode, Pattern, PatternStyle, TipShape};

use super::*;

fn app() -> PhotocraftApp {
    PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default())
}

fn last_journal(app: &PhotocraftApp) -> Option<(String, Value)> {
    app.session.journal.last().cloned()
}

#[test]
fn sections_with_boxes_have_flags() {
    let mut b = BrushSettings::default();
    for (i, (_, has_box)) in SECTIONS.iter().enumerate() {
        assert_eq!(section_flag(&mut b, i).is_some(), *has_box, "{i}");
    }
    *section_flag(&mut b, 3).unwrap() = true;
    assert!(b.texture.enabled);
}

/// What each section's controls edit, set to non-default values.
fn edit_section(i: usize, b: &mut BrushSettings) {
    let dynamic = |jitter: f32, control: Control, minimum: f32| paint::Dynamic { jitter, control, fade_steps: 40, minimum };
    match i {
        0 => {
            b.size = 87.0;
            b.flip_x = true;
            b.flip_y = true;
            b.angle = -35.0;
            b.roundness = 0.4;
            b.hardness = 0.25;
            b.spacing = 0.6;
            b.spacing_enabled = false;
        }
        1 => {
            let sd = &mut b.shape_dynamics;
            sd.enabled = true;
            sd.size = dynamic(0.3, Control::Fade, 0.2);
            sd.angle = dynamic(0.1, Control::Direction, 0.0);
            sd.roundness = dynamic(0.5, Control::PenTilt, 0.25);
            sd.flip_x_jitter = true;
            sd.flip_y_jitter = true;
            sd.tilt_scale = 1.25;
            sd.brush_projection = true;
        }
        2 => {
            let sc = &mut b.scattering;
            sc.enabled = true;
            sc.scatter = dynamic(2.5, Control::PenPressure, 0.0);
            sc.both_axes = true;
            sc.count = 4;
            sc.count_jitter = dynamic(0.4, Control::StylusWheel, 0.0);
        }
        3 => {
            let tx = &mut b.texture;
            tx.enabled = true;
            tx.pattern = Pattern::Procedural { style: PatternStyle::Canvas, size: 200, seed: 1 };
            tx.invert = true;
            tx.scale = 1.5;
            tx.brightness = -0.2;
            tx.contrast = 0.6;
            tx.each_tip = true;
            tx.mode = MaskMode::HardMix;
            tx.depth = 0.7;
            tx.depth_jitter = dynamic(0.3, Control::Rotation, 0.1);
        }
        4 => {
            let d = &mut b.dual_brush;
            d.enabled = true;
            d.mode = MaskMode::ColorBurn;
            d.flip = true;
            d.tip = TipShape::Sampled(GrayTile::from_fn(8, 6, |x, y| ((x + y) % 2) as f32));
            d.size = 33.0;
            d.hardness = 0.5;
            d.spacing = 0.8;
            d.scatter = 1.2;
            d.both_axes = true;
            d.count = 3;
        }
        5 => {
            let c = &mut b.color_dynamics;
            c.enabled = true;
            c.per_tip = false;
            c.fg_bg = dynamic(0.6, Control::Fade, 0.0);
            c.hue_jitter = 0.1;
            c.saturation_jitter = 0.2;
            c.brightness_jitter = 0.3;
            c.purity = -0.4;
        }
        6 => {
            b.transfer.enabled = true;
            b.transfer.opacity = dynamic(0.5, Control::PenPressure, 0.1);
            b.transfer.flow = dynamic(0.25, Control::Fade, 0.3);
            b.transfer.wetness = dynamic(0.35, Control::PenPressure, 0.1);
            b.transfer.mix = dynamic(0.45, Control::Fade, 0.2);
            b.mixer = paint::MixerSettings { wet: 0.9, load: 0.15, mix: 0.7, flow: 0.8, sample_all_layers: true };
        }
        7 => {
            let p = &mut b.pose;
            p.enabled = true;
            p.tilt_x = 45.0;
            p.tilt_y = -27.0;
            p.override_tilt = true;
            p.rotation = 120.0;
            p.override_rotation = true;
            p.pressure = 0.4;
            p.override_pressure = true;
        }
        8 => b.noise = true,
        9 => b.wet_edges = true,
        10 => {
            b.build_up = true;
            b.build_up_rate = 55.0;
        }
        11 => {
            let s = &mut b.smoothing;
            s.amount = 0.45;
            s.pulled_string = true;
            s.catch_up = false;
            s.catch_up_on_end = false;
            s.adjust_for_zoom = false;
        }
        _ => b.protect_texture = true,
    }
}

#[test]
fn every_section_round_trips_through_set_brush() {
    for (i, (name, _)) in SECTIONS.iter().enumerate() {
        let mut app = app();
        let before = app.session.tools.brush.clone();
        let mut after = before.clone();
        edit_section(i, &mut after);
        assert_ne!(before, after, "section {i} edits nothing");
        commit(&mut app, &before, &after);
        assert_eq!(app.session.tools.brush, after, "section {i} ({name})");
        // Through the command, with only the changed fields.
        let (id, p) = last_journal(&app).expect("journaled");
        assert_eq!(id, "tools.setBrush");
        let patch = &p["brush"];
        assert!(patch.as_object().is_some_and(|o| !o.is_empty() && o.len() <= 8), "section {i}: {patch}");
        // And the engine's view (`brush.get`) reads back the same settings.
        let got = app.session.execute("brush.get", json!({})).unwrap();
        assert_eq!(serde_json::from_value::<BrushSettings>(got).unwrap(), after, "section {i}");
        // Replaying the journaled call onto a fresh session gives the same brush (drivable).
        let mut s = photocraft_engine::Session::new();
        s.execute(&id, p).unwrap();
        assert_eq!(s.tools.brush, after, "replay of section {i}");
    }
}

#[test]
fn patches_are_minimal_and_skip_unchanged_bitmaps() {
    let tip = TipShape::Sampled(GrayTile::from_fn(400, 300, |x, y| ((x * y) % 7) as f32 / 7.0));
    let old = BrushSettings {
        tip: tip.clone(),
        texture: paint::Texture { pattern: Pattern::Tile(GrayTile::from_fn(64, 64, |x, _| x as f32 / 64.0)), ..Default::default() },
        ..Default::default()
    };
    let new = BrushSettings { size: 44.0, ..old.clone() };
    assert_eq!(brush_patch(&old, &new), json!({"size": 44.0}));
    let new = BrushSettings { shape_dynamics: paint::ShapeDynamics { enabled: true, ..Default::default() }, ..old.clone() };
    assert_eq!(brush_patch(&old, &new), json!({"shapeDynamics": {"enabled": true}}));
    assert_eq!(brush_patch(&old, &old), json!({}));
    // A changed tip is sent whole.
    let new = BrushSettings { tip: TipShape::Round, ..old.clone() };
    assert_eq!(brush_patch(&old, &new), json!({"tip": "round"}));
    // Applying any of these through the engine gives back `new`.
    let mut app = app();
    app.session.tools.brush = old.clone();
    let new = BrushSettings { size: 9.0, roundness: 0.5, ..old.clone() };
    commit(&mut app, &old, &new);
    assert_eq!(app.session.tools.brush, new);
    // No change, no command.
    let n = app.session.journal.len();
    commit(&mut app, &new, &new);
    assert_eq!(app.session.journal.len(), n);
}

#[test]
fn bad_patches_fail_without_touching_the_brush() {
    let mut app = app();
    let before = app.session.tools.brush.clone();
    for bad in [
        json!({"brush": {"size": "huge"}}),
        json!({"brush": {"shapeDynamics": {"size": {"control": "telepathy"}}}}),
        json!({"brush": {"tip": {"sampled": {"width": 0}}}}),
    ] {
        assert!(app.run("tools.setBrush", bad).is_err());
        assert_eq!(app.session.tools.brush, before);
    }
}

#[test]
fn section_locks_round_trip_and_survive_preset_picks() {
    let mut b = BrushSettings::default();
    for (i, (name, _)) in SECTIONS.iter().enumerate() {
        assert_eq!(section_lock(&mut b, i).is_some(), i > 0, "{name} has a lock");
    }
    let mut app = app();
    let before = app.session.tools.brush.clone();
    let mut after = before.clone();
    edit_section(2, &mut after);
    for i in [2, 6, 8] {
        *section_lock(&mut after, i).unwrap() = true;
    }
    commit(&mut app, &before, &after);
    let (_, p) = last_journal(&app).unwrap();
    assert_eq!(p["brush"]["locks"], json!({"scattering": true, "transfer": true, "noise": true}));
    // Picking a preset keeps the locked sections (and the locks), and the preset still reads as current.
    let chalk = paint::presets::find(&app.session.tools.presets, "Chalk").unwrap().brush.clone();
    app.run("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    let b = &app.session.tools.brush;
    assert_eq!(b.scattering, after.scattering);
    assert!(b.locks.scattering && b.locks.noise);
    assert_eq!(b.tip, chalk.tip);
}

#[test]
fn section_locks_are_always_visible_and_toggle_on_click() {
    use egui_kittest::kittest::Queryable;

    fn collect_icon_rects(shape: &egui::Shape, out: &mut Vec<egui::Rect>) {
        match shape {
            egui::Shape::Rect(r) if (r.rect.size() - vec2(12.0, 12.0)).length() < 0.5 => out.push(r.rect),
            egui::Shape::Mesh(mesh) => out.push(mesh.calc_bounds()),
            egui::Shape::Vec(shapes) => {
                for s in shapes {
                    collect_icon_rects(s, out);
                }
            }
            _ => {}
        }
    }

    let mut h = harness(0, 0);
    let mut icons = Vec::new();
    for clipped in &h.output().shapes {
        collect_icon_rects(&clipped.shape, &mut icons);
    }

    // Every lockable section (1..SECTIONS.len()) paints its lock icon even when unlocked and unhovered.
    for (name, _) in SECTIONS.iter().skip(1) {
        let row = h.get_by_label(name).rect();
        let lock_center = egui::pos2(row.right() - 11.0, row.center().y);
        assert!(icons.iter().any(|r| r.contains(lock_center)), "expected lock icon at {lock_center:?} for {name}");
    }

    // Clicking the lock toggles it on and off without changing the selected section.
    let row = h.get_by_label("Scattering").rect();
    let lock_pos = egui::pos2(row.right() - 11.0, row.center().y);
    assert!(!h.state().session.tools.brush.locks.scattering);
    assert_eq!(h.state().ui.brush_section, 0);

    h.hover_at(lock_pos);
    h.run_steps(1);
    h.drag_at(lock_pos);
    h.run_steps(1);
    h.drop_at(lock_pos);
    h.run_steps(2);
    assert!(h.state().session.tools.brush.locks.scattering);
    assert_eq!(h.state().ui.brush_section, 0);

    h.hover_at(lock_pos);
    h.run_steps(1);
    h.drag_at(lock_pos);
    h.run_steps(1);
    h.drop_at(lock_pos);
    h.run_steps(2);
    assert!(!h.state().session.tools.brush.locks.scattering);
    assert_eq!(h.state().ui.brush_section, 0);
}

/// A harness that shows the options bar with `tool` in the Studio (non-pro) theme.
fn options_bar_harness(tool: crate::state::Tool) -> Harness<'static, PhotocraftApp> {
    let mut app = app();
    app.ui.tool = tool;
    let mut h = Harness::builder().with_size(vec2(1400.0, 60.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::panels::options_bar(app, ui);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
    h.run_steps(4);
    h
}

#[test]
fn clone_stamp_opacity_and_flow_are_journaled_per_gesture() {
    use egui_kittest::kittest::Queryable;
    let mut h = options_bar_harness(crate::state::Tool::CloneStamp);
    for (label, field) in [("Opacity", "opacity"), ("Flow", "flow")] {
        let n = h.state().session.journal.len();
        let rect = h.get_by_label(label).rect();
        let start = egui::pos2(rect.right() + 40.0, rect.center().y);
        h.hover_at(start);
        h.run();
        h.drag_at(start);
        h.run();
        for k in 1..=8 {
            h.hover_at(start - vec2(5.0 * k as f32, 0.0));
            h.run();
        }
        h.drop_at(start - vec2(40.0, 0.0));
        h.run_steps(2);
        let brush = &h.state().session.tools.brush;
        let value = if field == "opacity" { brush.opacity } else { brush.flow };
        assert!(value < 0.95, "the drag lowered {field}: {value}");
        let journal = &h.state().session.journal;
        assert_eq!(journal.len(), n + 1, "one command per gesture");
        let (id, params) = journal.last().unwrap();
        assert_eq!(id, "tools.setBrush");
        assert_eq!(params["brush"][field].as_f64().map(|v| v as f32), Some(value));
    }
    h.state_mut().ui.clone_source = Some([0.0, 0.0]);
    let params = crate::retouch_ui::clone_params(h.state()).unwrap();
    assert_eq!(params["opacity"], json!(h.state().session.tools.brush.opacity * 100.0));
    assert_eq!(params["flow"], json!(h.state().session.tools.brush.flow * 100.0));
    let mut replay = photocraft_engine::Session::new();
    for (id, params) in &h.state().session.journal {
        replay.execute(id, params.clone()).unwrap();
    }
    assert_eq!(replay.tools.brush, h.state().session.tools.brush);
}

#[test]
fn options_bar_edits_are_one_set_brush_per_gesture() {
    use egui_kittest::kittest::Queryable;
    let mut h = options_bar_harness(crate::state::Tool::Brush);
    let n = h.state().session.journal.len();
    h.run_steps(3);
    assert_eq!(h.state().session.journal.len(), n, "showing the bar issues nothing");
    // Drag the Opacity field left: many frames, one journal entry.
    let label = h.get_by_label("Opacity").rect();
    let start = egui::pos2(label.right() + 40.0, label.center().y);
    h.hover_at(start);
    h.run();
    h.drag_at(start);
    h.run();
    for k in 1..=8 {
        h.hover_at(start - vec2(5.0 * k as f32, 0.0));
        h.run();
    }
    h.drop_at(start - vec2(40.0, 0.0));
    h.run_steps(2);
    let opacity = h.state().session.tools.brush.opacity;
    assert!(opacity < 0.95, "the drag lowered the opacity: {opacity}");
    let j = &h.state().session.journal;
    assert_eq!(j.len(), n + 1, "{:?}", &j[n..]);
    let (id, p) = j.last().unwrap();
    assert_eq!(id, "tools.setBrush");
    assert_eq!(p["brush"]["opacity"].as_f64().map(|v| v as f32), Some(opacity));
    // A click is its own gesture: the Pressure for Size switch sits left of its label.
    let label = h.get_by_label("Pressure for Size").rect();
    let was = h.state().session.tools.brush.pressure_size;
    let at = egui::pos2(label.left() - 18.0, label.center().y);
    h.hover_at(at);
    h.run();
    h.drag_at(at);
    h.run();
    h.drop_at(at);
    h.run_steps(2);
    assert_eq!(h.state().session.tools.brush.pressure_size, !was);
    assert_eq!(h.state().session.journal.len(), n + 2);
    assert_eq!(h.state().session.journal.last().unwrap().1["brush"], json!({"pressureSize": !was}));
    // Replaying the journal gives the same brush.
    let mut s = photocraft_engine::Session::new();
    for (id, p) in &h.state().session.journal {
        s.execute(id, p.clone()).unwrap();
    }
    assert_eq!(s.tools.brush, h.state().session.tools.brush);
}

#[test]
fn brush_mode_dropdown_shows_outside_the_pro_themes() {
    // #4512: only the Pro themes drew the Brush's Mode; the Pencil has it in every theme.
    use egui_kittest::kittest::{NodeT, Queryable};
    let mut h = options_bar_harness(crate::state::Tool::Brush);
    let n = h.state().session.journal.len();
    assert!(h.query_by_label("Mode").is_some(), "the Brush's Mode label");
    h.get_all_by_role(egui::accesskit::Role::ComboBox)
        .find(|c| c.accesskit_node().value().as_deref() == Some("Normal"))
        .expect("the Brush's Mode dropdown")
        .click();
    h.run();
    h.key_press(egui::Key::ArrowDown);
    h.run_steps(3);
    assert_eq!(h.state().session.tools.brush.mode, photocraft_doc::BlendMode::Dissolve);
    let j = &h.state().session.journal;
    assert_eq!(j.len(), n + 1, "{:?}", &j[n..]);
    assert_eq!(j[n].0, "tools.setBrush");
}

#[test]
fn airbrush_smoothing_options_and_symmetry_buttons_work() {
    // #1177: in the default (Photoshop) theme these options-bar buttons did nothing.
    use egui_kittest::kittest::Queryable;
    let mut h = app_harness();
    let brush = |h: &Harness<'static, PhotocraftApp>| h.state().session.tools.brush.clone();

    // The airbrush toggles Build-up, one setBrush each click.
    let was = brush(&h).build_up;
    h.get_by_label("Enable airbrush-style build-up effects").click();
    h.run_steps(3);
    assert_eq!(brush(&h).build_up, !was);
    let (id, p) = last_journal(h.state()).unwrap();
    assert_eq!((id.as_str(), &p["brush"]), ("tools.setBrush", &json!({"buildUp": !was})));

    // The gear opens Photoshop's four smoothing options, each editing the brush.
    assert!(h.query_by_label("Pulled String Mode").is_none());
    h.get_by_label("Set additional smoothing options").click();
    h.run_steps(3);
    let before = brush(&h).smoothing;
    for label in ["Pulled String Mode", "Stroke Catch-up", "Catch-up on Stroke End", "Adjust for Zoom"] {
        h.get_by_label(label).click();
        h.run_steps(3);
    }
    let after = brush(&h).smoothing;
    assert_eq!(
        (after.pulled_string, after.catch_up, after.catch_up_on_end, after.adjust_for_zoom),
        (!before.pulled_string, !before.catch_up, !before.catch_up_on_end, !before.adjust_for_zoom)
    );
    assert_eq!(after.amount, before.amount);

    // Symmetry: the menu lists the document's paths and turns symmetry on and off.
    h.state_mut().run("file.new", json!({"width": 80, "height": 60})).unwrap();
    h.state_mut().run("path.set", json!({"name": "work", "path": {"subpaths": [{"closed": false, "knots": [[40, 0], [40, 60]]}]}})).unwrap();
    h.run_steps(3);
    h.get_by_label("Set painting symmetry options").click();
    h.run_steps(3);
    h.get_by_label("Work Path").click();
    h.run_steps(3);
    let source = |h: &Harness<'static, PhotocraftApp>| h.state().session.active().unwrap().symmetry.as_ref().and_then(|a| a.path_source().map(str::to_owned));
    assert_eq!(source(&h).as_deref(), Some("work"));
    assert_eq!(last_journal(h.state()).unwrap(), ("paint.symmetryFromPath".to_string(), json!({"name": "work"})));
    h.get_by_label("Set painting symmetry options").click();
    h.run_steps(3);
    h.get_by_label("Symmetry Off").click();
    h.run_steps(3);
    assert_eq!(source(&h), None);
    assert!(h.state().ui.status.is_empty(), "{}", h.state().ui.status);
}

#[test]
fn mixer_brush_options_update_the_persisted_mixer_settings() {
    use egui_kittest::kittest::Queryable;
    let mut h = options_bar_harness(crate::state::Tool::MixerBrush);
    for label in ["Wet", "Load", "Mix", "Flow", "Sample All Layers"] {
        assert!(h.get_by_label(label).rect().is_positive(), "missing Mixer Brush option {label}");
    }

    let initial = h.state().session.journal.len();
    let at = h.get_by_label("Sample All Layers").rect().center();
    h.hover_at(at);
    h.run();
    h.drag_at(at);
    h.run();
    h.drop_at(at);
    h.run_steps(2);

    assert!(h.state().session.tools.brush.mixer.sample_all_layers);
    assert_eq!(h.state().session.journal.len(), initial + 1);
    let (id, params) = h.state().session.journal.last().unwrap();
    assert_eq!(id, "tools.setBrush");
    assert_eq!(params["brush"]["mixer"]["sampleAllLayers"], true);
}

#[test]
fn drop_targets_and_actions_reorder_presets() {
    use crate::brushes_tab::{Action, apply, drop_target, group_key};
    let mut app = app();
    for (n, g) in [("A1", "Alpha"), ("A2", "Alpha"), ("A3", "Alpha"), ("B1", "Beta")] {
        app.session.tools.presets.push(paint::BrushPreset {
            name: n.into(),
            brush: Default::default(),
            builtin: false,
            group: g.into(),
            folder: Vec::new(),
            captures: Default::default(),
        });
    }
    let p = &app.session.tools.presets;
    // Dropping A1 on A3's lower half lands after A3 (index 2 once A1 is out); on its upper half, before it.
    assert_eq!(drop_target(p, "A1", "A3", true), Some(("Alpha".into(), vec![], 2)));
    assert_eq!(drop_target(p, "A1", "A3", false), Some(("Alpha".into(), vec![], 1)));
    assert_eq!(drop_target(p, "A3", "A1", false), Some(("Alpha".into(), vec![], 0)));
    assert_eq!(drop_target(p, "A2", "B1", true), Some(("Beta".into(), vec![], 1)));
    assert_eq!(drop_target(p, "A2", "Nope", true), None);
    assert_eq!(group_key(p, UNGROUPED), "");
    let (group, folder, index) = drop_target(p, "A1", "A3", true).unwrap();
    apply(&mut app, vec![Action::Move { name: "A1".into(), group, folder: Some(folder), index: Some(index) }]);
    let names = |app: &PhotocraftApp, g: &str| app.session.tools.presets.iter().filter(|x| x.group == g).map(|x| x.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&app, "Alpha"), ["A2", "A3", "A1"]);
    apply(&mut app, vec![Action::Move { name: "A2".into(), group: "Beta".into(), folder: None, index: None }]);
    assert_eq!(names(&app, "Beta"), ["B1", "A2"]);
    apply(&mut app, vec![Action::MoveGroup { group: "Beta".into(), folder: vec![], before: Some("Alpha".into()) }]);
    let order = photocraft_engine::brush_preset_cmds::group_order(&app.session.tools.presets);
    assert!(order.iter().position(|g| g == "Beta") < order.iter().position(|g| g == "Alpha"));
    // Rename and delete from the context menu.
    apply(&mut app, vec![Action::Rename(Renaming { group: false, name: "A3".into(), folder: vec![], text: "Third".into() })]);
    assert!(paint::presets::find(&app.session.tools.presets, "Third").is_some());
    assert!(app.ui.brushes_panel.renaming.is_none());
    apply(&mut app, vec![Action::Rename(Renaming { group: true, name: "Beta".into(), folder: vec![], text: "Bees".into() })]);
    assert_eq!(names(&app, "Bees"), ["B1", "A2"]);
    apply(&mut app, vec![Action::Delete("Third".into()), Action::DeleteGroup { group: "Bees".into(), folder: vec![] }]);
    assert_eq!(names(&app, "Alpha"), ["A1"]);
    assert!(names(&app, "Bees").is_empty());
    // Every action went through a journaled command.
    let ids: Vec<&str> = app.session.journal.iter().map(|(id, _)| id.as_str()).collect();
    for id in [
        "brush.presets.move",
        "brush.presets.moveGroup",
        "brush.presets.rename",
        "brush.presets.renameGroup",
        "brush.presets.delete",
        "brush.presets.deleteGroup",
    ] {
        assert!(ids.contains(&id), "{id}: {ids:?}");
    }
}

fn harness(tab: usize, section: usize) -> Harness<'static, PhotocraftApp> {
    let mut app = app();
    app.ui.panels.brush_settings = true;
    app.ui.brush_tab = tab;
    app.ui.brush_section = section;
    let mut h = Harness::builder().with_size(vec2(1200.0, 900.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            window(app, &ctx);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

fn toggle_value(app: &PhotocraftApp, section: usize) -> bool {
    let brush = serde_json::to_value(&app.session.tools.brush).unwrap();
    let path = match section {
        8 => "/noise",
        9 => "/wetEdges",
        10 => "/buildUp",
        11 => "/smoothing/enabled",
        12 => "/protectTexture",
        _ => panic!("not a checkbox-only row"),
    };
    brush.pointer(path).and_then(Value::as_bool).expect("the checkbox has a boolean brush setting")
}

#[test]
fn hue_jitter_amount_edits_keep_percentage_units_and_command_routing() {
    use egui::accesskit::Role;
    use egui_kittest::kittest::{NodeT, Queryable};

    let mut h = harness(0, 5);
    h.state_mut()
        .session
        .execute(
            "tools.setBrush",
            json!({"brush": {"colorDynamics": {
                "enabled": true, "hueJitter": 0.37, "saturationJitter": 0.23, "brightnessJitter": 0.41, "purity": -0.16
            }}}),
        )
        .unwrap();
    let before = h.state().session.tools.brush.clone();
    h.run_steps(2);
    let field = h.query_all_by_role(Role::SpinButton).find(|n| n.accesskit_node().numeric_value() == Some(37.0)).unwrap();
    let center = field.rect().center();
    field.click();
    h.key_press(egui::Key::ArrowUp);
    h.run_steps(2);
    assert_eq!(h.state().session.tools.brush.color_dynamics.hue_jitter, 0.38);
    h.key_press(egui::Key::Enter);
    // Click near the high end of the track, then beyond its right edge while dragging.
    let from = egui::pos2(center.x, center.y + 24.0);
    h.event(egui::Event::PointerMoved(from));
    h.event(egui::Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
    h.run_steps(1);
    let to = from + vec2(200.0, 0.0);
    h.event(egui::Event::PointerMoved(to));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
    h.run_steps(2);
    let mut expected = before;
    expected.color_dynamics.hue_jitter = 1.0;
    assert_eq!(h.state().session.tools.brush, expected, "the amount clamps at 100% and preserves neighboring settings");
    assert_eq!(last_journal(h.state()).unwrap().0, "tools.setBrush");
    let mut replay = photocraft_engine::Session::new();
    for (id, params) in h.state().session.journal.clone() {
        replay.execute(&id, params).unwrap();
    }
    assert_eq!(replay.tools.brush, expected, "the edit remains replayable through the engine");
}

#[test]
fn checkbox_only_rows_toggle_without_changing_the_selected_controls() {
    use egui_kittest::kittest::Queryable;
    for (section, &(name, _)) in SECTIONS.iter().enumerate().skip(8) {
        let mut h = harness(0, 1);
        let before = toggle_value(h.state(), section);
        let n = h.state().session.journal.len();
        h.get_by_label(name).click();
        h.run_steps(3);
        assert_eq!(h.state().ui.brush_section, 1, "{name} must leave Shape Dynamics open");
        assert_eq!(toggle_value(h.state(), section), !before, "{name}");
        assert_eq!(h.state().session.journal.len(), n + 1, "one command for {name}");
        assert_eq!(last_journal(h.state()).unwrap().0, "tools.setBrush");
        h.get_by_label(name).click();
        h.run_steps(3);
        assert_eq!(toggle_value(h.state(), section), before, "a second row click turns {name} back");
        assert_eq!(h.state().ui.brush_section, 1);
        assert_eq!(h.state().session.journal.len(), n + 2);
    }
}

#[test]
fn checkbox_only_box_and_lock_clicks_do_not_select_or_double_toggle() {
    use egui_kittest::kittest::Queryable;
    for (section, &(name, _)) in SECTIONS.iter().enumerate().skip(8) {
        let mut h = harness(0, 1);
        let rect = h.get_by_label(name).rect();
        let before = toggle_value(h.state(), section);
        let n = h.state().session.journal.len();
        let at = egui::pos2(rect.left() + 12.0, rect.center().y);
        h.hover_at(at);
        h.run_steps(2);
        h.drag_at(at);
        h.run_steps(2);
        h.drop_at(at);
        h.run_steps(3);
        assert_eq!(toggle_value(h.state(), section), !before, "checkbox click on {name}");
        assert_eq!(h.state().ui.brush_section, 1);
        assert_eq!(h.state().session.journal.len(), n + 1);
        let at = egui::pos2(rect.right() - 11.0, rect.center().y);
        h.hover_at(at);
        h.run_steps(2);
        h.drag_at(at);
        h.run_steps(2);
        h.drop_at(at);
        h.run_steps(3);
        assert_eq!(toggle_value(h.state(), section), !before, "lock click must not toggle {name}");
        assert_eq!(h.state().ui.brush_section, 1);
        assert_eq!(h.state().session.journal.len(), n + 2);
        let mut brush = h.state().session.tools.brush.clone();
        assert!(*section_lock(&mut brush, section).unwrap(), "{name} lock");
    }
}

#[test]
fn checkbox_only_rows_cannot_restore_an_empty_properties_pane() {
    for (section, &(name, _)) in SECTIONS.iter().enumerate().skip(8) {
        let h = harness(0, section);
        assert_eq!(h.state().ui.brush_section, 0, "an old selection of {name} returns to Brush Tip Shape");
    }
}

#[test]
fn smoothing_row_preserves_its_amount_and_options_across_toggles() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness(0, 1);
    h.state_mut().run("tools.setBrush", json!({"brush": {"smoothing": {"amount": 0.65, "pulledString": true, "catchUpOnEnd": false}}})).unwrap();
    let before = h.state().session.tools.brush.smoothing.clone();
    h.run_steps(2);
    for enabled in [false, true] {
        h.get_by_label("Smoothing").click();
        h.run_steps(3);
        let mut expected = before.clone();
        expected.enabled = enabled;
        assert_eq!(h.state().session.tools.brush.smoothing, expected);
        assert_eq!(h.state().ui.brush_section, 1);
        assert_eq!(last_journal(h.state()).unwrap().1["brush"], json!({"smoothing": {"enabled": enabled}}));
    }
}

#[test]
fn checkbox_only_rows_are_accessible_checkboxes_and_accept_space() {
    use egui::accesskit::Role;
    use egui_kittest::kittest::{NodeT, Queryable};
    let mut h = harness(0, 1);
    for (section, &(name, _)) in SECTIONS.iter().enumerate().skip(8) {
        let node = h.get_by_label(name);
        assert_eq!(node.accesskit_node().role(), Role::CheckBox, "{name}");
        node.focus();
        h.run_steps(2);
        let before = toggle_value(h.state(), section);
        h.key_press(egui::Key::Space);
        h.run_steps(3);
        assert_eq!(toggle_value(h.state(), section), !before, "Space toggles {name}");
        assert_eq!(h.state().ui.brush_section, 1);
    }
}

#[test]
fn editing_the_options_bar_amount_reenables_smoothing() {
    use egui_kittest::kittest::Queryable;
    let mut h = options_bar_harness(crate::state::Tool::Brush);
    h.state_mut().run("tools.setBrush", json!({"brush": {"smoothing": {"enabled": false, "amount": 0.4}}})).unwrap();
    h.run_steps(3);
    let n = h.state().session.journal.len();
    let label = h.get_by_label("Smoothing").rect();
    let at = egui::pos2(label.right() + 25.0, label.center().y);
    h.hover_at(at);
    h.run_steps(2);
    h.drag_at(at);
    h.run_steps(2);
    h.hover_at(at + vec2(20.0, 0.0));
    h.run_steps(2);
    h.drop_at(at + vec2(20.0, 0.0));
    h.run_steps(3);
    let smoothing = &h.state().session.tools.brush.smoothing;
    assert!(smoothing.enabled && smoothing.amount != 0.4);
    assert_eq!(h.state().session.journal.len(), n + 1, "one options-bar gesture");
    assert_eq!(last_journal(h.state()).unwrap().1["brush"]["smoothing"]["enabled"], true);
}

/// #4282: every name in the section list starts in the same column, Brush Tip Shape included,
/// whether or not its row has an enable box (Photoshop indents the rows without one).
#[test]
fn section_names_line_up_whether_or_not_they_have_a_box() {
    fn text_x(shape: &egui::Shape, out: &mut Vec<(String, f32)>) {
        match shape {
            egui::Shape::Text(text) => out.push((text.galley.job.text.clone(), text.pos.x)),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| text_x(s, out)),
            _ => {}
        }
    }
    // Shape Dynamics is shown, so its heading paints too; the list's copy is the leftmost.
    let h = harness(0, 1);
    let mut painted = Vec::new();
    for shape in &h.output().shapes {
        text_x(&shape.shape, &mut painted);
    }
    let left = |name: &str| painted.iter().filter(|(t, _)| t == name).map(|(_, x)| *x).reduce(f32::min);
    let columns: Vec<(&str, f32)> = SECTIONS.iter().map(|(name, _)| (*name, left(name).unwrap_or_else(|| panic!("{name} painted")))).collect();
    let (_, boxed) = columns[1];
    for (name, x) in &columns {
        assert!((x - boxed).abs() < 0.5, "{name} starts at {x}, the boxed names at {boxed}: {columns:?}");
    }
}

#[test]
fn preview_strip_re_renders_only_when_the_brush_changes() {
    let mut h = harness(0, 0);
    let ctx = h.ctx.clone();
    let n0 = brush_preview::render_count(&ctx);
    assert!(n0 >= 1, "the strip rendered");
    h.run_steps(6);
    assert_eq!(brush_preview::render_count(&ctx), n0, "idle frames re-rendered a preview");
    // Change the brush through the engine: the strip renders once more, then stays cached.
    h.state_mut().run("tools.setBrush", json!({"brush": {"scattering": {"enabled": true, "scatter": {"jitter": 2.0}}}})).unwrap();
    h.run_steps(5);
    assert_eq!(brush_preview::render_count(&ctx), n0 + 1);
    // Colour and the jitter seed of strokes aren't preview inputs: no re-render.
    h.state_mut().session.tools.brush.color = [1.0, 0.0, 0.0, 1.0];
    h.run_steps(3);
    assert_eq!(brush_preview::render_count(&ctx), n0 + 1);
}

#[test]
fn every_section_draws_without_issuing_commands() {
    for tab in 0..3 {
        for section in 0..SECTIONS.len() {
            // Tab 2: the Brushes tab in grid view.
            let mut h = harness(tab.min(1), section);
            if tab == 2 {
                h.state_mut().ui.brushes_panel.view = BrushesView::Grid;
            }
            // A brush with every section on and a sampled tip.
            let mut b = BrushSettings::default();
            for i in 0..SECTIONS.len() {
                edit_section(i, &mut b);
            }
            b.tip = TipShape::Sampled(GrayTile::from_fn(50, 40, |x, y| ((x ^ y) & 1) as f32));
            h.state_mut().session.tools.brush = b.clone();
            let n = h.state().session.journal.len();
            h.run_steps(4);
            // Merely showing the panel must not rewrite the brush (no rounding drift, no commands).
            assert_eq!(h.state().session.journal.len(), n, "tab {tab} section {section}");
            assert_eq!(h.state().session.tools.brush, b, "tab {tab} section {section}");
        }
    }
}

#[test]
fn brushes_panel_groups_collapse_and_filter() {
    let mut h = harness(1, 0);
    let groups = grouped_presets(&h.state().session.tools.presets);
    assert!(groups.len() >= 2, "{groups:?}");
    let ctx = h.ctx.clone();
    let shown = |h: &mut Harness<'static, PhotocraftApp>| {
        h.run_steps(2);
        brush_preview::with_cache(&ctx, |c| c.len())
    };
    let all = shown(&mut h);
    // Collapse every group: no more previews are needed, and the old ones are kept (cached).
    h.state_mut().ui.brushes_panel.collapsed = groups.iter().map(|(g, _)| g.clone()).collect();
    let n = brush_preview::render_count(&ctx);
    assert_eq!(shown(&mut h), all);
    assert_eq!(brush_preview::render_count(&ctx), n);
    // A filter shows matching presets even in collapsed groups.
    let first = h.state().session.tools.presets[0].name.clone();
    h.state_mut().ui.brushes_panel.filter = first.to_uppercase();
    h.run_steps(2);
    assert!(brush_preview::render_count(&ctx) >= n);
}

fn kit(name: &str, folder: &[&str]) -> paint::BrushPreset {
    paint::BrushPreset {
        name: name.into(),
        brush: Default::default(),
        builtin: false,
        group: "Kit".into(),
        folder: folder.iter().map(|s| s.to_string()).collect(),
        captures: Default::default(),
    }
}

#[test]
fn folder_tree_nests_presets_in_panel_order() {
    use crate::brushes_tab::{MAX_VIEW_DEPTH, Node, folder_tree};
    let deep: Vec<&str> = vec!["d"; MAX_VIEW_DEPTH + 5];
    let presets = [kit("Top", &[]), kit("Pen", &["Inks"]), kit("Chalk", &["Dry"]), kit("Nib", &["Inks", "Fine"]), kit("Quill", &["Inks"]), kit("Deep", &deep)];
    let items: Vec<&paint::BrushPreset> = presets.iter().collect();
    let tree = folder_tree(&items);
    fn show(nodes: &[Node], out: &mut Vec<String>) {
        for n in nodes {
            match n {
                Node::Preset(p) => out.push(p.name.clone()),
                Node::Folder(f) => {
                    out.push(format!("[{} {}]", f.path.join("/"), f.count));
                    show(&f.children, out);
                    out.push("[end]".into());
                }
            }
        }
    }
    let mut out = Vec::new();
    show(&tree, &mut out);
    let deep_path = vec!["d"; MAX_VIEW_DEPTH].join("/");
    assert_eq!(out[..12], ["Top", "[Inks 3]", "Pen", "[Inks/Fine 1]", "Nib", "[end]", "Quill", "[end]", "[Dry 1]", "Chalk", "[end]", "[d 1]"]);
    assert!(out.contains(&format!("[{deep_path} 1]")), "paths are cut at the view depth");
    assert_eq!(out.iter().filter(|s| s.starts_with("[d")).count(), MAX_VIEW_DEPTH);
}

/// Nested folders show under their group, open and close, and their context menu renames and
/// deletes them through the group commands (#1851).
#[test]
fn brushes_panel_shows_nested_folders() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness(1, 0);
    let builtin_groups: Vec<String> = grouped_presets(&h.state().session.tools.presets).into_iter().map(|(g, _)| g).collect();
    h.state_mut().ui.brushes_panel.collapsed = builtin_groups;
    for p in [kit("Kit Top", &[]), kit("Kit Pen", &["Test Inks"]), kit("Kit Nib", &["Test Inks", "Test Fine"]), kit("Kit Chalk", &["Test Dry"])] {
        h.state_mut().session.tools.presets.push(p);
    }
    h.run_steps(3);
    for label in ["Kit", "Test Inks", "Test Fine", "Test Dry", "Kit Nib", "Kit Chalk"] {
        assert!(h.query_by_label(label).is_some(), "{label} is shown");
    }
    // Closing a folder hides what it holds, and only that.
    h.get_by_label("Test Inks").click();
    h.run_steps(3);
    assert_eq!(h.state().ui.brushes_panel.collapsed.last().map(String::as_str), Some("Kit/Test Inks"));
    assert!(h.query_by_label("Kit Nib").is_none() && h.query_by_label("Test Fine").is_none());
    assert!(h.query_by_label("Kit Chalk").is_some());
    // Rename a nested folder from its context menu.
    h.get_by_label("Test Dry").click_secondary();
    h.run_steps(2);
    h.get_by_label("Rename Group…").click();
    h.run_steps(2);
    assert_eq!(h.state().ui.brushes_panel.renaming.as_ref().map(|r| r.folder.clone()), Some(vec!["Test Dry".to_string()]));
    h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
    h.event(egui::Event::Text("Test Wet".into()));
    h.run_steps(1);
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert_eq!(
        last_journal(h.state()).unwrap(),
        ("brush.presets.renameGroup".to_string(), json!({ "group": "Kit", "folder": ["Test Dry"], "newName": "Test Wet" }))
    );
    assert_eq!(paint::presets::find(&h.state().session.tools.presets, "Kit Chalk").unwrap().folder, ["Test Wet"]);
    // Delete it: only its presets go.
    h.get_by_label("Test Wet").click_secondary();
    h.run_steps(2);
    h.get_by_label("Delete Group").click();
    h.run_steps(3);
    assert_eq!(last_journal(h.state()).unwrap(), ("brush.presets.deleteGroup".to_string(), json!({ "group": "Kit", "folder": ["Test Wet"] })));
    let kit_left: Vec<String> = h.state().session.tools.presets.iter().filter(|p| p.group == "Kit").map(|p| p.name.clone()).collect();
    assert_eq!(kit_left, ["Kit Top", "Kit Pen", "Kit Nib"]);
}

#[test]
fn folder_actions_become_folder_commands() {
    use crate::brushes_tab::{Action, apply, drop_target};
    let mut app = app();
    for p in [kit("Kit Top", &[]), kit("Kit Pen", &["Inks"]), kit("Kit Nib", &["Inks"]), kit("Kit Chalk", &["Dry"])] {
        app.session.tools.presets.push(p);
    }
    // Dropping a preset on one inside a folder moves it into that folder.
    let (group, folder, index) = drop_target(&app.session.tools.presets, "Kit Top", "Kit Nib", false).unwrap();
    assert_eq!((group.as_str(), folder.clone(), index), ("Kit", vec!["Inks".to_string()], 1));
    apply(&mut app, vec![Action::Move { name: "Kit Top".into(), group, folder: Some(folder), index: Some(index) }]);
    assert_eq!(paint::presets::find(&app.session.tools.presets, "Kit Top").unwrap().folder, ["Inks"]);
    // Dropping on a folder header appends to the folder; a folder header dropped on a sibling's moves it.
    apply(&mut app, vec![Action::Move { name: "Kit Pen".into(), group: "Kit".into(), folder: Some(vec!["Dry".into()]), index: None }]);
    assert_eq!(paint::presets::find(&app.session.tools.presets, "Kit Pen").unwrap().folder, ["Dry"]);
    apply(&mut app, vec![Action::MoveGroup { group: "Kit".into(), folder: vec!["Dry".into()], before: Some("Inks".into()) }]);
    assert_eq!(photocraft_engine::brush_preset_cmds::subfolder_order(&app.session.tools.presets, "Kit", &[]), ["Dry", "Inks"]);
    assert_eq!(last_journal(&app).unwrap(), ("brush.presets.moveGroup".to_string(), json!({ "group": "Kit", "folder": ["Dry"], "before": "Inks" })));
}

/// Dragging a folder or group header onto another header (#2014): the upper edge places it
/// before, the rest puts it inside; the action runs `brush.presets.moveGroup` with `into`.
#[test]
fn folder_drops_map_to_move_commands() {
    use crate::brushes_tab::{Action, apply, folder_drop};
    let v = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<String>>();
    // Into: a folder onto another folder, a group onto a group.
    assert_eq!(
        folder_drop("Kit", &v(&["Dry"]), "Kit", &v(&["Inks"]), false),
        Some(Action::MoveInto { group: "Kit".into(), folder: v(&["Dry"]), into: Some(("Kit".into(), v(&["Inks"]))), before: None })
    );
    assert_eq!(
        folder_drop("Pack", &[], "Kit", &v(&["Inks"]), false),
        Some(Action::MoveInto { group: "Pack".into(), folder: vec![], into: Some(("Kit".into(), v(&["Inks"]))), before: None })
    );
    // Onto itself, into its own folder, or into the folder it is already in: nothing.
    assert_eq!(folder_drop("Kit", &v(&["Inks"]), "Kit", &v(&["Inks"]), false), None);
    assert_eq!(folder_drop("Kit", &v(&["Inks"]), "Kit", &v(&["Inks", "Fine"]), true), None);
    assert_eq!(folder_drop("Kit", &[], "Kit", &v(&["Inks"]), false), None);
    assert_eq!(folder_drop("Kit", &v(&["Inks", "Fine"]), "Kit", &v(&["Inks"]), false), None);
    // Upper edge: siblings reorder; a nested folder before a group goes to the top level; a
    // folder before a folder elsewhere moves next to it.
    assert_eq!(
        folder_drop("Kit", &v(&["Dry"]), "Kit", &v(&["Inks"]), true),
        Some(Action::MoveGroup { group: "Kit".into(), folder: v(&["Dry"]), before: Some("Inks".into()) })
    );
    assert_eq!(folder_drop("Pack", &[], "Kit", &[], true), Some(Action::MoveGroup { group: "Pack".into(), folder: vec![], before: Some("Kit".into()) }));
    assert_eq!(
        folder_drop("Kit", &v(&["Inks", "Fine"]), "Pack", &[], true),
        Some(Action::MoveInto { group: "Kit".into(), folder: v(&["Inks", "Fine"]), into: None, before: Some("Pack".into()) })
    );
    assert_eq!(
        folder_drop("Kit", &v(&["Inks", "Fine"]), "Pack", &v(&["Box"]), true),
        Some(Action::MoveInto { group: "Kit".into(), folder: v(&["Inks", "Fine"]), into: Some(("Pack".into(), vec![])), before: Some("Box".into()) })
    );

    let mut app = app();
    for p in [kit("Kit Pen", &["Inks"]), kit("Kit Nib", &["Inks", "Fine"]), kit("Kit Chalk", &["Dry"])] {
        app.session.tools.presets.push(p);
    }
    let into = folder_drop("Kit", &v(&["Dry"]), "Kit", &v(&["Inks", "Fine"]), false).unwrap();
    apply(&mut app, vec![into]);
    assert_eq!(paint::presets::find(&app.session.tools.presets, "Kit Chalk").unwrap().folder, ["Inks", "Fine", "Dry"]);
    assert_eq!(
        last_journal(&app).unwrap(),
        ("brush.presets.moveGroup".to_string(), json!({ "group": "Kit", "folder": ["Dry"], "into": {"group": "Kit", "folder": ["Inks", "Fine"]} }))
    );
    // Out to the top level, before a group.
    let out = folder_drop("Kit", &v(&["Inks", "Fine"]), "Kit", &[], true).unwrap();
    apply(&mut app, vec![out]);
    let chalk = paint::presets::find(&app.session.tools.presets, "Kit Chalk").unwrap();
    assert_eq!((chalk.group.as_str(), chalk.folder.clone()), ("Fine", v(&["Dry"])));
    let order = photocraft_engine::brush_preset_cmds::group_order(&app.session.tools.presets);
    assert!(order.iter().position(|g| g == "Fine") < order.iter().position(|g| g == "Kit"));
}

/// A folder header dragged onto another folder's header in the Brushes panel goes inside it.
#[test]
fn dragging_a_folder_header_onto_another_moves_it_inside() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness(1, 0);
    let builtin_groups: Vec<String> = grouped_presets(&h.state().session.tools.presets).into_iter().map(|(g, _)| g).collect();
    h.state_mut().ui.brushes_panel.collapsed = builtin_groups;
    for p in [kit("Kit Pen", &["Test Inks"]), kit("Kit Chalk", &["Test Dry"])] {
        h.state_mut().session.tools.presets.push(p);
    }
    h.run_steps(3);
    let from = h.get_by_label("Test Dry").rect().center();
    let to = h.get_by_label("Test Inks").rect();
    let to = egui::pos2(to.center().x, to.top() + to.height() * 0.7);
    h.hover_at(from);
    h.run();
    h.drag_at(from);
    h.run();
    for k in 1..=6 {
        h.hover_at(from + (to - from) * (k as f32 / 6.0));
        h.run();
    }
    h.drop_at(to);
    h.run_steps(3);
    assert_eq!(paint::presets::find(&h.state().session.tools.presets, "Kit Chalk").unwrap().folder, ["Test Inks", "Test Dry"]);
    assert_eq!(last_journal(h.state()).unwrap().0, "brush.presets.moveGroup");
}

/// A fresh session with the Brush tool: shortcuts, the options bar and the Brush Settings window,
/// in the default theme (#258).
fn app_harness() -> Harness<'static, PhotocraftApp> {
    app_harness_with_language(crate::i18n::Lang::EN)
}

fn app_harness_with_language(lang: crate::i18n::Lang) -> Harness<'static, PhotocraftApp> {
    let mut app = app();
    app.ui.tool = crate::state::Tool::Brush;
    let mut h = Harness::builder().with_size(vec2(1400.0, 900.0)).build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            let previous = crate::i18n::current();
            crate::i18n::set_current(lang);
            crate::shortcuts::handle(app, &ctx);
            crate::panels::options_bar(app, ui);
            window(app, &ctx);
            crate::paint_mouse::show_picker(app, &ctx);
            crate::i18n::set_current(previous);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
    h.run_steps(4);
    h
}

#[test]
fn brush_sections_paint_simplified_chinese_labels() {
    fn collect_text(shape: &egui::Shape, out: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(text) => out.push(text.galley.job.text.clone()),
            egui::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_text(shape, out);
                }
            }
            _ => {}
        }
    }
    let zh = crate::i18n::Lang::from_code("zh-hans").unwrap();
    let mut h = app_harness_with_language(zh);
    h.key_press(egui::Key::F5);
    h.run_steps(3);
    let mut painted = Vec::new();
    for shape in &h.output().shapes {
        collect_text(&shape.shape, &mut painted);
    }
    for (name, _) in SECTIONS {
        let translated = crate::i18n::tr(zh, name);
        assert!(painted.iter().any(|s| s == translated), "{name} must paint as {translated}");
        assert!(!painted.iter().any(|s| s == name), "{name} must not paint in English");
    }
}

/// #3477: the options pane has no title of its own. The selected row in the section list is the
/// only place the section's name is painted.
#[test]
fn options_pane_does_not_repeat_the_selected_sections_name() {
    fn collect_text(shape: &egui::Shape, out: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(text) => out.push(text.galley.job.text.clone()),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect_text(s, out)),
            _ => {}
        }
    }
    for (section, (name, _)) in SECTIONS.iter().enumerate().take(CHECKBOX_ONLY_START) {
        let h = harness(0, section);
        let mut painted = Vec::new();
        for shape in &h.output().shapes {
            collect_text(&shape.shape, &mut painted);
        }
        assert_eq!(painted.iter().filter(|s| s.as_str() == *name).count(), 1, "{name} is painted once, in the section list");
    }
}

#[test]
fn f5_edits_the_default_brush_with_every_section() {
    use egui_kittest::kittest::Queryable;
    let mut h = app_harness();
    assert!(!h.state().ui.panels.brush_settings);
    let fresh = h.state().session.tools.brush.clone();
    assert!(!fresh.shape_dynamics.enabled && !fresh.transfer.enabled, "the default round brush has no dynamics yet");
    h.key_press(egui::Key::F5);
    h.run_steps(3);
    assert!(h.state().ui.panels.brush_settings && h.state().ui.brush_tab == 0, "F5 shows Brush Settings");
    // Every section is there for the built-in round brush, not only for sampled or imported ones.
    for (name, _) in SECTIONS {
        assert!(h.query_all_by_label(name).next().is_some(), "{name} listed");
    }
    // Clicking a section's name shows it and turns it on: one tools.setBrush.
    let n = h.state().session.journal.len();
    h.get_by_label("Shape Dynamics").click();
    h.run_steps(3);
    assert_eq!(h.state().ui.brush_section, 1);
    assert!(h.state().session.tools.brush.shape_dynamics.enabled);
    assert_eq!(h.state().session.journal.len(), n + 1);
    let (id, p) = last_journal(h.state()).unwrap();
    assert_eq!(id, "tools.setBrush");
    assert_eq!(p["brush"]["shapeDynamics"]["enabled"], json!(true));
    // A section's box toggles it without showing it.
    h.get_by_label("Enable Transfer").click();
    h.run_steps(3);
    assert!(h.state().session.tools.brush.transfer.enabled);
    assert_eq!(h.state().ui.brush_section, 1);
    assert_eq!(h.state().session.journal.len(), n + 2);
    // The rest of the brush is untouched.
    let b = &h.state().session.tools.brush;
    assert_eq!((b.size, b.hardness, &b.tip), (fresh.size, fresh.hardness, &fresh.tip));
    // F5 again hides the panel.
    h.key_press(egui::Key::F5);
    h.run_steps(2);
    assert!(!h.state().ui.panels.brush_settings);
}

#[test]
fn update_current_brush_button_overwrites_the_picked_preset_in_place() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness(0, 0);
    // Nothing picked yet: the button is disabled and clicking it does nothing.
    assert!(h.state().session.tools.current_preset.is_none());
    h.get_by_label("Update the current brush with these settings").click();
    h.run_steps(3);
    assert!(!h.state().session.journal.iter().any(|(id, _)| id == "brush.presets.update"));
    // Pick a preset, then turn a section on.
    h.state_mut().run("tools.setBrush", json!({ "preset": "Chalk" })).unwrap();
    h.run_steps(2);
    let n = h.state().session.tools.presets.len();
    let at = h.state().session.tools.presets.iter().position(|p| p.name == "Chalk").unwrap();
    h.get_by_label("Enable Scattering").click();
    h.run_steps(3);
    assert!(h.state().session.tools.brush.scattering.enabled);
    // The update button overwrites the picked preset in place: no new preset, position kept.
    h.get_by_label("Update the current brush with these settings").click();
    h.run_steps(3);
    assert!(h.state().session.journal.iter().any(|(id, p)| id == "brush.presets.update" && p["brush"]["scattering"]["enabled"] == json!(true)));
    assert_eq!(h.state().session.tools.presets.len(), n);
    let chalk = h.state().session.tools.presets.get(at).unwrap();
    assert_eq!(chalk.name, "Chalk");
    assert!(chalk.brush.scattering.enabled);
    assert!(!chalk.builtin, "an updated built-in becomes the user's preset");
}

/// Each painting tool keeps its own brush (#218): the preset picked for one tool is not the
/// current preset of the next, so Update can't overwrite it with the other tool's brush.
#[test]
fn switching_tools_forgets_the_picked_preset() {
    let mut app = app();
    app.ui.tool = crate::state::Tool::Brush;
    crate::paint_mouse::sync_tool_brush(&mut app);
    app.run("tools.setBrush", json!({ "preset": "Chalk" })).unwrap();
    assert_eq!(app.session.tools.current_preset.as_deref(), Some("Chalk"));
    app.ui.tool = crate::state::Tool::Eraser;
    crate::paint_mouse::sync_tool_brush(&mut app);
    assert_eq!(app.session.tools.current_preset, None);
    assert!(app.run("brush.presets.update", json!({})).is_err());
}

#[test]
fn options_bar_reaches_brush_settings_and_the_preset_library() {
    use egui_kittest::kittest::Queryable;
    let mut h = app_harness();
    // Photoshop's "Toggle the Brush Settings panel" button beside the brush chip.
    h.get_by_label("Toggle the Brush Settings panel").click();
    h.run_steps(3);
    assert!(h.state().ui.panels.brush_settings);
    h.get_by_label("Toggle the Brush Settings panel").click();
    h.run_steps(3);
    assert!(!h.state().ui.panels.brush_settings);
    // The chip opens the same picker a right-click on the canvas does, below the chip; it lists
    // the session's presets in their groups, not a fixed set of tips.
    h.get_by_label("Brush Preset picker").click();
    h.run_steps(3);
    let chip = h.get_by_label("Brush Preset picker").rect();
    let at = h.state().ui.brush_picker.expect("the chip opened the picker");
    assert!((at[0] - chip.left()).abs() < 1.0 && at[1] >= chip.bottom(), "{at:?} below {chip:?}");
    let presets = h.state().session.tools.presets.clone();
    for (group, _) in grouped_presets(&presets) {
        assert!(h.query_by_label(&group).is_some(), "group {group}");
    }
    let target = presets.iter().find(|p| p.group == "Dry Media").or(presets.last()).unwrap().name.clone();
    assert_ne!(h.state().session.tools.current_preset.as_deref(), Some(target.as_str()));
    // The picker's cards are tall: scroll the target into the list before clicking it.
    h.get_by_label(&target).scroll_to_me();
    h.run_steps(3);
    h.get_by_label(&target).click();
    h.run_steps(3);
    assert_eq!(last_journal(h.state()).unwrap(), ("tools.setBrush".to_string(), json!({ "preset": target })));
    assert_eq!(h.state().session.tools.current_preset.as_deref(), Some(target.as_str()));
    // A click picks and keeps the picker open; the chip's click closes it again.
    assert!(h.state().ui.brush_picker.is_some());
    h.get_by_label("Brush Preset picker").click();
    h.run_steps(3);
    assert_eq!(h.state().ui.brush_picker, None, "the chip toggles the picker");
    // Its Brush Settings button opens the full editor and closes the picker.
    h.get_by_label("Brush Preset picker").click();
    h.run_steps(3);
    h.get_by_label("Brush Settings…").click();
    h.run_steps(3);
    assert!(h.state().ui.panels.brush_settings && h.state().ui.brush_tab == 0);
    assert_eq!(h.state().ui.brush_picker, None);
}

/// A preset's context menu (the Brushes tab's and the picker's) renames it in the rename bar.
#[test]
fn context_menu_renames_a_preset() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness(1, 0);
    let name = h.state().session.tools.presets[0].name.clone();
    h.get_by_label(&name).click_secondary();
    h.run_steps(2);
    h.get_by_label("Rename Brush…").click();
    h.run_steps(2);
    assert_eq!(h.state().ui.brushes_panel.renaming.as_ref().map(|r| r.name.clone()), Some(name.clone()), "the rename bar opened");
    h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
    h.event(egui::Event::Text("Renamed".into()));
    h.run_steps(1);
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert!(paint::presets::find(&h.state().session.tools.presets, "Renamed").is_some());
    assert!(h.state().ui.brushes_panel.renaming.is_none());
    assert_eq!(last_journal(h.state()).unwrap(), ("brush.presets.rename".to_string(), json!({ "name": name, "newName": "Renamed" })));
}

#[test]
fn hand_tool_offers_scale_buttons_and_requests_the_right_view() {
    use egui_kittest::kittest::Queryable;

    let mut app = app();
    app.run("file.new", json!({"width": 400, "height": 200})).unwrap();
    app.sync_views();
    app.ui.tool = crate::state::Tool::Hand;
    app.ui.views[0].zoom = 2.0;
    let mut h = Harness::builder().with_size(vec2(1400.0, 60.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                crate::panels::options_bar(app, ui);
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
    h.run_steps(4);

    h.get_by_label("100%").click();
    h.run_steps(1);
    assert_eq!(h.state().ui.views[0].zoom, 1.0);

    h.get_by_label("Fit Screen").click();
    h.run_steps(1);
    assert!(h.state().ui.views[0].fit_pending);
    assert!(!h.state().ui.views[0].fill_pending);

    h.get_by_label("Fill Screen").click();
    h.run_steps(1);
    assert!(h.state().ui.views[0].fill_pending);
    assert!(!h.state().ui.views[0].fit_pending);
}

#[test]
fn brush_settings_hamburger_menu_actions() {
    use egui_kittest::kittest::Queryable;

    // 1. Menu opens and offers Photoshop actions
    let mut h = harness(0, 0);
    h.get_by_label("Panel menu").click();
    h.run_steps(2);

    assert!(h.query_all_by_label("New Brush Preset…").count() > 0);
    assert!(h.query_all_by_label("Clear Brush Controls").count() > 0);
    assert!(h.query_all_by_label("Reset All Locked Settings").count() > 0);
    assert!(h.query_all_by_label("Copy Texture to Other Tools").count() > 0);
    assert!(h.query_all_by_label("Close").count() > 0);
    assert!(h.query_all_by_label("Close Tab Group").count() > 0);

    // 2. New Brush Preset… saves a preset
    let count0 = h.state().session.tools.presets.len();
    h.get_by_label("New Brush Preset…").click();
    h.run_steps(2);
    assert_eq!(h.state().session.tools.presets.len(), count0 + 1);
    assert_eq!(last_journal(h.state()).unwrap().0, "brush.presets.save");

    // 3. Clear Brush Controls resets dynamics, keeps locked sections and smoothing
    let mut h = harness(0, 0);
    h.state_mut().session.tools.brush.size = 45.0;
    h.state_mut().session.tools.brush.scattering.enabled = true;
    h.state_mut().session.tools.brush.texture.enabled = true;
    h.state_mut().session.tools.brush.locks.texture = true;
    h.state_mut().session.tools.brush.smoothing.amount = 0.5;
    h.run_steps(1);

    h.get_by_label("Panel menu").click();
    h.run_steps(2);
    h.get_by_label("Clear Brush Controls").click();
    h.run_steps(2);

    let b = &h.state().session.tools.brush;
    assert!(!b.scattering.enabled, "unlocked scattering cleared");
    assert!(b.texture.enabled, "locked texture preserved");
    assert_eq!(b.smoothing.amount, 0.5, "smoothing preserved");
    assert_eq!(b.size, 45.0, "tip size preserved");

    // 4. Reset All Locked Settings clears all locks
    assert!(b.locks.texture);
    h.get_by_label("Panel menu").click();
    h.run_steps(2);
    h.get_by_label("Reset All Locked Settings").click();
    h.run_steps(2);
    assert_eq!(h.state().session.tools.brush.locks, photocraft_paint::brush::SectionLocks::default());

    // 5. Copy Texture to Other Tools copies texture to other tools with texture enabled
    let mut h = harness(0, 0);
    h.state_mut().session.tools.brush.texture.enabled = true;
    h.state_mut().session.tools.brush.texture.scale = 2.5;
    let mut pencil_brush = BrushSettings::default();
    pencil_brush.texture.enabled = true;
    pencil_brush.texture.scale = 1.0;
    h.state_mut().ui.tool_brushes = vec![(crate::state::Tool::Pencil, pencil_brush)];
    h.run_steps(1);

    h.get_by_label("Panel menu").click();
    h.run_steps(2);
    h.get_by_label("Copy Texture to Other Tools").click();
    h.run_steps(2);

    let pencil = h.state().ui.tool_brushes.iter().find(|(t, _)| *t == crate::state::Tool::Pencil).unwrap();
    assert_eq!(pencil.1.texture.scale, 2.5);

    // 6. Close closes the window
    let mut h = harness(0, 0);
    assert!(h.state().ui.panels.brush_settings);
    h.get_by_label("Panel menu").click();
    h.run_steps(2);
    let close_buttons: Vec<_> = h.query_all_by_label("Close").collect();
    assert!(close_buttons.len() >= 2);
    close_buttons.last().unwrap().click();
    h.run_steps(2);
    assert!(!h.state().ui.panels.brush_settings);

    // 7. Close Tab Group closes the window
    let mut h = harness(0, 0);
    assert!(h.state().ui.panels.brush_settings);
    h.get_by_label("Panel menu").click();
    h.run_steps(2);
    h.get_by_label("Close Tab Group").click();
    h.run_steps(2);
    assert!(!h.state().ui.panels.brush_settings);
}

#[test]
fn brushes_button_at_top_of_brush_settings_opens_brushes_panel() {
    use egui_kittest::kittest::Queryable;

    let mut h = harness(0, 0);
    assert_eq!(h.state().ui.brush_tab, 0);
    h.get_by_label("Brushes").click();
    h.run_steps(3);
    assert!(h.state().ui.panels.brush_settings);
    assert_eq!(h.state().ui.brush_tab, 1);
}

/// The fill painted for the check box in `rect` in the last frame (the topmost filled rect there).
fn box_fill(h: &Harness<'_, PhotocraftApp>, rect: egui::Rect) -> Option<Color32> {
    h.output()
        .shapes
        .iter()
        .filter_map(|c| match &c.shape {
            egui::Shape::Rect(r) if (r.rect.min - rect.min).length() < 1.0 && (r.rect.max - rect.max).length() < 1.0 && r.fill != Color32::TRANSPARENT => {
                Some(r.fill)
            }
            _ => None,
        })
        .next_back()
}

/// #4476: while the mouse is down on a section's row or its box, the box shows it's pressed, as
/// in Photoshop: an unchecked box darkens, a checked one turns less opaque. The release then
/// acts as before.
#[test]
fn section_boxes_show_a_pressed_state() {
    use egui_kittest::kittest::Queryable;
    let mut h = app_harness();
    h.key_press(egui::Key::F5);
    h.run_steps(3);
    let t = Tokens::get(&h.ctx);
    let boxr = h.get_by_label("Enable Shape Dynamics").rect();
    let row = h.get_by_label("Shape Dynamics").rect();
    assert!(!h.state().session.tools.brush.shape_dynamics.enabled);
    assert_eq!(box_fill(&h, boxr), Some(widgets::check_box_fill(false, false, &t)));
    // Held on the row's name: the unchecked box darkens; the release turns the section on.
    let at = egui::pos2(row.center().x, row.center().y);
    h.drag_at(at);
    h.run_steps(2);
    assert_eq!(box_fill(&h, boxr), Some(widgets::check_box_fill(false, true, &t)), "pressed, unchecked");
    h.drop_at(at);
    h.run_steps(3);
    assert!(h.state().session.tools.brush.shape_dynamics.enabled);
    assert_eq!(box_fill(&h, boxr), Some(widgets::check_box_fill(true, false, &t)));
    // Held on the checked box itself: less opaque, then the release turns it off.
    h.drag_at(boxr.center());
    h.run_steps(2);
    assert_eq!(box_fill(&h, boxr), Some(widgets::check_box_fill(true, true, &t)), "pressed, checked");
    h.drop_at(boxr.center());
    h.run_steps(3);
    assert!(!h.state().session.tools.brush.shape_dynamics.enabled);
    // A checkbox-only row (no settings of its own) is one checkbox across the row: a press
    // anywhere on it presses its box, and the release toggles it.
    let wet = h.get_by_label("Wet Edges").rect();
    let wet_box = egui::Rect::from_center_size(egui::pos2(wet.left() + 12.0, wet.center().y), vec2(13.0, 13.0));
    let was = h.state().session.tools.brush.wet_edges;
    h.drag_at(wet.center());
    h.run_steps(2);
    assert_eq!(box_fill(&h, wet_box), Some(widgets::check_box_fill(was, true, &t)), "pressed checkbox-only row");
    h.drop_at(wet.center());
    h.run_steps(3);
    assert_ne!(h.state().session.tools.brush.wet_edges, was);
    // The four looks are distinct.
    let looks = [(false, false), (false, true), (true, false), (true, true)].map(|(on, pressed)| widgets::check_box_fill(on, pressed, &t));
    for (i, a) in looks.iter().enumerate() {
        assert!(looks.iter().skip(i + 1).all(|b| b != a), "{looks:?}");
    }
}

/// The shared check box (options bars, dialogs) does the same, and pressing its label counts.
#[test]
fn checkboxes_show_a_pressed_state_from_their_label() {
    use egui_kittest::kittest::Queryable;
    let mut h = Harness::builder().with_size(vec2(300.0, 100.0)).build_ui_state(
        |ui, on: &mut bool| {
            widgets::checkbox(ui, on, "Sample All Layers");
        },
        false,
    );
    h.run();
    let t = Tokens::get(&h.ctx);
    let label = h.get_by_label("Sample All Layers").rect();
    let boxr = egui::Rect::from_min_size(egui::pos2(label.left() - 6.0 - 14.0, label.center().y - 7.0), vec2(14.0, 14.0));
    let fill = |h: &Harness<'_, bool>| {
        h.output()
            .shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Rect(r) if (r.rect.center() - boxr.center()).length() < 1.5 && r.rect.width() < 15.0 && r.fill != Color32::TRANSPARENT => {
                    Some(r.fill)
                }
                _ => None,
            })
            .next_back()
    };
    assert_eq!(fill(&h), Some(widgets::check_box_fill(false, false, &t)));
    h.drag_at(label.center());
    h.run();
    assert_eq!(fill(&h), Some(widgets::check_box_fill(false, true, &t)), "pressing the label presses the box");
    h.drop_at(label.center());
    h.run();
    assert!(*h.state(), "the release checks it");
    assert_eq!(fill(&h), Some(widgets::check_box_fill(true, false, &t)));
}
