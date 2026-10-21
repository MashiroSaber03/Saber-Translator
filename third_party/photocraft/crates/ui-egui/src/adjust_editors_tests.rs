//! UI tests for the adjustment editors: the whole app (eframe harness, CPU canvas) with an
//! adjustment layer on a new document, edited through the Properties panel with real pointer and
//! key input (issue #12: "add an adjustment layer to a new document, edit it, crash"), and the
//! Curves point interactions (issue #43).

use egui::{Pos2, Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::{Adjustment, LayerContent, LayerId};
use serde_json::json;

use crate::PhotocraftApp;

const ALL_KINDS: [&str; 16] = [
    "brightnessContrast",
    "levels",
    "curves",
    "exposure",
    "vibrance",
    "hueSaturation",
    "colorBalance",
    "blackWhite",
    "photoFilter",
    "channelMixer",
    "invert",
    "posterize",
    "threshold",
    "gradientMap",
    "selectiveColor",
    "colorLookup",
];

fn app_harness(kind: &str, mode: &str, depth: u32) -> Harness<'static, PhotocraftApp> {
    app_harness_stepped(kind, mode, depth, 0.25)
}

/// [`app_harness`] advancing `step_dt` seconds per frame (double-clicks need real frame times).
fn app_harness_stepped(kind: &str, mode: &str, depth: u32, step_dt: f32) -> Harness<'static, PhotocraftApp> {
    let (kind, mode) = (kind.to_string(), mode.to_string());
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).with_step_dt(step_dt).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 96, "height": 64, "mode": mode, "depth": depth})).unwrap();
        // Two tones so histograms (and Levels' Auto) have something to work with.
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 48, "height": 64})).unwrap();
        s.execute("edit.fill", json!({"color": "#6a3020"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        s.execute(&format!("layer.newAdjustmentLayer.{kind}"), json!({})).unwrap();
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        // The default Properties group gives way to Layers (#147) and scrolls taller editors;
        // these tests drive every control without scrolling, so they size it like a user would.
        app.ui.dock.heights.insert(crate::dock::Group::Properties, 560.0);
        app
    });
    h.run_steps(8);
    h
}

fn layer(h: &Harness<'_, PhotocraftApp>) -> (LayerId, Adjustment) {
    let st = h.state().session.active().unwrap();
    let id = st.active_layer.unwrap();
    match &st.doc.layer(id).unwrap().content {
        LayerContent::Adjustment(a) => (id, a.clone()),
        other => panic!("{other:?}"),
    }
}

fn drag(h: &mut Harness<'_, PhotocraftApp>, from: Pos2, to: Pos2) {
    h.hover_at(from);
    h.run_steps(1);
    h.drag_at(from);
    h.run_steps(1);
    for i in 1..=6 {
        h.hover_at(from + (to - from) * (i as f32 / 6.0));
        h.run_steps(1);
    }
    h.drop_at(to);
    h.run_steps(3);
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    h.drag_at(at);
    h.run_steps(1);
    h.drop_at(at);
    h.run_steps(3);
}

fn label_rect(h: &Harness<'_, PhotocraftApp>, text: &str) -> Option<Rect> {
    // The last match: the Properties header repeats the kind's name above its controls.
    h.query_all_by_label(text).last().map(|n| n.rect())
}

fn curves_graph(h: &Harness<'_, PhotocraftApp>, id: LayerId) -> Rect {
    h.ctx.data(|d| d.get_temp::<Rect>(egui::Id::new(("adjust-layer", id.0)).with("curves-graph"))).expect("curves graph drawn")
}

/// Issue #12: every adjustment kind, on a new document at 8/16/32 bits and in RGB, Grayscale,
/// CMYK and Lab, opens its Properties editor and takes an edit through real input without
/// panicking; slider kinds commit exactly one history step per gesture.
#[test]
fn every_adjustment_layer_edits_through_properties() {
    for (mode, depth) in [("rgb", 8), ("rgb", 16), ("rgb", 32), ("gray", 8), ("cmyk", 8), ("lab", 16)] {
        for kind in ALL_KINDS {
            let mut h = app_harness(kind, mode, depth);
            let (id, before) = layer(&h);
            let steps = h.state().session.active().unwrap().history.past_len();
            // A probe label per editor; the control sits just below it.
            let probe = match kind {
                "brightnessContrast" => Some("Brightness"),
                "exposure" => Some("Exposure"),
                "vibrance" => Some("Vibrance"),
                "hueSaturation" => Some("Hue"),
                "colorBalance" => Some("Cyan  ·  Red"),
                "blackWhite" => Some("Reds"),
                "photoFilter" => Some("Density"),
                "channelMixer" => Some("Constant"),
                "posterize" => Some("Levels"),
                "threshold" => Some("Threshold Level"),
                "selectiveColor" => Some("Cyan"),
                _ => None,
            };
            match kind {
                "curves" => {
                    let g = curves_graph(&h, id);
                    drag(&mut h, g.center(), g.center() + vec2(0.0, -40.0));
                }
                "levels" => {
                    // The output black handle sits below the output gradient bar.
                    let r = label_rect(&h, "Output Levels:").unwrap_or_else(|| panic!("{kind} {mode}: Output Levels"));
                    let y = r.bottom() + 17.0;
                    drag(&mut h, Pos2::new(r.left() + 6.0, y), Pos2::new(r.left() + 60.0, y));
                }
                "gradientMap" => {
                    let r = label_rect(&h, "Reverse").unwrap_or_else(|| panic!("{kind} {mode}: Reverse"));
                    click(&mut h, r.center());
                }
                _ => {
                    if let Some(p) = probe {
                        let r = label_rect(&h, p).unwrap_or_else(|| panic!("{kind} {mode}: no `{p}` label"));
                        let y = r.bottom() + 14.0;
                        drag(&mut h, Pos2::new(r.left() + 40.0, y), Pos2::new(r.left() + 120.0, y));
                    }
                }
            }
            h.run_steps(4);
            let (_, after) = layer(&h);
            assert!(h.state().live_adjust.is_none(), "{kind} {mode}: preview ended with the gesture");
            if !matches!(kind, "invert" | "colorLookup") {
                assert_ne!(after, before, "{kind} {mode}@{depth}: the edit reached the layer");
                assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "{kind} {mode}: one history step");
            }
        }
    }
}

/// #3264: wheel browsing previews without recording one edit per selected LUT.
#[test]
fn color_lookup_picker_cycles_luts_with_the_mouse_wheel() {
    let mut h = app_harness("colorLookup", "rgb", 8);
    let (_, before) = layer(&h);
    let Adjustment::ColorLookup { name: before_name, .. } = before else { panic!("Color Lookup layer") };
    let label = label_rect(&h, "3D LUT File").expect("Color Lookup file label");
    let at = h
        .query_all_by_role(egui::accesskit::Role::ComboBox)
        .min_by(|a, b| a.rect().center().distance(label.center()).total_cmp(&b.rect().center().distance(label.center())))
        .expect("Color Lookup picker")
        .rect()
        .center();
    let steps = h.state().session.active().unwrap().history.past_len();
    h.hover_at(at);
    h.run_steps(1);
    h.query_all_by_role(egui::accesskit::Role::ComboBox)
        .min_by(|a, b| a.rect().center().distance(label.center()).total_cmp(&b.rect().center().distance(label.center())))
        .expect("Color Lookup picker")
        .click();
    h.run_steps(1);
    h.key_press(egui::Key::ArrowDown);
    h.run_steps(1);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "arrow navigation previews");
    assert!(h.state().live_adjust.is_some(), "arrow navigation selects a preview LUT");
    for _ in 0..3 {
        h.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: vec2(0.0, -1.0),
            phase: egui::TouchPhase::Move,
            modifiers: Default::default(),
        });
        h.run_steps(1);
    }
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "cycling remains a preview");
    assert!(h.state().live_adjust.is_some(), "the selected LUT is previewed");
    h.run_steps(4);
    let (_, after) = layer(&h);
    let Adjustment::ColorLookup { name: after_name, .. } = after else { panic!("Color Lookup layer") };
    assert_ne!(after_name, before_name, "wheel browsing selects a different LUT");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "one wheel gesture makes one adjustment edit");
    assert!(h.state().live_adjust.is_none(), "the completed wheel gesture is committed");
}

fn curve_of(a: &Adjustment) -> Vec<[f32; 2]> {
    match a {
        Adjustment::Curves { master, .. } => master.iter().map(|p| [(p.input * 255.0).round(), (p.output * 255.0).round()]).collect(),
        other => panic!("{other:?}"),
    }
}

/// Issue #43: pressing on a point selects it (no new point), Delete/Backspace and ⌘/Ctrl-click
/// remove points, dragging off the graph removes one, endpoints stay.
#[test]
fn curves_points_are_added_selected_and_deleted() {
    let mut h = app_harness("curves", "rgb", 8);
    let (id, _) = layer(&h);
    let g = curves_graph(&h, id);
    let at = |v: [f32; 2]| Pos2::new(g.left() + v[0] / 255.0 * g.width(), g.bottom() - v[1] / 255.0 * g.height());
    // Add a point.
    click(&mut h, at([128.0, 128.0]));
    assert_eq!(curve_of(&layer(&h).1).len(), 3);
    // Press 4 px off it and drag: moves that point, adds none.
    drag(&mut h, at([128.0, 128.0]) + vec2(4.0, 3.0), at([128.0, 180.0]));
    let c = curve_of(&layer(&h).1);
    assert_eq!(c.len(), 3, "{c:?}");
    assert!(c[1][1] > 165.0, "{c:?}");
    // Delete key removes the selected point (the graph has focus).
    h.key_press(egui::Key::Delete);
    h.run_steps(3);
    assert_eq!(curve_of(&layer(&h).1).len(), 2);
    // Backspace too.
    click(&mut h, at([64.0, 100.0]));
    assert_eq!(curve_of(&layer(&h).1).len(), 3);
    h.key_press(egui::Key::Backspace);
    h.run_steps(3);
    assert_eq!(curve_of(&layer(&h).1).len(), 2);
    // ⌘/Ctrl-click removes.
    click(&mut h, at([190.0, 120.0]));
    assert_eq!(curve_of(&layer(&h).1).len(), 3);
    let p = at([190.0, 120.0]);
    h.event_modifiers(egui::Event::PointerMoved(p), egui::Modifiers::COMMAND);
    h.event_modifiers(
        egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::COMMAND },
        egui::Modifiers::COMMAND,
    );
    h.run_steps(1);
    h.event_modifiers(
        egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::COMMAND },
        egui::Modifiers::COMMAND,
    );
    h.run_steps(3);
    assert_eq!(curve_of(&layer(&h).1).len(), 2);
    // Dragging a point off the graph removes it; endpoints survive both.
    click(&mut h, at([100.0, 60.0]));
    drag(&mut h, at([100.0, 60.0]), g.right_center() + vec2(80.0, 0.0));
    assert_eq!(curve_of(&layer(&h).1).len(), 2);
    drag(&mut h, at([0.0, 0.0]), g.left_top() + vec2(-80.0, -80.0));
    click(&mut h, at([0.0, 0.0]));
    h.key_press(egui::Key::Delete);
    h.run_steps(3);
    assert_eq!(curve_of(&layer(&h).1).len(), 2, "endpoints can't be deleted");
}

/// Capture and history are shared with Camera Raw, but Properties commits on release.
#[test]
fn curves_fast_release_commits_one_edit_and_undo_restores_the_curve() {
    let mut h = app_harness("curves", "rgb", 8);
    let (id, _) = layer(&h);
    let graph = curves_graph(&h, id);
    click(&mut h, graph.center());
    let (_, before) = layer(&h);
    let steps = h.state().session.active().unwrap().history.past_len();
    let at = graph.center() + vec2(4.0, 3.0);
    h.hover_at(at);
    h.run_steps(1);
    h.drag_at(at);
    h.run_steps(1);
    // Only the release frame supplies the destination, as a very fast physical gesture can.
    h.drop_at(at + vec2(40.0, -32.0));
    h.run_steps(3);
    let after = curve_of(&layer(&h).1);
    assert_eq!(after.len(), 3);
    assert!(after[1][0] > 150.0 && after[1][1] > 150.0, "{after:?}");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
    assert!(h.state().live_adjust.is_none());
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(layer(&h).1, before);
}

/// Real pointer input: `n` clicks at `at` with `mods` held (2 = a double-click).
fn clicks_with(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, n: usize, mods: egui::Modifiers) {
    // Well past the double-click time since the last gesture.
    h.run_steps(40);
    h.event(egui::Event::ModifiersChanged(mods));
    h.hover_at(at);
    h.run_steps(1);
    for _ in 0..n {
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: mods });
            h.run_steps(1);
        }
    }
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(3);
}

fn wheel_at(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, notches: f32, mods: egui::Modifiers) {
    h.event(egui::Event::ModifiersChanged(mods));
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: vec2(0.0, notches), phase: egui::TouchPhase::Move, modifiers: mods });
    h.run_steps(1);
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(3);
}

fn key_with(h: &mut Harness<'_, PhotocraftApp>, key: egui::Key, mods: egui::Modifiers) {
    h.event(egui::Event::ModifiersChanged(mods));
    for pressed in [true, false] {
        h.event(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: mods });
    }
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(3);
}

fn color_balance_of(a: &Adjustment) -> ([f32; 3], [f32; 3], [f32; 3], bool) {
    match a {
        Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity } => (*shadows, *midtones, *highlights, *preserve_luminosity),
        other => panic!("{other:?}"),
    }
}

/// Photoshop 25.4, Color Balance in the Properties panel (measured on the real application by
/// scripted clicks and reading the layer back): a double-click on a slider's knob or track sets that
/// slider of the shown tone to 0, whatever modifier is held; the label doesn't; a click jumps; the
/// wheel does nothing; Up/Down in a field step 1, with Shift 10.
#[test]
fn color_balance_sliders_reset_and_step_like_photoshop() {
    let mut h = app_harness_stepped("colorBalance", "rgb", 8, 1.0 / 60.0);
    let (id, _) = layer(&h);
    let start = json!({"layer": id.0, "shadows": [10, 0, 0], "midtones": [40, -30, 20], "highlights": [0, 0, -15], "preserveLuminosity": false});
    h.state_mut().run("layer.setAdjustment", start.clone()).unwrap();
    h.run_steps(4);
    let label = label_rect(&h, "Cyan  ·  Red").expect("Cyan · Red row");
    let on_track = Pos2::new(label.left() + 60.0, label.bottom() + 14.0);
    let cb = |h: &Harness<'_, PhotocraftApp>| color_balance_of(&layer(h).1);

    // A click jumps to the pointer, with or without modifiers.
    clicks_with(&mut h, on_track, 1, egui::Modifiers::NONE);
    let jumped = cb(&h).1[0];
    assert!(jumped < 0.0 && jumped > -100.0, "click jumps to the pointer: {jumped}");
    for mods in [egui::Modifiers::SHIFT, egui::Modifiers::CTRL, egui::Modifiers::ALT] {
        h.state_mut().run("layer.setAdjustment", start.clone()).unwrap();
        h.run_steps(3);
        clicks_with(&mut h, on_track, 1, mods);
        assert_eq!(cb(&h).1[0], jumped, "{mods:?}-click is a plain click");
    }
    // A double-click anywhere on the slider zeroes it, with any modifier; nothing else changes.
    for mods in [egui::Modifiers::NONE, egui::Modifiers::SHIFT, egui::Modifiers::CTRL, egui::Modifiers::ALT] {
        h.state_mut().run("layer.setAdjustment", start.clone()).unwrap();
        h.run_steps(3);
        clicks_with(&mut h, on_track, 2, mods);
        assert_eq!(cb(&h), ([10.0, 0.0, 0.0], [0.0, -30.0, 20.0], [0.0, 0.0, -15.0], false), "{mods:?} double-click");
    }
    // The label is not a reset target.
    h.state_mut().run("layer.setAdjustment", start.clone()).unwrap();
    h.run_steps(3);
    clicks_with(&mut h, label.center(), 2, egui::Modifiers::NONE);
    assert_eq!(cb(&h).1, [40.0, -30.0, 20.0], "double-click on the label");
    // The panel's sliders ignore the wheel.
    wheel_at(&mut h, on_track, 3.0, egui::Modifiers::NONE);
    wheel_at(&mut h, on_track, 3.0, egui::Modifiers::SHIFT);
    assert_eq!(cb(&h).1, [40.0, -30.0, 20.0], "wheel over a Properties slider");
    // Up/Down in the field: 1, Shift: 10 (also below zero, where a half step used to round away).
    let field = h
        .query_all_by_role(egui::accesskit::Role::SpinButton)
        .map(|n| n.rect())
        .find(|r| (r.center().y - label.center().y).abs() < 8.0)
        .expect("Cyan · Red field");
    clicks_with(&mut h, field.center(), 1, egui::Modifiers::NONE);
    key_with(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
    assert_eq!(cb(&h).1[0], 41.0, "Up");
    key_with(&mut h, egui::Key::ArrowUp, egui::Modifiers::SHIFT);
    assert_eq!(cb(&h).1[0], 51.0, "Shift+Up");
    for _ in 0..8 {
        key_with(&mut h, egui::Key::ArrowDown, egui::Modifiers::SHIFT);
    }
    key_with(&mut h, egui::Key::ArrowDown, egui::Modifiers::NONE);
    assert_eq!(cb(&h).1[0], -30.0, "Shift+Down ×8, Down");
    key_with(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
    assert_eq!(cb(&h).1[0], -29.0, "Up below zero");
}

/// Photoshop 25.4, Color Balance in the Properties panel: edits that follow each other make one
/// history step (a double-click, whose first click moved the knob, undoes in one go); Reset first
/// returns to the settings the layer had when the panel started showing it (Tone choice kept),
/// then to the defaults (every tone 0, Midtones, Preserve Luminosity kept), within that step.
#[test]
fn color_balance_properties_history_and_reset_like_photoshop() {
    let mut h = app_harness_stepped("colorBalance", "rgb", 8, 1.0 / 60.0);
    let (id, _) = layer(&h);
    let start = ([10.0, 0.0, 0.0], [40.0, -30.0, 20.0], [0.0, 0.0, -15.0], false);
    h.state_mut()
        .run(
            "layer.setAdjustment",
            json!({"layer": id.0, "shadows": [10, 0, 0], "midtones": [40, -30, 20], "highlights": [0, 0, -15], "preserveLuminosity": false}),
        )
        .unwrap();
    // Show another layer, then this one again: the panel starts showing it with these settings.
    let bg = h.state().session.active().unwrap().doc.layers[0].id;
    h.state_mut().session.select_layer(bg).unwrap();
    h.run_steps(4);
    h.state_mut().session.select_layer(id).unwrap();
    h.run_steps(4);
    let cb = |h: &Harness<'_, PhotocraftApp>| color_balance_of(&layer(h).1);
    let steps = |h: &Harness<'_, PhotocraftApp>| h.state().session.active().unwrap().history.past_len();
    let label = label_rect(&h, "Cyan  ·  Red").expect("Cyan · Red row");
    let on_track = Pos2::new(label.left() + 60.0, label.bottom() + 14.0);
    let before = steps(&h);
    clicks_with(&mut h, on_track, 1, egui::Modifiers::NONE);
    clicks_with(&mut h, Pos2::new(label.left() + 90.0, on_track.y), 2, egui::Modifiers::NONE);
    assert_eq!(cb(&h).1, [0.0, -30.0, 20.0]);
    assert_eq!(steps(&h), before + 1, "a click and a double-click: one history step");
    assert!(h.state_mut().session.undo());
    h.run_steps(3);
    assert_eq!(cb(&h), start, "one Undo returns to the settings before them");
    // Edit with Highlights shown; Reset returns to the settings the panel started with.
    let hl = label_rect(&h, "Highlights").expect("Highlights radio");
    clicks_with(&mut h, hl.center(), 1, egui::Modifiers::NONE);
    clicks_with(&mut h, on_track, 1, egui::Modifiers::NONE);
    assert_ne!(cb(&h), start);
    let tone = crate::adjust_editors::layer_mem(id).with("cb-tone");
    let reset = h.get_by_label("Reset to defaults").rect();
    let before = steps(&h);
    clicks_with(&mut h, reset.center(), 1, egui::Modifiers::NONE);
    assert_eq!(cb(&h), start, "first Reset: the settings the panel started with");
    assert_eq!(h.ctx.data(|d| d.get_temp::<usize>(tone)), Some(2), "Tone choice kept");
    clicks_with(&mut h, reset.center(), 1, egui::Modifiers::NONE);
    assert_eq!(cb(&h), ([0.0; 3], [0.0; 3], [0.0; 3], false), "second Reset: defaults, Preserve Luminosity kept");
    assert_eq!(h.ctx.data(|d| d.get_temp::<usize>(tone)), Some(1), "defaults show Midtones");
    clicks_with(&mut h, reset.center(), 1, egui::Modifiers::NONE);
    assert_eq!(cb(&h), ([0.0; 3], [0.0; 3], [0.0; 3], false), "third Reset: still the defaults");
    assert_eq!(steps(&h), before, "the edit and the Resets share one history step");
}

fn map_stops(a: &Adjustment) -> Vec<(f32, [u8; 3])> {
    let q = |v: f32| (v * 255.0).round() as u8;
    match a {
        Adjustment::GradientMap { stops, .. } => stops.iter().map(|(t, c)| (*t, [q(c[0]), q(c[1]), q(c[2])])).collect(),
        other => panic!("{other:?}"),
    }
}

fn map_handle(h: &Harness<'_, PhotocraftApp>, at: f32) -> Pos2 {
    let strip = h.get_by_label("Gradient stops").rect();
    Pos2::new(strip.left() + 8.0 + at * (strip.width() - 16.0), strip.top() + 36.0)
}

#[test]
fn gradient_map_handle_drag_previews_and_commits_once_after_crossing_stops() {
    for depth in [8, 16, 32] {
        let mut h = app_harness("gradientMap", "rgb", depth);
        let (id, _) = layer(&h);
        h.state_mut()
            .run(
                "layer.setAdjustment",
                json!({
                    "layer": id.0, "stops": [[0, "#000000"], [0.25, "#ff0000"], [0.5, "#00ff00"], [1, "#ffffff"]]
                }),
            )
            .unwrap();
        h.run_steps(3);
        let before = layer(&h).1;
        let steps = h.state().session.active().unwrap().history.past_len();
        let start = map_handle(&h, 0.25);
        h.hover_at(start);
        h.run_steps(1);
        h.drag_at(start);
        h.run_steps(1);
        for location in [0.4, 0.5, 0.6, 0.75] {
            let to = map_handle(&h, location);
            h.hover_at(to);
            h.run_steps(1);
            assert!(h.state().live_adjust.is_some(), "{depth}: live preview");
            assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "{depth}: dragging records no history");
        }
        let end = map_handle(&h, 0.8);
        h.drop_at(end);
        h.run_steps(3);
        assert_eq!(map_stops(&layer(&h).1), vec![(0.0, [0, 0, 0]), (0.5, [0, 255, 0]), (0.8, [255, 0, 0]), (1.0, [255, 255, 255])]);
        assert_eq!(h.ctx.data(|d| d.get_temp::<usize>(crate::adjust_editors::layer_mem(id).with("gm-stop-selection"))), Some(2));
        assert!(h.state().live_adjust.is_none());
        assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
        let after = layer(&h).1;
        h.state_mut().run("edit.undo", json!({})).unwrap();
        h.run_steps(3);
        assert_eq!(layer(&h).1, before);
        h.state_mut().run("edit.redo", json!({})).unwrap();
        h.run_steps(3);
        assert_eq!(layer(&h).1, after);
    }
}

#[test]
fn gradient_map_strip_adds_selects_and_removes_stops_without_deleting_the_last_pair() {
    let mut h = app_harness("gradientMap", "rgb", 8);
    let (id, original) = layer(&h);
    let steps = h.state().session.active().unwrap().history.past_len();
    let middle = map_handle(&h, 0.5);
    click(&mut h, middle);
    let added = map_stops(&layer(&h).1);
    assert_eq!(added.len(), 3);
    assert!((added[1].0 - 0.5).abs() < 0.001);
    assert_eq!(added[1].1, [128, 128, 128]);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
    let first = map_handle(&h, 0.0);
    click(&mut h, first);
    assert_eq!(h.ctx.data(|d| d.get_temp::<usize>(crate::adjust_editors::layer_mem(id).with("gm-stop-selection"))), Some(0));
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "selection is view state");
    click(&mut h, middle);
    let remove = h.get_by_label("Remove stop").rect().center();
    click(&mut h, remove);
    assert_eq!(layer(&h).1, original);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 2);
    let first = map_handle(&h, 0.0);
    click(&mut h, first);
    let remove = h.get_by_label("Remove stop").rect().center();
    click(&mut h, remove);
    assert_eq!(layer(&h).1, original, "keep two stops");
}

#[test]
fn gradient_map_drag_clamps_at_the_endpoints_and_reverse_keeps_stop_coordinates() {
    let mut h = app_harness("gradientMap", "rgb", 8);
    let reverse = label_rect(&h, "Reverse").unwrap().center();
    click(&mut h, reverse);
    let first = map_handle(&h, 0.0);
    let target = map_handle(&h, 0.3);
    drag(&mut h, first, target);
    assert!((map_stops(&layer(&h).1)[0].0 - 0.3).abs() < 0.001);
    let target = map_handle(&h, 0.3);
    drag(&mut h, target, first - vec2(60.0, 0.0));
    assert_eq!(map_stops(&layer(&h).1)[0].0, 0.0);
    let last = map_handle(&h, 1.0);
    let beyond = last + vec2(60.0, 0.0);
    let before = layer(&h).1;
    let steps = h.state().session.active().unwrap().history.past_len();
    drag(&mut h, last, beyond);
    assert_eq!(layer(&h).1, before);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "a clamped no-op makes no history");
}

#[test]
fn gradient_map_stop_controls_fit_a_narrow_properties_panel_in_every_theme() {
    use crate::adjust_editors::{EditorCx, editor};
    for kind in crate::theme::ThemeKind::ALL {
        for width in [240.0, 280.0] {
            let mut h = Harness::builder().with_size(vec2(width, 360.0)).build_ui_state(
                |ui, values: &mut serde_json::Value| {
                    if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                        return;
                    }
                    let cx = EditorCx {
                        mem: egui::Id::new("map-layout"),
                        hist: None,
                        gray: false,
                        swatches: [[0.0; 3], [1.0; 3]],
                        gradients: Vec::new(),
                        dialog: false,
                        dynamic_color_sliders: true,
                    };
                    editor(ui, "gradientMap", values, &cx);
                },
                json!({"stops": [[0, "#000000"], [1, "#ffffff"]]}),
            );
            PhotocraftApp::setup_context(&h.ctx, kind);
            h.run_steps(4);
            for role in [egui::accesskit::Role::Button, egui::accesskit::Role::ComboBox, egui::accesskit::Role::SpinButton] {
                for control in h.query_all_by_role(role) {
                    let r = control.rect();
                    assert!(r.left() >= 0.0 && r.right() <= width && r.width() >= 8.0, "{kind:?} width {width}: {role:?} {r:?}");
                }
            }
        }
    }
}

/// Issue #1588: the Gradient Map preset picker offers the session's gradient library (built-in
/// groups beyond Basics and the user's own saved gradients), and picking one writes its stops as
/// one history step.
#[test]
fn gradient_map_picker_offers_the_gradient_library() {
    use photocraft_engine::presets::{Group, gradients};
    let mut h = app_harness("gradientMap", "rgb", 8);
    // A short library so every row fits the popup: Basics, Blues and a user-saved group.
    let mut lib: Vec<_> = gradients::builtin().into_iter().take(2).collect();
    lib[1].items.truncate(3);
    lib.push(Group::new("Mine", vec![gradients::GradientPreset::new("My Teal", &["#008080", "#ffee00"])]));
    h.state_mut().session.presets.gradients = lib;
    h.run_steps(2);
    let steps = |h: &Harness<'_, PhotocraftApp>| h.state().session.active().unwrap().history.past_len();
    for (name, want) in
        [("Blue 03", vec![(0.0, [0x00, 0xb4, 0xd8]), (1.0, [0x03, 0x04, 0x5e])]), ("My Teal", vec![(0.0, [0x00, 0x80, 0x80]), (1.0, [0xff, 0xee, 0x00])])]
    {
        let before = steps(&h);
        // The picker sits right of its "Preset:" label.
        let preset = label_rect(&h, "Preset:").expect("the preset picker");
        click(&mut h, Pos2::new(preset.right() + 60.0, preset.center().y));
        let item = h.query_by_label(name).unwrap_or_else(|| panic!("`{name}` in the picker")).rect();
        click(&mut h, item.center());
        h.run_steps(4);
        assert_eq!(map_stops(&layer(&h).1), want, "{name}");
        assert_eq!(steps(&h), before + 1, "{name}: one history step");
    }
}

fn import_gradients_from_picker(h: &mut Harness<'_, PhotocraftApp>) {
    let preset = label_rect(h, "Preset:").unwrap();
    click(h, Pos2::new(preset.right() + 60.0, preset.center().y));
    h.get_by_label("Gradient Preset Options").click();
    h.run_steps(2);
    h.get_by_label("Import .GRD").click();
    h.run_steps(3);
}

#[test]
fn gradient_map_picker_imports_grd_without_editing_the_map() {
    use photocraft_psd::grd::{GrdColor, GrdGradient, GrdStop};
    let mut h = app_harness("gradientMap", "rgb", 16);
    h.state_mut().session.presets.gradients.truncate(1);
    let before = layer(&h);
    let history = h.state().session.active().unwrap().history.past_len();
    let tool = h.state().session.presets.gradient.clone();
    let bytes = photocraft_psd::grd::write(&[GrdGradient {
        name: "Imported Teal".into(),
        stops: vec![
            GrdStop { location: 0.0, midpoint: 0.5, color: GrdColor::Rgb([0.0, 0.5, 0.5]) },
            GrdStop { location: 1.0, midpoint: 0.5, color: GrdColor::Rgb([1.0, 1.0, 0.0]) },
        ],
        ..Default::default()
    }]);
    h.state_mut().services.file_dialog = Some(Box::new(move |request, _, reply| {
        assert!(matches!(request, crate::FileDialogRequest::Open { multiple: false, extensions: Some(ref exts), .. } if exts == &["grd"]));
        reply.send(Some(crate::FileDialogAnswer::Contents("My Gradients.GRD".into(), bytes.clone())));
    }));
    import_gradients_from_picker(&mut h);
    assert_eq!(layer(&h), before, "import does not edit the adjustment");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    assert_eq!(h.state().session.presets.gradient, tool);
    assert!(!h.state().ui.status_error);
    assert!(h.state().ui.status.contains("Imported 1 gradient"));
    let preset = label_rect(&h, "Preset:").unwrap();
    click(&mut h, Pos2::new(preset.right() + 60.0, preset.center().y));
    h.get_by_label("My Gradients");
    h.get_by_label("Imported Teal").click();
    h.run_steps(3);
    assert_eq!(map_stops(&layer(&h).1), vec![(0.0, [0, 128, 128]), (1.0, [255, 255, 0])]);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(layer(&h), before);
}

#[test]
fn gradient_map_picker_import_cancel_and_errors_preserve_state() {
    use crate::FileDialogAnswer;
    for answer in [
        None,
        Some(FileDialogAnswer::Contents("broken.grd".into(), b"invalid".to_vec())),
        Some(FileDialogAnswer::Contents("old.grd".into(), b"8BGR\0\x03".to_vec())),
        Some(FileDialogAnswer::Contents(
            "noise.grd".into(),
            photocraft_psd::grd::write(&[photocraft_psd::grd::GrdGradient { noise: true, ..Default::default() }]),
        )),
        Some(FileDialogAnswer::Contents("wrong.abr".into(), Vec::new())),
        Some(FileDialogAnswer::Paths(vec!["/nonexistent/photocraft-gradients.grd".into()])),
        Some(FileDialogAnswer::Unavailable("dialog unavailable".into())),
    ] {
        let mut h = app_harness("gradientMap", "rgb", 8);
        h.state_mut().session.presets.gradients.truncate(1);
        let before = layer(&h);
        let presets = h.state().session.presets.gradients.clone();
        let history = h.state().session.active().unwrap().history.past_len();
        let error = answer.is_some();
        h.state_mut().services.file_dialog = Some(Box::new(move |_, _, reply| reply.send(answer.clone())));
        import_gradients_from_picker(&mut h);
        assert_eq!(layer(&h), before);
        assert_eq!(h.state().session.presets.gradients, presets);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
        assert_eq!(h.state().ui.status_error, error);
        if error {
            assert!(!h.state().ui.status.is_empty());
        }
    }
}

#[test]
fn gradient_map_dialog_import_preserves_pending_settings() {
    let mut h = app_harness("gradientMap", "rgb", 8);
    let id = crate::adjust_dialog::open(h.state_mut(), "image.adjustments.gradientMap").unwrap();
    let fields = h.state_mut().ui.dialog_mut(id).unwrap().fields.clone();
    h.state_mut().services.file_dialog = Some(Box::new(|request, _, reply| {
        assert!(matches!(request, crate::FileDialogRequest::Open { extensions: Some(ref exts), .. } if exts == &["grd"]));
        reply.send(None);
    }));
    h.run_steps(3);
    import_gradients_from_picker(&mut h);
    assert_eq!(h.state_mut().ui.dialog_mut(id).unwrap().fields, fields);
    assert!(!h.state().file_dialog_open());
    assert!(!h.state().ui.status_error);
}

/// Document point (x, y) on screen in the app's first view.
fn doc_on_screen(h: &Harness<'_, PhotocraftApp>, x: f32, y: f32) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    let xf = crate::canvas::ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    };
    xf.to_screen(x, y)
}

/// UI-217-15 (#3110, #3116): the Properties panel of a Curves or Levels layer has the Set
/// Black/Neutral Gray/White Point eyedroppers. With Set White Point armed, a click on the image
/// makes that pixel white as one history step (sampled below the layer), and Esc disarms the
/// picker.
#[test]
fn curves_properties_white_point_eyedropper_edits_the_layer() {
    properties_white_point_eyedropper_edits_the_layer("curves");
}

#[test]
fn levels_properties_white_point_eyedropper_edits_the_layer() {
    properties_white_point_eyedropper_edits_the_layer("levels");
}

fn properties_white_point_eyedropper_edits_the_layer(kind: &str) {
    let mut h = app_harness(kind, "rgb", 8);
    let (_, before) = layer(&h);
    let steps = h.state().session.active().unwrap().history.past_len();
    h.get_by_label("Set White Point").click();
    h.run_steps(2);
    assert!(crate::layer_picker::armed(h.state()).is_some(), "armed from the Properties panel");
    // The left half is #6a3020: the white point moves to it, so it composites white.
    let at = doc_on_screen(&h, 10.0, 10.0);
    click(&mut h, at);
    let pixel = |h: &mut Harness<'_, PhotocraftApp>| -> Vec<f32> {
        serde_json::from_value(h.state_mut().run("document.pixel", json!({"x": 10, "y": 10})).unwrap()).unwrap()
    };
    let px = pixel(&mut h);
    assert!(px.iter().take(3).all(|v| *v > 0.99), "{kind}: {px:?}");
    assert_ne!(layer(&h).1, before, "{kind}: the layer changed");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "{kind}: one history step");
    assert!(crate::layer_picker::armed(h.state()).is_some(), "stays armed for another click");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);
    assert!(h.state().ui.layer_picker.is_none(), "Esc disarms");
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(layer(&h).1, before);
}

/// A bare `kind` editor over `params`, as the Properties panel and the dialog host it.
fn editor_harness(kind: &'static str, params: serde_json::Value) -> Harness<'static, serde_json::Value> {
    use crate::adjust_editors::{EditorCx, editor};
    let mut h = Harness::builder().with_size(vec2(300.0, 400.0)).build_ui_state(
        move |ui, values: &mut serde_json::Value| {
            let cx = EditorCx {
                mem: egui::Id::new("precision"),
                hist: None,
                gray: false,
                swatches: [[0.0; 3], [1.0; 3]],
                gradients: Vec::new(),
                dialog: true,
                dynamic_color_sliders: true,
            };
            editor(ui, kind, values, &cx);
        },
        params,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h
}

/// Types `text` into the `index`th number field and commits it with Enter.
fn type_into_field(h: &mut Harness<'static, serde_json::Value>, index: usize, text: &str) {
    h.query_all_by_role(egui::accesskit::Role::SpinButton).nth(index).unwrap().click();
    h.run_steps(2);
    h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
    h.run_steps(1);
    h.event(egui::Event::Text(text.into()));
    h.run_steps(1);
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
}

/// The text the `index`th number field shows.
fn field_text(h: &Harness<'static, serde_json::Value>, index: usize) -> String {
    h.query_all_by_role(egui::accesskit::Role::SpinButton).nth(index).unwrap().value().unwrap_or_default()
}

/// #4164: Exposure keeps fractional stops (Photoshop: two decimals) and Offset four decimals,
/// instead of rounding Exposure to whole stops because its range is wider than 10.
#[test]
fn exposure_keeps_fractional_stops_and_offset_four_decimals() {
    let mut h = editor_harness("exposure", json!({}));
    type_into_field(&mut h, 0, "0.5");
    assert_eq!(h.state()["exposure"].as_f64(), Some(0.5), "{}", h.state());
    type_into_field(&mut h, 0, "-1.37");
    assert!((h.state()["exposure"].as_f64().unwrap() + 1.37).abs() < 1e-4, "{}", h.state());
    type_into_field(&mut h, 1, "0.0123");
    assert!((h.state()["offset"].as_f64().unwrap() - 0.0123).abs() < 1e-6, "{}", h.state());
    type_into_field(&mut h, 2, "1.234");
    assert!((h.state()["gamma"].as_f64().unwrap() - 1.23).abs() < 1e-4, "Gamma keeps two decimals: {}", h.state());
    h.run_steps(2);
    assert_eq!(field_text(&h, 0), "-1.37");
    assert_eq!(field_text(&h, 1), "0.0123");
    assert_eq!(field_text(&h, 2), "1.23");
}

/// #4164: the integer rows that share Exposure's slider code stay whole numbers.
#[test]
fn integer_adjustment_rows_stay_whole_numbers() {
    for (kind, key, typed, want) in [
        ("posterize", "levels", "4.6", 5.0),
        ("threshold", "level", "100.4", 100.0),
        ("brightnessContrast", "brightness", "12.7", 13.0),
        ("vibrance", "vibrance", "-20.4", -20.0),
    ] {
        let mut h = editor_harness(kind, json!({}));
        type_into_field(&mut h, 0, typed);
        assert_eq!(h.state()[key].as_f64(), Some(want), "{kind}: {}", h.state());
    }
}

/// #4164: a row's own decimals win over the range rule; rows without them keep it.
#[test]
fn row_values_round_to_their_own_decimals() {
    use crate::adjust_editors::row_value;
    assert_eq!(row_value(0.5, -20.0, 20.0, Some(2)), 0.5);
    assert_eq!(row_value(-1.374, -20.0, 20.0, Some(2)), -1.37);
    assert!((row_value(0.012_34, -0.5, 0.5, Some(4)) - 0.0123).abs() < 1e-7);
    assert_eq!(row_value(4.6, 2.0, 255.0, None), 5.0, "wide ranges stay whole numbers");
    assert!((row_value(1.234, 0.01, 9.99, None) - 1.23).abs() < 1e-6, "small ranges keep two decimals");
    assert!(row_value(f32::NAN, -20.0, 20.0, Some(u32::MAX)).is_nan(), "no panic on odd input");
}
