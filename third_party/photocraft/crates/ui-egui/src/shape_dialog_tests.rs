use photocraft_doc::LayerContent;
use photocraft_doc::vector::LiveShape;
use serde_json::json;

use super::*;
use crate::canvas::{Drag, finish_gesture};

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 400, "height": 300, "background": "white"})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    app
}

fn press(app: &mut PhotocraftApp, tool: Tool, start: [f64; 2], end: [f64; 2]) {
    app.ui.tool = tool;
    let d = Drag::new(tool, start, vec![[start[0], start[1], 1.0], [end[0], end[1], 1.0]], egui::Modifiers::NONE, false);
    finish_gesture(app, d);
}

fn layers(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().doc.layers.len()
}

fn live(app: &PhotocraftApp) -> LiveShape {
    let st = app.session.active().unwrap();
    let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
    let LayerContent::Shape(sh) = &l.content else { panic!("not a shape layer") };
    sh.live.clone().unwrap()
}

fn set(app: &mut PhotocraftApp, id: u64, fields: Value) {
    let d = app.ui.dialog_mut(id).unwrap();
    for (k, v) in fields.as_object().unwrap() {
        d.fields.insert(k.clone(), v.clone());
    }
}

fn only_dialog(app: &PhotocraftApp) -> u64 {
    assert_eq!(app.ui.dialogs.len(), 1, "a click opens one dialog");
    let d = &app.ui.dialogs[0];
    assert!(owns(&d.fields));
    d.id
}

#[test]
fn a_rectangle_click_opens_create_rectangle_and_ok_draws_it_at_the_click() {
    let mut app = app();
    let before = layers(&app);
    press(&mut app, Tool::Rectangle, [20.4, 30.2], [20.4, 30.2]);
    let id = only_dialog(&app);
    assert_eq!(crate::dialogs::title(&app.ui.dialogs[0]), "Create Rectangle");
    assert_eq!(layers(&app), before, "the click itself draws nothing");
    set(&mut app, id, json!({"width": 120, "height": 45.5, "radii": [4, 4, 4, 4]}));
    crate::dialogs::confirm(&mut app, id).unwrap();
    assert!(app.ui.dialogs.is_empty());
    assert_eq!(layers(&app), before + 1);
    assert_eq!(live(&app), LiveShape::Rect { rect: [20.0, 30.0, 120.0, 45.5], radii: [4.0; 4] });
    // The options bar's fill (foreground colour) is used, as for a drag.
    let st = app.session.active().unwrap();
    let LayerContent::Shape(sh) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!() };
    assert!(sh.fill.is_some());
}

#[test]
fn from_center_and_the_other_tools() {
    let mut app = app();
    press(&mut app, Tool::EllipseShape, [200.0, 150.0], [200.0, 150.0]);
    let id = only_dialog(&app);
    assert_eq!(crate::dialogs::title(&app.ui.dialogs[0]), "Create Ellipse");
    set(&mut app, id, json!({"width": 80, "height": 40, "fromCenter": true}));
    crate::dialogs::confirm(&mut app, id).unwrap();
    assert_eq!(live(&app), LiveShape::Ellipse { rect: [160.0, 130.0, 80.0, 40.0] });

    press(&mut app, Tool::Polygon, [10.0, 10.0], [10.0, 10.0]);
    let id = only_dialog(&app);
    set(&mut app, id, json!({"width": 50, "height": 60, "sides": 7, "starRatio": 50}));
    crate::dialogs::confirm(&mut app, id).unwrap();
    assert_eq!(live(&app), LiveShape::Polygon { rect: [10.0, 10.0, 50.0, 60.0], sides: 7, star_ratio: 0.5 });

    press(&mut app, Tool::Triangle, [5.0, 6.0], [5.0, 6.0]);
    let id = only_dialog(&app);
    crate::dialogs::confirm(&mut app, id).unwrap();
    let LiveShape::Polygon { sides: 3, .. } = live(&app) else { panic!("a triangle") };
}

#[test]
fn drags_still_draw_and_tools_without_a_dialog_ignore_clicks() {
    let mut app = app();
    let before = layers(&app);
    press(&mut app, Tool::Rectangle, [10.0, 10.0], [60.0, 40.0]);
    assert!(app.ui.dialogs.is_empty(), "a drag draws without a dialog");
    assert_eq!(live(&app), LiveShape::Rect { rect: [10.0, 10.0, 50.0, 30.0], radii: [0.0; 4] });
    for tool in [Tool::Line, Tool::CustomShape] {
        press(&mut app, tool, [30.0, 30.0], [30.0, 30.0]);
        assert!(app.ui.dialogs.is_empty(), "{tool:?}");
    }
    assert_eq!(layers(&app), before + 1);
}

#[test]
fn click_dialogs_and_drags_keep_complete_stroke_defaults() {
    let style = crate::shape_stroke_ui::StrokeOptions {
        align: photocraft_doc::StrokeAlign::Outside,
        cap: photocraft_doc::LineCap::Square,
        join: photocraft_doc::LineJoin::Bevel,
        miter_limit: 7.0,
        dashes: vec![3.0, 1.0, 0.0, 2.0],
        dash_offset: -0.75,
        opacity: 0.6,
    };
    for tool in [Tool::Rectangle, Tool::EllipseShape, Tool::Triangle, Tool::Polygon] {
        for click in [false, true] {
            let mut app = app();
            app.ui.tool_options.stroke_width = 7.0;
            app.ui.tool_options.shape_stroke = style.clone();
            let history = app.session.active().unwrap().history.past_len();
            press(&mut app, tool, [20.0, 30.0], if click { [20.0, 30.0] } else { [120.0, 130.0] });
            if click {
                let id = only_dialog(&app);
                crate::dialogs::confirm(&mut app, id).unwrap();
            } else {
                assert!(app.ui.dialogs.is_empty());
            }
            let st = app.session.active().unwrap();
            let LayerContent::Shape(sh) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("expected shape") };
            let stroke = sh.stroke.as_ref().unwrap();
            assert_eq!(crate::shape_stroke_ui::StrokeOptions::from(stroke), style, "{tool:?}, click={click}");
            assert_eq!(stroke.width, 7.0);
            assert_eq!(st.history.past_len(), history + 1);
        }
    }
}

#[test]
fn click_slop_scales_with_zoom() {
    let pts = |d: f64| vec![[10.0, 10.0, 1.0], [10.0 + d, 10.0, 1.0]];
    assert!(is_click([10.0, 10.0], &pts(2.0), 1.0));
    assert!(!is_click([10.0, 10.0], &pts(5.0), 1.0));
    assert!(!is_click([10.0, 10.0], &pts(2.0), 4.0), "2 document px at 400% is a drag");
    assert!(is_click([10.0, 10.0], &[], 0.0));
}

/// #4205: the click slop is in screen points, so on a 2x (HiDPI) display at 100 % a 2 pt jitter
/// (4 device px = 4 document px) is still a click that opens Create Rectangle, not a tiny shape.
#[test]
fn a_click_with_jitter_opens_the_dialog_on_hidpi() {
    let mut app = app();
    app.ppp = 2.0;
    let i = app.session.active_index().unwrap();
    app.ui.views[i].zoom = 1.0;
    let before = layers(&app);
    press(&mut app, Tool::Rectangle, [20.0, 30.0], [24.0, 30.0]);
    only_dialog(&app);
    assert_eq!(layers(&app), before, "a jittery click draws nothing");
}

#[test]
fn cancel_draws_nothing_and_ok_remembers_the_size() {
    let mut app = app();
    let before = layers(&app);
    press(&mut app, Tool::Rectangle, [10.0, 10.0], [10.0, 10.0]);
    let id = only_dialog(&app);
    app.ui.close_dialog(id);
    assert_eq!(layers(&app), before);

    press(&mut app, Tool::Rectangle, [10.0, 10.0], [10.0, 10.0]);
    let id = only_dialog(&app);
    set(&mut app, id, json!({"width": 33, "height": 44, "fromCenter": true}));
    crate::dialogs::confirm(&mut app, id).unwrap();
    press(&mut app, Tool::Rectangle, [50.0, 50.0], [50.0, 50.0]);
    let f = &app.ui.dialogs[0].fields;
    assert_eq!((f["width"].as_f64(), f["height"].as_f64(), f["fromCenter"].as_bool()), (Some(33.0), Some(44.0), Some(true)));
    // Each tool remembers its own values.
    let id = only_dialog(&app);
    app.ui.close_dialog(id);
    press(&mut app, Tool::EllipseShape, [50.0, 50.0], [50.0, 50.0]);
    assert_eq!(app.ui.dialogs[0].fields["width"].as_f64(), Some(100.0));
}

#[test]
fn bad_values_are_errors_not_panics() {
    let mut app = app();
    for bad in [
        json!({"width": 0}),
        json!({"width": -5}),
        json!({"height": 1e12}),
        json!({"width": "wide"}),
        json!({"width": null}),
        json!({"__at": "here"}),
        json!({"__at": [1]}),
        json!({"__at": [1e308, 1e308], "fromCenter": true, "width": 300000, "height": 300000}),
        json!({"radii": "round"}),
        json!({"radii": [-1, 1e300, "x", null, 5]}),
    ] {
        let before = layers(&app);
        press(&mut app, Tool::Rectangle, [10.0, 10.0], [10.0, 10.0]);
        let id = only_dialog(&app);
        set(&mut app, id, bad.clone());
        let r = crate::dialogs::confirm(&mut app, id);
        if r.is_ok() {
            // Odd radii are clamped, not rejected.
            assert!(bad.get("radii").is_some(), "{bad} should be rejected");
            app.run("edit.undo", json!({})).unwrap();
        } else {
            assert!(app.ui.status_error);
        }
        assert!(app.ui.dialogs.is_empty());
        assert_eq!(layers(&app), before, "{bad}");
    }
    // Bad remembered values fall back to the defaults.
    app.session.prefs.edit(|p| p.dialogs.insert("shape.create.rectangle".into(), json!({"width": -1, "height": "x", "fromCenter": 3})));
    press(&mut app, Tool::Rectangle, [10.0, 10.0], [10.0, 10.0]);
    let f = &app.ui.dialogs[0].fields;
    assert_eq!((f["width"].as_f64(), f["height"].as_f64(), f["fromCenter"].as_bool()), (Some(100.0), Some(100.0), Some(false)));
}

#[test]
fn agents_click_set_and_confirm_over_the_control_channel() {
    let mut app = app();
    let ctx = egui::Context::default();
    let call = |app: &mut PhotocraftApp, method: &str, params: Value| {
        let (req, _rx) = crate::control::ControlRequest::new(method, params);
        match crate::control::handle(app, &ctx, &req) {
            crate::control::Outcome::Done(v) => v,
            _ => panic!("deferred"),
        }
    };
    call(&mut app, "ui.pointer", json!({"tool": "rectangle", "events": [{"kind": "down", "x": 40, "y": 50}, {"kind": "up", "x": 40, "y": 50}]}));
    let id = only_dialog(&app);
    for (field, value) in [("width", json!(64)), ("height", json!(32))] {
        call(&mut app, "ui.dialog.set", json!({"dialog": id, "field": field, "value": value}));
    }
    let r = call(&mut app, "ui.dialog.confirm", json!({"dialog": id}));
    assert_eq!(r["ok"], json!(true), "{r}");
    assert_eq!(live(&app), LiveShape::Rect { rect: [40.0, 50.0, 64.0, 32.0], radii: [0.0; 4] });
}

#[test]
fn the_dialog_body_renders_for_every_tool_and_unit() {
    let mut app = app();
    for (tool, unit) in [(Tool::Rectangle, "pixels"), (Tool::EllipseShape, "mm"), (Tool::Polygon, "percent"), (Tool::Triangle, "inches")] {
        app.run("prefs.set", json!({"path": "unitsAndRulers.rulers", "value": unit})).unwrap();
        let id = open(&mut app, tool, [1.0, 2.0]).unwrap();
        let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
        let ctx = egui::Context::default();
        ctx.run_ui(egui::RawInput::default(), |ui| body(&app, ui, &mut fields)).textures_delta.clear();
        app.ui.close_dialog(id);
    }
}
