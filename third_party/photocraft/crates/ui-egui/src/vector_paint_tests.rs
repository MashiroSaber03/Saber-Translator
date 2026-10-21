//! Pointer, keyboard, colour precision and history regressions for shape appearance controls.
use super::*;
use egui::{Event, Key, Modifiers, PointerButton};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_doc::ShapeLayer;

fn app(mode: &str, depth: u32) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 120, "height": 100, "mode": mode, "depth": depth})).unwrap();
    app.ui.tool = Tool::Rectangle;
    app.sync_views();
    app
}

fn shape(app: &PhotocraftApp) -> &ShapeLayer {
    let st = app.session.active().unwrap();
    let LayerContent::Shape(shape) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("expected editable shape") };
    shape
}

fn assert_saved_path(saved: &Path, original: &Path) {
    assert_eq!((saved.fill_rule, saved.inverted), (original.fill_rule, original.inverted));
    assert_eq!(saved.subpaths.len(), original.subpaths.len());
    for (saved, original) in saved.subpaths.iter().zip(&original.subpaths) {
        assert_eq!((saved.closed, saved.op, saved.knots.len()), (original.closed, original.op, original.knots.len()));
        for (saved, original) in saved.knots.iter().zip(&original.knots) {
            assert_eq!(saved.smooth, original.smooth);
            for (saved, original) in [(saved.anchor, original.anchor), (saved.in_ctrl, original.in_ctrl), (saved.out_ctrl, original.out_ctrl)] {
                // Native manifests use JSON f64 coordinates, whose existing round trip may
                // differ by one ulp for generated ellipse/polygon control points.
                assert!((saved.x - original.x).abs() <= 1e-10 && (saved.y - original.y).abs() <= 1e-10, "{saved:?} differs from {original:?}");
            }
        }
    }
}

fn harness(app: PhotocraftApp, properties: bool) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            crate::screen_picker::tick(app, ui.ctx());
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            if properties {
                let id = app.session.active().unwrap().active_layer.unwrap();
                ui.set_max_width(320.0);
                shape_properties(app, ui, id);
            } else {
                let tool = app.ui.tool;
                ui.horizontal(|ui| options_bar(app, ui, tool));
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
    h.run_steps(3);
    h
}

fn pointer(h: &mut Harness<'_, PhotocraftApp>, pos: Pos2, pressed: bool) {
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    h.step();
}

#[test]
fn fill_and_stroke_swatches_open_with_pointer_and_keyboard_without_changing_paint() {
    for label in ["Fill color", "Stroke color"] {
        let mut app = app("rgb", 32);
        let foreground = [0.123456, 0.345678, 0.567891, 0.372819];
        let background = [0.891234, 0.678912, 0.456789, 0.0];
        app.run("tools.setColors", json!({"foreground": foreground, "background": background})).unwrap();
        let document = app.session.active().unwrap().doc.clone();
        let mut h = harness(app, false);
        h.get_by_label(label).click();
        h.run_steps(4);
        assert!(h.query_by_label("No Color").is_some(), "{label} opens its picker");
        assert_eq!(h.state().session.tools.foreground, foreground);
        assert_eq!(h.state().session.tools.background, background);
        assert_eq!(h.state().ui.tool_options.stroke_width, 0.0);
        assert_eq!(h.state().session.active().unwrap().doc, document);
        h.key_press(Key::Escape);
        h.run_steps(2);
        assert!(h.query_by_label("No Color").is_none());
        h.get_by_label(label).focus();
        h.run_steps(2);
        h.key_press(Key::Enter);
        h.run_steps(3);
        assert!(h.query_by_label("No Color").is_some(), "{label} opens from keyboard focus");
        h.key_press(Key::Escape);
        h.run_steps(2);
        assert_eq!(h.state().session.active().unwrap().doc, document);
        assert_eq!(h.state().session.tools.foreground, foreground);
        assert_eq!(h.state().session.tools.background, background);
    }
}

#[test]
fn picking_tool_colours_enables_paint_and_reaches_the_next_shape() {
    let mut app = app("rgb", 32);
    let mut picks = std::collections::VecDeque::from([[0.91, 0.17, 0.32], [0.12, 0.68, 0.93]]);
    app.services.screen_pick = Some(Box::new(move |_| {
        let (tx, receiver) = std::sync::mpsc::channel();
        tx.send(Ok(crate::screen_picker::Capture::Color(picks.pop_front()))).unwrap();
        crate::screen_picker::Pending { receiver, cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) }
    }));
    app.ui.tool_options.shape_fill = false;
    let mut h = harness(app, false);
    for label in ["Fill color", "Stroke color"] {
        h.get_by_label(label).click();
        h.run_steps(3);
        h.get_by_label("Pick screen color").click();
        h.run_steps(4);
        h.key_press(Key::Escape);
        h.run_steps(2);
    }
    assert!(h.state().ui.tool_options.shape_fill);
    assert_eq!(h.state().ui.tool_options.stroke_width, 3.0);
    assert_eq!(h.state().session.tools.foreground, [0.91, 0.17, 0.32, 1.0]);
    assert_eq!(h.state().session.tools.background, [0.12, 0.68, 0.93, 1.0]);
    finish_shape(h.state_mut(), Tool::Rectangle, [20.0, 20.0], [100.0, 80.0], Modifiers::NONE);
    assert_eq!(shape(h.state()).fill, Some(solid_paint([0.91, 0.17, 0.32, 1.0])));
    assert_eq!(shape(h.state()).stroke.as_ref().unwrap().paint, solid_paint([0.12, 0.68, 0.93, 1.0]));
}

#[test]
fn no_colour_and_solid_choices_are_explicit_live_tool_defaults() {
    let mut h = harness(app("rgb", 8), false);
    h.get_by_label("Fill color").click();
    h.run_steps(3);
    h.get_by_label("No Color").click();
    h.run_steps(2);
    assert!(!h.state().ui.tool_options.shape_fill);
    h.get_by_label("Solid").click();
    h.run_steps(2);
    assert!(h.state().ui.tool_options.shape_fill);
    h.key_press(Key::Escape);
    h.run_steps(2);
    h.get_by_label("Stroke color").click();
    h.run_steps(3);
    h.get_by_label("Solid").click();
    h.run_steps(2);
    assert_eq!(h.state().ui.tool_options.stroke_width, 3.0, "choosing unchanged white enables a stroke");
    h.get_by_label("No Color").click();
    h.run_steps(2);
    assert_eq!(h.state().ui.tool_options.stroke_width, 0.0);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), 0, "tool defaults do not edit the document");
}

#[test]
fn float_paints_match_preview_creation_native_roundtrip_and_document_depth() {
    let foreground = [0.123456, 0.345678, 0.567891, 0.372819];
    let background = [0.891234, 0.234567, 0.456789, 0.618237];
    for (mode, depth) in [("rgb", 8), ("rgb", 16), ("rgb", 32), ("gray", 16), ("cmyk", 8), ("lab", 16)] {
        for tool in [Tool::Rectangle, Tool::Triangle, Tool::EllipseShape] {
            let mut app = app(mode, depth);
            set_tool_paint(&mut app, false, PaintChange::Color(foreground)).unwrap();
            set_tool_paint(&mut app, true, PaintChange::Color(background)).unwrap();
            app.ui.tool_options.shape_stroke.cap = photocraft_doc::LineCap::Round;
            app.ui.tool_options.shape_stroke.dashes = vec![0.0, 2.0];
            let preview = shape_preview(&app, tool, [20.0, 20.0], [100.0, 80.0], Modifiers::NONE).unwrap();
            finish_shape(&mut app, tool, [20.0, 20.0], [100.0, 80.0], Modifiers::NONE);
            let committed = shape(&app).clone();
            assert_eq!(preview.fill, committed.fill);
            assert_eq!(preview.stroke, committed.stroke);
            assert_eq!(preview.path, committed.path);
            assert_eq!(committed.fill, Some(solid_paint(foreground)));
            assert_eq!(committed.stroke.as_ref().unwrap().paint, solid_paint(background));
            let doc = &app.session.active().unwrap().doc;
            let rendered = photocraft_vector::render_shape(&preview, doc.pixel_format(), doc.bounds());
            assert_eq!(committed.cache.as_ref().unwrap(), &rendered, "{tool:?} {mode}/{depth}");
            let bytes = photocraft_format::save_to_bytes(doc, &Default::default()).unwrap();
            let loaded = photocraft_format::load_from_bytes(&bytes).unwrap();
            let id = app.session.active().unwrap().active_layer.unwrap();
            let LayerContent::Shape(saved) = &loaded.layer(id).unwrap().content else { panic!("saved editable shape") };
            assert_eq!((&saved.fill, &saved.stroke), (&committed.fill, &committed.stroke));
            assert_saved_path(&saved.path, &committed.path);
            app.run("edit.undo", json!({})).unwrap();
            app.run("edit.redo", json!({})).unwrap();
            assert_eq!(shape(&app), &committed);
        }
    }
}

#[test]
fn transparent_and_absent_paints_remain_distinct_and_open_pen_keeps_foreground() {
    let mut app = app("rgb", 32);
    let transparent = [0.124578, 0.564321, 0.765432, 0.0];
    set_tool_paint(&mut app, false, PaintChange::Color(transparent)).unwrap();
    set_tool_paint(&mut app, true, PaintChange::Color(transparent)).unwrap();
    let preview = shape_preview(&app, Tool::Rectangle, [20.0, 20.0], [80.0, 70.0], Modifiers::NONE).unwrap();
    assert_eq!(preview.fill, Some(solid_paint(transparent)));
    assert_eq!(preview.stroke.as_ref().unwrap().paint, solid_paint(transparent));
    finish_shape(&mut app, Tool::Rectangle, [20.0, 20.0], [80.0, 70.0], Modifiers::NONE);
    assert_eq!(shape(&app).fill, preview.fill);
    assert_eq!(shape(&app).stroke, preview.stroke);
    assert_eq!(shape(&app).cache.as_ref().unwrap().pixel(40, 40), vec![0.0; 4]);
    set_tool_paint(&mut app, false, PaintChange::None).unwrap();
    set_tool_paint(&mut app, true, PaintChange::None).unwrap();
    finish_shape(&mut app, Tool::Triangle, [20.0, 20.0], [80.0, 70.0], Modifiers::NONE);
    assert!(shape(&app).fill.is_none() && shape(&app).stroke.is_none());
    let foreground = [0.836521, 0.125468, 0.745632, 0.438271];
    app.run("tools.setColors", json!({"foreground": foreground, "background": [0.0, 1.0, 0.0, 1.0]})).unwrap();
    app.ui.tool_options.vector_mode = "shape".into();
    app.ui.pen = Some(PenPath { knots: vec![[[20.0, 20.0]; 3], [[80.0, 60.0]; 3]], ..Default::default() });
    pen_commit(&mut app, false);
    assert!(shape(&app).fill.is_none());
    assert!(!shape(&app).path.subpaths[0].closed);
    assert_eq!(shape(&app).stroke.as_ref().unwrap().paint, solid_paint(foreground));
}

#[test]
fn selected_shape_style_is_available_when_absent_and_dismissal_does_not_enable_it() {
    let mut app = app("rgb", 8);
    app.run("shape.create", json!({"rect": [20,20,80,60], "stroke": null})).unwrap();
    let doc = app.session.active().unwrap().doc.clone();
    let history = app.session.active().unwrap().history.past_len();
    let mut h = harness(app, true);
    h.get_by_label("Stroke Options").click();
    h.run_steps(3);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().session.active().unwrap().doc, doc);
    h.get_by_label("Stroke Options").click();
    h.run_steps(3);
    h.get_by_role_and_label(egui::accesskit::Role::RadioButton, "Dotted").click();
    h.run_steps(3);
    assert_eq!(shape(h.state()).stroke.as_ref().unwrap().width, 3.0);
    assert_eq!(shape(h.state()).stroke.as_ref().unwrap().dashes, [0.0, 2.0]);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
    h.key_press(Key::Escape);
    h.run_steps(2);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert!(shape(h.state()).stroke.is_none());
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(shape(h.state()).stroke.as_ref().unwrap().dashes, [0.0, 2.0]);
}

fn popup_rect(h: &Harness<'_, PhotocraftApp>) -> Rect {
    h.ctx
        .memory(|m| m.areas().visible_layer_ids().into_iter().filter(|l| l.order == egui::Order::Foreground).find_map(|l| m.area_rect(l.id)))
        .expect("colour picker is open")
}

#[test]
fn selected_shape_colour_drags_apply_live_coalesce_per_gesture_and_keep_tool_defaults() {
    let mut app = app("rgb", 32);
    app.run("shape.create", json!({"rect": [20,20,80,60], "fill": "#3270c0", "stroke": {"width": 5, "color": "#ee3355"}})).unwrap();
    let original = shape(&app).clone();
    let defaults = (app.session.tools.foreground, app.session.tools.background, app.ui.tool_options.clone());
    let history = app.session.active().unwrap().history.past_len();
    let mut h = harness(app, true);
    h.get_by_label("Set shape fill type").click();
    h.run_steps(4);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    let rect = popup_rect(&h);
    let start = rect.min + vec2(55.0, rect.height() * 0.5);
    h.hover_at(start);
    h.step();
    pointer(&mut h, start, true);
    for offset in [10.0, 20.0, 30.0] {
        h.event(Event::PointerMoved(start + vec2(offset, 6.0)));
        h.step();
    }
    pointer(&mut h, start + vec2(30.0, 6.0), false);
    h.run_steps(2);
    let first = shape(h.state()).clone();
    assert_ne!(first.fill, original.fill);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
    let next = start + vec2(60.0, 15.0);
    h.hover_at(next);
    h.step();
    pointer(&mut h, next, true);
    h.event(Event::PointerMoved(next + vec2(12.0, 4.0)));
    h.step();
    pointer(&mut h, next + vec2(12.0, 4.0), false);
    h.run_steps(2);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 2);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!((h.state().session.tools.foreground, h.state().session.tools.background, h.state().ui.tool_options.clone()), defaults);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(shape(h.state()).fill, first.fill);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(shape(h.state()).fill, original.fill);
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(shape(h.state()).fill, first.fill);
}

/// #3997, #3999: after drawing a shape, the options bar's Fill swatch recolours it, as Stroke
/// does (in v0.7.0 the Fill swatch was only a preview of the foreground colour).
#[test]
fn options_bar_fill_swatch_recolours_the_shape_just_drawn() {
    let mut app = app("rgb", 8);
    app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    app.ui.tool_options.stroke_width = 3.0;
    finish_shape(&mut app, Tool::Rectangle, [20.0, 20.0], [100.0, 80.0], Modifiers::NONE);
    let original = shape(&app).clone();
    let history = app.session.active().unwrap().history.past_len();
    let mut h = harness(app, false);
    h.get_by_label("Fill color").click();
    h.run_steps(4);
    let rect = popup_rect(&h);
    let start = rect.min + vec2(55.0, rect.height() * 0.5);
    h.hover_at(start);
    h.step();
    pointer(&mut h, start, true);
    h.event(Event::PointerMoved(start + vec2(30.0, 6.0)));
    h.step();
    pointer(&mut h, start + vec2(30.0, 6.0), false);
    h.run_steps(2);
    let edited = shape(h.state()).clone();
    assert_ne!(edited.fill, original.fill, "the Fill swatch recoloured the shape");
    assert_eq!(edited.fill.as_ref().and_then(color_of), Some(h.state().session.tools.foreground), "and the next shape's fill");
    assert_eq!(edited.stroke, original.stroke, "the stroke is untouched");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(shape(h.state()).fill, original.fill);
}
