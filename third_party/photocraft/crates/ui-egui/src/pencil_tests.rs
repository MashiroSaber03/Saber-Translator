//! The Pencil tool (#213): aliased strokes through the live-stroke path, Auto Erase, the square
//! pixel-grid cursor, ⇧-click lines, Control+Alt resizing and the B group.

use egui::{Key, Modifiers, Pos2, Rect, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform, pencil_cursor_rect, pencil_tip_is_round, tool_event};
use crate::state::Tool;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 120, "height": 80, "background": "transparent"})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#000000", "background": "#ffffff"})).unwrap();
    app.run("tools.setBrush", json!({"brush": {"size": 3, "hardness": 0.0}})).unwrap();
    // Each tool keeps its own brush (#218): switch first, then set the Pencil's options, as the
    // options bar does (its first frame takes the Brush's tip at Photoshop's 10 % smoothing).
    app.ui.tool = Tool::Pencil;
    crate::paint_mouse::sync_tool_brush(&mut app);
    app.run("tools.setBrush", json!({"brush": {"smoothing": {"amount": 0.0}}})).unwrap();
    app
}

fn layer(app: &PhotocraftApp) -> photocraft_engine::doc::Layer {
    let st = app.session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().clone()
}

fn rgba(app: &PhotocraftApp, x: i32, y: i32) -> [f32; 4] {
    layer(app).surface().unwrap().rgba(x, y)
}

fn drag(app: &mut PhotocraftApp, pts: &[(f64, f64)], mods: Modifiers) {
    let (x, y) = pts[0];
    tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, mods);
    for &(x, y) in &pts[1..] {
        tool_event(app, ToolEvent::Move { x, y, pressure: 1.0 }, mods);
    }
    let &(x, y) = pts.last().unwrap();
    tool_event(app, ToolEvent::Up { x, y }, mods);
}

fn last(app: &PhotocraftApp) -> (String, serde_json::Value) {
    app.session.journal.last().cloned().unwrap()
}

#[test]
fn pencil_strokes_are_aliased_live_and_one_undo_step() {
    let mut app = app();
    let pts = [(10.0, 10.0), (40.3, 22.7), (70.0, 50.0), (100.0, 30.0)];
    let (x, y) = pts[0];
    tool_event(&mut app, ToolEvent::Down { x, y, pressure: 1.0 }, Modifiers::NONE);
    for &(x, y) in &pts[1..] {
        tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, Modifiers::NONE);
    }
    // The canvas shows the engine's real dabs while dragging.
    assert!(app.live_stroke.is_some(), "the Pencil strokes live");
    tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 30.0 }, Modifiers::NONE);
    let (id, p) = last(&app);
    assert_eq!(id, "paint.pencil");
    assert_eq!(p["autoErase"], json!(false));
    assert_eq!(app.session.active().unwrap().history.past_len(), 1);
    // 8 bits, a soft session brush: still no partial alpha anywhere.
    let s = layer(&app).surface().unwrap().clone();
    let mut painted = 0;
    for y in 0..80 {
        for x in 0..120 {
            let a = s.rgba(x, y)[3];
            assert!(a == 0.0 || a == 1.0, "partial alpha {a} at ({x},{y})");
            painted += usize::from(a == 1.0);
        }
    }
    assert!(painted > 200, "{painted}");
    assert_eq!(rgba(&app, 70, 50), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn auto_erase_paints_the_background_colour_from_foreground_pixels() {
    let mut app = app();
    drag(&mut app, &[(10.0, 40.0), (110.0, 40.0)], Modifiers::NONE);
    app.ui.tool_options.pencil_auto_erase = true;
    // Starting on the black line: white.
    drag(&mut app, &[(50.0, 40.0), (50.0, 70.0)], Modifiers::NONE);
    assert_eq!(rgba(&app, 50, 60), [1.0; 4]);
    assert_eq!(last(&app).1["autoErase"], json!(true));
    // Starting off it: black.
    drag(&mut app, &[(80.0, 10.0), (80.0, 30.0)], Modifiers::NONE);
    assert_eq!(rgba(&app, 80, 20), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn shift_click_draws_a_pencil_line() {
    let mut app = app();
    drag(&mut app, &[(10.0, 10.0), (20.0, 10.0)], Modifiers::NONE);
    drag(&mut app, &[(100.0, 60.0)], Modifiers::SHIFT);
    let (id, p) = last(&app);
    assert_eq!(id, "paint.pencil");
    assert_eq!(p["points"].as_array().unwrap().len(), 2, "{p}");
    assert_eq!(rgba(&app, 60, 35)[3], 1.0);
    assert_eq!(app.session.active().unwrap().history.past_len(), 2);
}

#[test]
fn ctrl_alt_drag_resizes_the_pencil_without_painting() {
    let mut app = app();
    let ctrl_alt = Modifiers { alt: true, ctrl: true, ..Default::default() };
    assert!(crate::brush_resize::applies(Tool::Pencil));
    drag(&mut app, &[(50.0, 40.0), (60.0, 40.0)], ctrl_alt);
    assert_eq!(app.session.tools.brush.size, 23.0);
    assert_eq!(app.session.active().unwrap().history.past_len(), 0);
}

#[test]
fn square_cursor_sits_on_the_pixel_grid() {
    let rect = Rect::from_min_size(Pos2::new(0.0, 0.0), vec2(800.0, 600.0));
    let xf = ViewXform { rect, zoom: 8.0, center: [50.0, 37.5], flip: false, rotation: 0.0, aspect: 1.0 };
    // 1 px at 800 %: the 8-point square of the pixel under the pointer.
    let r = pencil_cursor_rect(&xf, [10.3, 5.7], 1.0, 1.0);
    assert_eq!(r, Rect::from_two_pos(xf.to_screen(10.0, 5.0), xf.to_screen(11.0, 6.0)));
    assert_eq!(r.size(), vec2(8.0, 8.0));
    // Even sizes centre on the nearest pixel corner; odd ones on the pixel.
    let r = pencil_cursor_rect(&xf, [10.3, 5.7], 4.0, 1.0);
    assert_eq!(r, Rect::from_two_pos(xf.to_screen(8.0, 4.0), xf.to_screen(12.0, 8.0)));
    let r = pencil_cursor_rect(&xf, [10.9, 5.1], 3.0, 1.0);
    assert_eq!(r, Rect::from_two_pos(xf.to_screen(9.0, 4.0), xf.to_screen(12.0, 7.0)));
    // At 2× and at an odd zoom the edges land on physical pixels.
    for (zoom, ppp) in [(3.3, 2.0), (0.5, 2.0), (1.0, 1.0), (13.7, 1.5)] {
        let xf = ViewXform {
            rect: Rect::from_min_size(Pos2::new(0.3, 0.7), vec2(800.0, 600.0)),
            zoom,
            center: [50.2, 37.9],
            flip: true,
            rotation: 0.0,
            aspect: 1.0,
        };
        let r = pencil_cursor_rect(&xf, [20.4, 30.6], 5.0, ppp);
        for v in [r.min.x, r.min.y, r.max.x, r.max.y] {
            assert!((v * ppp - (v * ppp).round()).abs() < 1e-3, "zoom {zoom} ppp {ppp}: {v}");
        }
        assert!((r.width() - 5.0 * zoom).abs() <= 1.0 / ppp + 1e-3, "{r:?}");
    }
    // Hostile numbers don't panic.
    let _ = pencil_cursor_rect(&xf, [f64::NAN, f64::INFINITY], f32::NAN, f32::NAN);
}

#[test]
fn the_cursor_is_round_exactly_when_the_pencil_dab_is() {
    // #2662: one dab per size; the cursor is the square only while the dab fills its corners.
    let mut app = app();
    for n in 1..=12u32 {
        let (x, y) = (8.0 + f64::from((n - 1) % 6) * 18.0 + 0.3, 15.0 + f64::from((n - 1) / 6) * 30.0 + 0.6);
        app.run("paint.pencil", json!({"points": [[x, y]], "size": n})).unwrap();
        let [x0, y0, ..] = photocraft_engine::paint::grid_square(x, y, n as f32);
        let corner = rgba(&app, x0 as i32, y0 as i32)[3] == 1.0;
        assert_eq!(pencil_tip_is_round(n as f32), !corner, "{n} px");
    }
    assert!(pencil_tip_is_round(9.0) && !pencil_tip_is_round(3.0));
    assert!(!pencil_tip_is_round(f32::NAN));
}

#[test]
fn the_eraser_in_pencil_mode_erases_aliased_pixels() {
    // #2662: an opaque layer, then a soft Eraser in Pencil mode erases with hard edges.
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[60.0, 40.0]], "size": 400})).unwrap();
    app.ui.tool = Tool::Eraser;
    crate::paint_mouse::sync_tool_brush(&mut app);
    app.run("tools.setBrush", json!({"brush": {"size": 15, "hardness": 0.0, "smoothing": {"amount": 0.0}}})).unwrap();
    app.ui.tool_options.eraser_mode = "pencil".into();
    drag(&mut app, &[(10.0, 40.0), (60.3, 25.7), (110.0, 40.0)], Modifiers::NONE);
    let (id, p) = last(&app);
    assert_eq!((id.as_str(), &p["erase"]), ("paint.pencil", &json!(true)));
    let s = layer(&app).surface().unwrap().clone();
    let mut erased = 0;
    for y in 0..80 {
        for x in 0..120 {
            let a = s.rgba(x, y)[3];
            assert!(a == 0.0 || a == 1.0, "partial alpha {a} at ({x},{y})");
            erased += usize::from(a == 0.0);
        }
    }
    assert!(erased > 500, "{erased}");
}

#[test]
fn the_eraser_in_block_mode_erases_a_16_screen_px_square() {
    // #2770: a soft, small, half-opacity Eraser in Block mode still clears a hard square of
    // 16 screen pixels: 16 document pixels at 100 %, 8 at 200 %.
    for (zoom, n) in [(1.0, 16usize), (2.0, 8)] {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[60.0, 40.0]], "size": 400})).unwrap();
        app.ui.tool = Tool::Eraser;
        crate::paint_mouse::sync_tool_brush(&mut app);
        app.run("tools.setBrush", json!({"brush": {"size": 5, "hardness": 0.0, "opacity": 0.5, "flow": 0.3, "smoothing": {"amount": 0.0}}})).unwrap();
        app.ui.tool_options.eraser_mode = "block".into();
        app.ui.views[0].zoom = zoom;
        drag(&mut app, &[(60.3, 40.6)], Modifiers::NONE);
        let (id, p) = last(&app);
        assert_eq!((id.as_str(), &p["block"]), ("paint.pencil", &json!(true)));
        let s = layer(&app).surface().unwrap().clone();
        let (mut erased, mut x0, mut y0, mut x1, mut y1) = (0, i32::MAX, i32::MAX, 0, 0);
        for y in 0..80 {
            for x in 0..120 {
                let a = s.rgba(x, y)[3];
                assert!(a == 0.0 || a == 1.0, "partial alpha {a} at ({x},{y}), zoom {zoom}");
                if a == 0.0 {
                    erased += 1;
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
                }
            }
        }
        assert_eq!((erased, x1 - x0 + 1, y1 - y0 + 1), (n * n, n as i32, n as i32), "zoom {zoom}");
    }
}

#[test]
fn b_cycles_brush_pencil_mixer_brush_and_color_replacement() {
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
        app
    });
    h.state_mut().ui.tool = Tool::Move;
    h.state_mut().session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = false);
    h.run_steps(4);
    let press = |h: &mut Harness<'_, PhotocraftApp>, m: Modifiers| {
        h.event(egui::Event::Key { key: Key::B, physical_key: None, pressed: true, repeat: false, modifiers: m });
        h.event(egui::Event::Key { key: Key::B, physical_key: None, pressed: false, repeat: false, modifiers: m });
        h.run_steps(2);
    };
    press(&mut h, Modifiers::NONE);
    assert_eq!(h.state().ui.tool, Tool::Brush);
    press(&mut h, Modifiers::NONE);
    assert_eq!(h.state().ui.tool, Tool::Pencil);
    press(&mut h, Modifiers::NONE);
    assert_eq!(h.state().ui.tool, Tool::MixerBrush);
    press(&mut h, Modifiers::NONE);
    assert_eq!(h.state().ui.tool, Tool::ColorReplacement);
    // Use Shift Key for Tool Switch: ⇧B cycles, B keeps the group's tool.
    h.state_mut().session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = true);
    press(&mut h, Modifiers::SHIFT);
    assert_eq!(h.state().ui.tool, Tool::Brush);
    press(&mut h, Modifiers::SHIFT);
    assert_eq!(h.state().ui.tool, Tool::Pencil);
    press(&mut h, Modifiers::SHIFT);
    assert_eq!(h.state().ui.tool, Tool::MixerBrush);
    press(&mut h, Modifiers::SHIFT);
    assert_eq!(h.state().ui.tool, Tool::ColorReplacement);
    press(&mut h, Modifiers::NONE);
    assert_eq!(h.state().ui.tool, Tool::ColorReplacement);
    assert_eq!(Tool::from_name("pencil"), Some(Tool::Pencil));
    assert_eq!(Tool::from_name("mixerBrush"), Some(Tool::MixerBrush));
}
