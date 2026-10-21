//! Type tool through the real canvas (#123, #124): clicks and drags at known glyph positions
//! place the caret and selection on the glyph under the pointer at any zoom, HiDPI scale, pan,
//! view flip and layer transform; size edits apply at layer and selection scope, for native and
//! PSD-round-tripped text, as one history step per drag.

use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::{LayerContent, LayerId, TextLayer};
use photocraft_geom::{Affine, Point};
use serde_json::json;

use super::{layout, text_layer};
use crate::PhotocraftApp;
use crate::canvas::ViewXform;

#[test]
fn focused_font_search_does_not_type_into_the_canvas_layer() {
    let mut app = new_app();
    let id = app.run("type.create", json!({"text": "Sample", "size": 24})).unwrap()["layer"].as_u64().unwrap();
    app.ui.text_edit = Some(crate::state::TextEdit {
        layer: id,
        caret: 6,
        anchor: 6,
        session: "s".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    let mut h = Harness::builder().build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let search_id = egui::Id::new("font-search-regression");
            let mut query = ui.data(|d| d.get_temp::<String>(search_id)).unwrap_or_default();
            ui.add(egui::TextEdit::singleline(&mut query).id(search_id)).request_focus();
            ui.data_mut(|d| d.insert_temp(search_id, query));
            super::handle_keys(app, ui.ctx());
        },
        app,
    );
    h.run_steps(2);
    h.event(egui::Event::Text("Ins".into()));
    h.run();
    assert_eq!(h.ctx.data(|d| d.get_temp::<String>(egui::Id::new("font-search-regression"))).as_deref(), Some("Ins"));
    assert_eq!(super::current_text(h.state_mut(), LayerId(id)).as_deref(), Some("Sample"));
}

#[test]
fn numeric_type_fields_have_localized_accessible_names() {
    let pt = crate::i18n::Lang::from_code("pt-br").unwrap();
    let mut h = Harness::builder().build_ui_state(
        move |ui, value: &mut f32| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::i18n::with_language(pt, || {
                if let Some(next) = super::num_field(ui, "tT", "Font size", *value, 1.0..=999.0, "pt", 80.0) {
                    *value = next;
                }
            });
        },
        24.0,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h.get_by_role_and_label(egui::accesskit::Role::SpinButton, "Tamanho da fonte").click();
    h.run();
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
    h.event(egui::Event::Text("36".into()));
    h.key_press(egui::Key::Enter);
    h.run();
    assert_eq!(*h.state(), 36.0);
}

fn harness(ppp: f32, app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    make_harness(ppp, app, false)
}

fn make_harness(ppp: f32, app: PhotocraftApp, render: bool) -> Harness<'static, PhotocraftApp> {
    make_harness_with(Harness::builder(), ppp, app, render)
}

/// 60 frames a second: every queued event is a frame, and the default quarter-second step puts
/// the second click of a double-click past egui's double-click delay.
fn fast_harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    make_harness_with(Harness::builder().with_step_dt(1.0 / 60.0), 1.0, app, false)
}

fn make_harness_with(builder: egui_kittest::HarnessBuilder<PhotocraftApp>, ppp: f32, app: PhotocraftApp, render: bool) -> Harness<'static, PhotocraftApp> {
    let builder = builder.with_size(vec2(1200.0, 800.0)).with_pixels_per_point(ppp);
    let builder = if render { builder.wgpu() } else { builder };
    let mut h = builder.build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

fn new_app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 1600, "height": 900})).unwrap();
    app.sync_views();
    app.ui.extras.rulers = false;
    app.ui.tool = crate::state::Tool::Type;
    app
}

fn xf(app: &PhotocraftApp) -> ViewXform {
    let v = &app.ui.views[0];
    ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom / app.canvas_ppp(),
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    }
}

/// Screen position of a text-space point of layer `id`.
fn screen(h: &mut Harness<'static, PhotocraftApp>, id: LayerId, x: f32, y: f32) -> Pos2 {
    let app = h.state_mut();
    let (_, aff, _) = layout(app, id).unwrap();
    let p = aff.apply(Point::new(f64::from(x), f64::from(y)));
    xf(app).to_screen(p.x as f32, p.y as f32)
}

/// A point inside glyph `i` (character index): `f` of the way along its advance, in the upper
/// half of its line.
fn glyph(h: &mut Harness<'static, PhotocraftApp>, id: LayerId, i: usize, f: f32) -> Pos2 {
    let (l, _, text) = layout(h.state_mut(), id).unwrap();
    let b = text.char_indices().nth(i).unwrap().0;
    let c = l.clusters.iter().find(|c| c.range.start == b).unwrap().clone();
    let ln = &l.lines[c.line];
    screen(h, id, c.x + c.advance * f, ln.baseline - ln.ascent * 0.35)
}

fn press(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: Modifiers::NONE });
    h.run_steps(1);
}

fn click(h: &mut Harness<'static, PhotocraftApp>, p: Pos2) {
    h.hover_at(p);
    h.run_steps(1);
    press(h, p, true);
    press(h, p, false);
    h.run_steps(1);
}

/// Two clicks within egui's double-click delay (use [`fast_harness`]).
fn double_click(h: &mut Harness<'static, PhotocraftApp>, p: Pos2) {
    h.hover_at(p);
    h.run_steps(1);
    for pressed in [true, false, true, false] {
        h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    }
    h.run_steps(2);
}

fn drag(h: &mut Harness<'static, PhotocraftApp>, a: Pos2, b: Pos2) {
    h.hover_at(a);
    h.run_steps(1);
    press(h, a, true);
    for k in 1..=6 {
        h.hover_at(a + (b - a) * (k as f32 / 6.0));
        h.run_steps(1);
    }
    press(h, b, false);
    h.run_steps(1);
}

fn selection(h: &Harness<'static, PhotocraftApp>) -> (usize, usize) {
    let Some(e) = h.state().ui.text_edit.clone() else { return (usize::MAX, usize::MAX) };
    (e.anchor, e.caret)
}

/// Every combination the issue lists that this canvas has: zoom 25–200 %, HiDPI 1×/2×, a panned
/// view, View › Flip Horizontal, and point / paragraph text with identity or rotated+scaled layer
/// transforms. Clicks land before/after the glyph under the pointer; drags select from the glyph
/// pressed (not where egui recognised the drag) to the glyph released over.
#[test]
fn clicks_and_drags_land_on_the_glyph_under_the_pointer() {
    let rotated = Affine { m: [1.4 * 0.94, 1.4 * 0.342, -1.4 * 0.342, 1.4 * 0.94, 500.0, 420.0] };
    for ppp in [1.0, 2.0] {
        for zoom in [0.25f32, 0.5, 1.0, 2.0] {
            for (flip, pan) in [(false, [0.0, 0.0]), (true, [37.0, -21.0])] {
                for (shape, tf) in [("point", None), ("box", None), ("point", Some(rotated)), ("box", Some(rotated))] {
                    let mut app = new_app();
                    let size = 60.0 / zoom;
                    let mut p = json!({"text": "HOHOHO HOHO", "size": size, "x": 300, "y": 420});
                    if shape == "box" {
                        p["box"] = json!([250.0, 250.0, size * 6.0, size * 4.0]);
                    }
                    let id = LayerId(app.run("type.create", p).unwrap()["layer"].as_u64().unwrap());
                    if let Some(a) = tf {
                        app.run("type.edit", json!({"layer": id.0, "transform": a.m})).unwrap();
                    }
                    app.ui.view.flip_horizontal = flip;
                    let mut h = harness(ppp, app);
                    // Centre the view on the text's middle, then pan.
                    let (l, aff, _) = layout(h.state_mut(), id).unwrap();
                    let b = l.bounds().unwrap();
                    let m = aff.apply(Point::new(f64::from(b[0] + b[2]) / 2.0, f64::from(b[1] + b[3]) / 2.0));
                    let v = &mut h.state_mut().ui.views[0];
                    (v.zoom, v.center, v.fit_pending) = (zoom, [m.x as f32 + pan[0], m.y as f32 + pan[1]], false);
                    h.run_steps(2);
                    let ctx = format!("ppp {ppp} zoom {zoom} flip {flip} {shape} transformed {}", tf.is_some());
                    let rect = xf(h.state()).rect;
                    for (i, f) in [(0, 0.2), (2, 0.8), (4, 0.3), (8, 0.7)] {
                        let p = glyph(&mut h, id, i, f);
                        assert!(rect.contains(p), "{ctx}: glyph {i} off screen at {p:?}");
                    }
                    // Click into the layer: caret before glyph 2 (left part), then after glyph 4.
                    let p = glyph(&mut h, id, 2, 0.2);
                    click(&mut h, p);
                    assert_eq!(selection(&h), (2, 2), "{ctx}: click on the left of glyph 2");
                    let p = glyph(&mut h, id, 4, 0.8);
                    click(&mut h, p);
                    assert_eq!(selection(&h), (5, 5), "{ctx}: click on the right of glyph 4");
                    // Drag from glyph 1 (left part) to glyph 8 (right part): characters 1..9.
                    let (a, b) = (glyph(&mut h, id, 1, 0.25), glyph(&mut h, id, 8, 0.75));
                    drag(&mut h, a, b);
                    assert_eq!(selection(&h), (1, 9), "{ctx}: drag selection");
                    // And backwards, starting inside the egui drag threshold of a glyph edge.
                    let (a, b) = (glyph(&mut h, id, 5, 0.85), glyph(&mut h, id, 3, 0.15));
                    drag(&mut h, a, b);
                    assert_eq!(selection(&h), (6, 3), "{ctx}: backwards drag selection");
                }
            }
        }
    }
}

fn text(app: &PhotocraftApp, id: LayerId) -> TextLayer {
    text_layer(&app.session.active().unwrap().doc, id).unwrap().clone()
}

/// A PSD type layer keeps Photoshop's pixels until it is edited. Clicking into it re-renders it
/// with our engine first (so the caret sits on what is shown), inside the edit session's single
/// history step; Cancel brings Photoshop's pixels back.
#[test]
fn editing_psd_type_shows_our_layout_and_cancel_restores_it() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "HOHOHO", "size": 120, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    // Stand-in for Photoshop's rendering: the same text drawn somewhere else.
    let moved = {
        let mut t = text(&app, id);
        t.transform = Affine::translate(330.0, 380.0);
        photocraft_text::shared().lock().unwrap().render(&t, 72.0, photocraft_color::PixelFormat::RGBA8).1.surface
    };
    let st = app.session.active_mut().unwrap();
    let mut doc = (*st.doc).clone();
    if let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) {
        t.cache = Some(moved.clone());
    }
    st.doc = std::sync::Arc::new(doc);
    st.revision += 1;
    let steps = app.session.active().unwrap().history.entries().len();
    let mut h = harness(1.0, app);
    let p = glyph(&mut h, id, 2, 0.2);
    click(&mut h, p);
    assert_eq!(selection(&h), (2, 2));
    let ours = text(h.state(), id).cache.unwrap().content_bounds();
    assert_ne!(ours, moved.content_bounds(), "re-rendered where the caret is");
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps + 1);
    super::insert(h.state_mut(), "x");
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps + 1, "one step for the session");
    super::cancel(h.state_mut());
    assert_eq!(text(h.state(), id).cache.unwrap().content_bounds(), moved.content_bounds());
    assert_eq!(text(h.state(), id).text, "HOHOHO");
    // A native layer shows our layout already: clicking into it records nothing.
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "HOHOHO", "size": 120, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    let steps = app.session.active().unwrap().history.entries().len();
    let mut h = harness(1.0, app);
    let p = glyph(&mut h, id, 3, 0.2);
    click(&mut h, p);
    assert_eq!(selection(&h), (3, 3));
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps);
}

/// Preferences ▸ Type ▸ Use Escape to Commit (default on): with it off, Escape cancels the
/// session like the Cancel button instead of committing.
#[test]
fn use_escape_to_commit_off_makes_escape_cancel_the_session() {
    for (use_esc_to_commit, expected) in [(true, "HOxHOHO"), (false, "HOHOHO")] {
        let mut app = new_app();
        let id = LayerId(app.run("type.create", json!({"text": "HOHOHO", "size": 120, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
        app.run("prefs.set", json!({"path": "type.useEscToCommit", "value": use_esc_to_commit})).unwrap();
        let mut h = harness(1.0, app);
        let p = glyph(&mut h, id, 2, 0.2);
        click(&mut h, p);
        super::insert(h.state_mut(), "x");
        assert_eq!(text(h.state(), id).text, "HOxHOHO", "typing goes into the session");
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert!(h.state().ui.text_edit.is_none(), "the session ends either way (useEscToCommit={use_esc_to_commit})");
        assert_eq!(text(h.state(), id).text, expected, "useEscToCommit={use_esc_to_commit}");
    }
}

fn size_at(app: &PhotocraftApp, id: LayerId, ci: usize) -> f32 {
    let t = text(app, id);
    let b = t.text.char_indices().nth(ci).map_or(t.text.len(), |(b, _)| b);
    let mut at = 0;
    for r in t.char_runs() {
        if b < at + r.len {
            return r.style.size_pt;
        }
        at += r.len;
    }
    0.0
}

/// Font size from the options bar / Properties: the whole layer when no characters are selected,
/// only the selection when editing with one, as the layer appears (transform scale included), and
/// for type that went through a PSD file.
#[test]
fn size_applies_at_layer_and_selection_scope() {
    for psd in [false, true] {
        let mut app = new_app();
        let id = LayerId(app.run("type.create", json!({"text": "Hello world", "size": 20, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
        app.run("type.edit", json!({"layer": id.0, "transform": [2.0, 0.0, 0.0, 2.0, 300.0, 420.0]})).unwrap();
        let id = if psd {
            let doc = (*app.session.active().unwrap().doc).clone();
            let out = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap();
            let back = photocraft_io::import("t.psd", &out.bytes).unwrap().document;
            app.session.add_document(back, None);
            app.sync_views();
            let d = &app.session.active().unwrap().doc;
            let id = d.walk().into_iter().find(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).map(|(_, _, l)| l.id).unwrap();
            let _ = app.session.select_layer(id);
            id
        } else {
            id
        };
        let ctx = egui::Context::default();
        assert!((super::shown_scale(&app) - 2.0).abs() < 1e-4, "psd {psd}");
        let w0 = text(&app, id).cache.unwrap().content_bounds().width();
        // Layer scope: 60 pt as shown = 30 pt in the layer's own space.
        let k = super::shown_scale(&app);
        super::apply(&mut app, &ctx, json!({"size": 60.0 / k}));
        assert_eq!((size_at(&app, id, 0), size_at(&app, id, 10)), (30.0, 30.0), "psd {psd}");
        let w1 = text(&app, id).cache.unwrap().content_bounds().width();
        assert!(w1 as f32 > w0 as f32 * 1.4, "psd {psd}: the pixels grow ({w0} → {w1})");
        // Selection scope: "world" only.
        app.ui.text_edit = Some(crate::state::TextEdit {
            layer: id.0,
            caret: 11,
            anchor: 6,
            session: "s".into(),
            created: false,
            mask: false,
            dragging: false,
            resize: None,
            preedit: None,
        });
        super::apply(&mut app, &ctx, json!({"size": 10.0}));
        assert_eq!((size_at(&app, id, 0), size_at(&app, id, 5), size_at(&app, id, 6), size_at(&app, id, 10)), (30.0, 30.0, 10.0, 10.0), "psd {psd}");
    }
}

/// Scrubbing the options bar's size field previews live (the size changes every frame) and is
/// one undo step per drag; each step only damages the type layer's area, not the whole canvas.
#[test]
fn size_drag_is_live_and_one_history_step_per_drag() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "Hello", "size": 20, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    let mut h = Harness::builder().with_size(vec2(900.0, 60.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            ui.horizontal(|ui| super::options_bar(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(3);
    let steps = h.state().session.active().unwrap().history.entries().len();
    let mut last = 20.0;
    for d in 0..2 {
        let field = h.get_all_by_role(egui::accesskit::Role::SpinButton).next().expect("size field").rect().center();
        h.hover_at(field);
        h.run_steps(1);
        press(&mut h, field, true);
        for k in 1..=5 {
            h.hover_at(field + vec2(8.0 * k as f32, 0.0));
            h.run_steps(1);
            let now = size_at(h.state(), id, 0);
            assert!(now > last, "drag {d} step {k}: live size {last} → {now}");
            last = now;
            let dmg = h.state().session.active().unwrap().last_damage.expect("damage rect, not a full refresh");
            assert!(!dmg.is_empty() && dmg.width() * dmg.height() <= 512 * 512, "{dmg:?}");
        }
        press(&mut h, field + vec2(40.0, 0.0), false);
        h.run_steps(2);
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps + d + 1, "one step per drag");
    }
    assert!(h.state_mut().session.undo());
    assert!(h.state_mut().session.undo());
    assert_eq!(size_at(h.state(), id, 0), 20.0);
}

/// A point inside glyph `i` of vertical type: `f` of the way down its advance, right of the
/// column's centre line.
fn vglyph(h: &mut Harness<'static, PhotocraftApp>, id: LayerId, i: usize, f: f32) -> Pos2 {
    let (l, _, text) = layout(h.state_mut(), id).unwrap();
    assert!(l.vertical);
    let b = text.char_indices().nth(i).unwrap().0;
    let c = l.clusters.iter().find(|c| c.range.start == b).unwrap().clone();
    let ln = &l.lines[c.line];
    let (x, y) = l.to_text(c.x + c.advance * f, ln.baseline - ln.ascent * 0.35);
    screen(h, id, x, y)
}

fn key(h: &mut Harness<'static, PhotocraftApp>, k: egui::Key) {
    h.event(egui::Event::Key { key: k, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(1);
}

#[test]
fn command_home_end_use_text_edges_and_shift_extends_the_selection() {
    for vertical in [false, true] {
        let mut app = new_app();
        let value = "AB🙂\nCDé";
        let id = LayerId(app.run("type.create", json!({"text": value, "size": 40, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
        if vertical {
            app.run("type.orientation.vertical", json!({"layer": id.0})).unwrap();
        }
        let mut h = harness(1.0, app);
        let p = if vertical { vglyph(&mut h, id, 5, 0.2) } else { glyph(&mut h, id, 5, 0.2) };
        click(&mut h, p);
        assert_eq!(selection(&h), (5, 5));
        let steps = h.state().session.active().unwrap().history.entries().len();
        h.key_press_modifiers(Modifiers::COMMAND, egui::Key::Home);
        h.run_steps(2);
        assert_eq!(selection(&h), (0, 0));
        h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, egui::Key::End);
        h.run_steps(2);
        assert_eq!(selection(&h), (0, value.chars().count()));
        h.key_press_modifiers(Modifiers::COMMAND, egui::Key::End);
        h.run_steps(2);
        assert_eq!(selection(&h), (value.chars().count(), value.chars().count()));
        h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, egui::Key::Home);
        h.run_steps(2);
        assert_eq!(selection(&h), (value.chars().count(), 0));
        assert_eq!(text(h.state(), id).text, value);
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps);
    }
}

/// Vertical type (#199): clicks and drags land on the glyph under the pointer down the column,
/// and the arrow keys follow the vertical flow (↓ next character, ← next column).
#[test]
fn vertical_type_caret_selection_and_arrows_follow_the_columns() {
    let rotated = Affine { m: [0.94, 0.342, -0.342, 0.94, 900.0, 200.0] };
    for ppp in [1.0, 2.0] {
        for tf in [None, Some(rotated)] {
            let mut app = new_app();
            let id = LayerId(app.run("type.create", json!({"text": "HOHOHO\nHOHO", "size": 60, "x": 800, "y": 120})).unwrap()["layer"].as_u64().unwrap());
            app.run("type.orientation.vertical", json!({"layer": id.0})).unwrap();
            // Snapping would pull the pointer onto the layer's edges and centre lines.
            app.ui.extras.snap = false;
            if let Some(a) = tf {
                app.run("type.edit", json!({"layer": id.0, "transform": a.m})).unwrap();
            }
            let mut h = harness(ppp, app);
            let (l, aff, _) = layout(h.state_mut(), id).unwrap();
            let b = l.bounds().unwrap();
            assert!(b[3] - b[1] > b[2] - b[0], "vertical bounds {b:?}");
            let m = aff.apply(Point::new(f64::from(b[0] + b[2]) / 2.0, f64::from(b[1] + b[3]) / 2.0));
            let v = &mut h.state_mut().ui.views[0];
            (v.zoom, v.center, v.fit_pending) = (0.5, [m.x as f32, m.y as f32], false);
            h.run_steps(2);
            let ctx = format!("ppp {ppp} transformed {}", tf.is_some());
            // Upper part of glyph 2 → caret before it; lower part of glyph 4 → after it.
            let p = vglyph(&mut h, id, 2, 0.2);
            click(&mut h, p);
            assert_eq!(selection(&h), (2, 2), "{ctx}");
            let p = vglyph(&mut h, id, 4, 0.8);
            click(&mut h, p);
            assert_eq!(selection(&h), (5, 5), "{ctx}");
            // Drag down the column from glyph 1 into the second column's glyph 2 (char 9).
            let (a, b) = (vglyph(&mut h, id, 1, 0.25), vglyph(&mut h, id, 9, 0.75));
            drag(&mut h, a, b);
            assert_eq!(selection(&h), (1, 10), "{ctx}: drag");
            // Arrows: ↓ next char, ↑ previous, ← next column (same position), → back.
            let p = vglyph(&mut h, id, 1, 0.2);
            click(&mut h, p);
            assert_eq!(selection(&h), (1, 1), "{ctx}");
            key(&mut h, egui::Key::ArrowDown);
            assert_eq!(selection(&h).1, 2, "{ctx}: ↓");
            key(&mut h, egui::Key::ArrowUp);
            assert_eq!(selection(&h).1, 1, "{ctx}: ↑");
            key(&mut h, egui::Key::ArrowLeft);
            assert_eq!(selection(&h).1, 8, "{ctx}: ← moves to the next column");
            key(&mut h, egui::Key::ArrowRight);
            assert_eq!(selection(&h).1, 1, "{ctx}: → moves back");
        }
    }
    assert_eq!(super::flow_key(egui::Key::ArrowUp, false), egui::Key::ArrowUp);
}

fn rgb_at(app: &PhotocraftApp, id: LayerId, ci: usize) -> [u8; 4] {
    let t = text(app, id);
    let b = t.text.char_indices().nth(ci).map_or(t.text.len(), |(b, _)| b);
    let mut at = 0;
    for r in t.char_runs() {
        if b < at + r.len {
            return r.style.color.to_rgba8();
        }
        at += r.len;
    }
    [0; 4]
}

/// A new foreground colour recolours the selected characters only, inside the editing session's
/// history step; with nothing selected (a caret, or not editing) the type keeps its colour.
#[test]
fn foreground_colour_recolours_only_selected_type() {
    let mut app = new_app();
    let id =
        LayerId(app.run("type.create", json!({"text": "Hello world", "size": 40, "x": 300, "y": 420, "color": "#000000"})).unwrap()["layer"].as_u64().unwrap());
    app.run("tools.setColors", json!({"foreground": "#ff0000"})).unwrap();
    super::foreground_changed(&mut app);
    assert_eq!(rgb_at(&app, id, 0), [0, 0, 0, 255], "not editing: unchanged");
    let edit = |caret, anchor| crate::state::TextEdit {
        layer: id.0,
        caret,
        anchor,
        session: "s".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    };
    app.ui.text_edit = Some(edit(3, 3));
    super::foreground_changed(&mut app);
    assert_eq!(rgb_at(&app, id, 0), [0, 0, 0, 255], "a caret: unchanged");
    let steps = app.session.active().unwrap().history.entries().len();
    app.ui.text_edit = Some(edit(11, 6));
    super::foreground_changed(&mut app);
    app.run("tools.setColors", json!({"foreground": "#00ff00"})).unwrap();
    super::foreground_changed(&mut app);
    assert_eq!((rgb_at(&app, id, 0), rgb_at(&app, id, 5)), ([0, 0, 0, 255], [0, 0, 0, 255]));
    assert_eq!((rgb_at(&app, id, 6), rgb_at(&app, id, 10)), ([0, 255, 0, 255], [0, 255, 0, 255]));
    assert_eq!(app.session.active().unwrap().history.entries().len(), steps + 1, "one step for the session");
}

#[test]
fn vertical_type_tool_creates_point_and_paragraph_text_with_one_undo() {
    use crate::canvas::{ToolEvent, tool_event};
    use photocraft_doc::text::{Orientation, TextShape};
    for end in [[100.0, 100.0], [240.0, 230.0]] {
        let mut app = new_app();
        app.ui.tool = crate::state::Tool::VerticalType;
        let before = app.session.active().unwrap().history.entries().len();
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 100.0, pressure: 1.0 }, Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Move { x: end[0], y: end[1], pressure: 1.0 }, Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Up { x: end[0], y: end[1] }, Modifiers::NONE);
        let id = LayerId(app.ui.text_edit.as_ref().unwrap().layer);
        let st = app.session.active().unwrap();
        let text = text_layer(&st.doc, id).unwrap();
        assert_eq!(text.orientation, Orientation::Vertical);
        assert_eq!(matches!(text.shape, TextShape::Box { .. }), end != [100.0, 100.0]);
        assert_eq!(st.history.entries().len(), before + 1);
        super::commit(&mut app);
        app.run("edit.undo", json!({})).unwrap();
        assert!(app.session.active().unwrap().doc.layer(id).is_none());
    }
}

/// #668: ⌘/Ctrl+T while typing shows or hides the Character panel, as in Photoshop, instead of
/// starting Free Transform on the layer being typed into.
#[test]
fn command_t_while_typing_toggles_the_character_panel() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "HOHO", "size": 60, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    let mut h = harness(1.0, app);
    let p = glyph(&mut h, id, 1, 0.5);
    click(&mut h, p);
    assert!(h.state().ui.text_edit.is_some(), "editing");
    let before = crate::view_cmds::checked(h.state(), "window.panel.character");
    h.event(egui::Event::Key { key: egui::Key::T, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::COMMAND });
    h.run_steps(2);
    assert!(h.state().ui.transform.is_none(), "no Free Transform while typing");
    assert!(h.state().ui.text_edit.is_some(), "still editing");
    assert_ne!(crate::view_cmds::checked(h.state(), "window.panel.character"), before, "the Character panel toggled");
}

/// #1881: Tab while typing inserts a tab character (laid out to the next tab stop), as in
/// Photoshop, instead of moving keyboard focus or hiding the panels.
#[test]
fn tab_while_typing_inserts_a_tab() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "HOHO", "size": 60, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    let mut h = harness(1.0, app);
    let p = glyph(&mut h, id, 1, 0.2);
    click(&mut h, p);
    assert_eq!(selection(&h), (1, 1));
    let panels = h.state().ui.panels.dock;
    key(&mut h, egui::Key::Tab);
    assert_eq!(text(h.state(), id).text, "H\tOHO");
    assert_eq!(selection(&h), (2, 2));
    assert!(h.state().ui.text_edit.is_some(), "still editing");
    assert_eq!(h.state().ui.panels.dock, panels, "the panels stay as they were");
}

#[test]
fn text_color_dialog_edits_only_the_selected_range_and_cancel_is_inert() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text":"Hello world","size":40,"color":"#000000"})).unwrap()["layer"].as_u64().unwrap());
    let peer = LayerId(app.run("type.create", json!({"text":"Selected peer","color":"#000000"})).unwrap()["layer"].as_u64().unwrap());
    app.run("layer.select", json!({"layer": id.0, "mode": "add"})).unwrap();
    app.ui.text_edit = Some(crate::state::TextEdit {
        layer: id.0,
        caret: 11,
        anchor: 6,
        session: "text-color-dialog-test".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    let foreground = app.session.tools.foreground;
    let before = app.session.active().unwrap().history.entries().len();
    let selection = app.ui.text_edit.clone();
    let dialog = super::open_color_picker(&mut app, [0.0; 3]);
    app.ui.dialogs.last_mut().unwrap().fields.insert("color".into(), json!("#ff0000"));
    app.ui.close_dialog(dialog).unwrap();
    assert_eq!(rgb_at(&app, id, 6), [0, 0, 0, 255]);
    assert_eq!(app.session.active().unwrap().history.entries().len(), before);
    assert_eq!(app.ui.text_edit, selection);
    let dialog = super::open_color_picker(&mut app, [0.0; 3]);
    assert_eq!(app.ui.dialogs.last().unwrap().fields["__label"], "Color Picker (Text Color)");
    app.ui.dialogs.last_mut().unwrap().fields.insert("color".into(), json!("#00ff00"));
    crate::dialogs::confirm(&mut app, dialog).unwrap();
    assert_eq!(rgb_at(&app, id, 0), [0, 0, 0, 255]);
    assert_eq!(rgb_at(&app, id, 5), [0, 0, 0, 255]);
    assert_eq!(rgb_at(&app, id, 6), [0, 255, 0, 255]);
    assert_eq!(rgb_at(&app, id, 10), [0, 255, 0, 255]);
    assert_eq!(app.session.tools.foreground, foreground, "text-only picker leaves the foreground alone");
    assert_eq!(app.ui.text_edit, selection, "dialog preserves the caret and selection");
    assert_eq!(app.session.active().unwrap().history.entries().len(), before + 1);
    let dialog = super::open_color_picker(&mut app, [0.0, 1.0, 0.0]);
    app.ui.dialogs.last_mut().unwrap().fields.insert("color".into(), json!("#0000ff"));
    crate::dialogs::confirm(&mut app, dialog).unwrap();
    assert_eq!(app.session.active().unwrap().history.entries().len(), before + 1, "one editing-session undo step");
    assert_eq!(rgb_at(&app, peer, 0), [0, 0, 0, 255], "a selected peer is outside the active text edit");
}

#[test]
fn text_color_dialog_without_a_selection_edits_the_whole_layer() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text":"Hello","color":"#000000"})).unwrap()["layer"].as_u64().unwrap());
    let dialog = super::open_color_picker(&mut app, [0.0; 3]);
    app.ui.dialogs.last_mut().unwrap().fields.insert("color".into(), json!("#ff8800"));
    crate::dialogs::confirm(&mut app, dialog).unwrap();
    assert_eq!(rgb_at(&app, id, 0), [255, 136, 0, 255]);
    assert_eq!(rgb_at(&app, id, 4), [255, 136, 0, 255]);
}

/// #1381: on Wayland the input method sends an empty preedit (and sometimes an empty commit)
/// whenever the caret rectangle moves, e.g. while ⌘A, ⇧-arrows or a drag select text. With
/// nothing being composed that must leave the text and the selection alone; a real composition
/// still replaces the selection, and clearing it removes only the composed text.
#[test]
fn empty_ime_events_never_delete_the_selection() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "Hello", "size": 40, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    app.ui.text_edit = Some(crate::state::TextEdit {
        layer: id.0,
        caret: 5,
        anchor: 0,
        session: "s".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    let mut h = harness(1.0, app);
    let ime = |h: &mut Harness<'static, PhotocraftApp>, e: egui::ImeEvent| {
        h.event(egui::Event::Ime(e));
        h.run_steps(1);
    };
    ime(&mut h, egui::ImeEvent::Preedit { text: String::new(), active_range_chars: None });
    ime(&mut h, egui::ImeEvent::Commit(String::new()));
    assert_eq!(text(h.state(), id).text, "Hello", "empty IME events keep the text");
    assert_eq!(selection(&h), (0, 5), "and the selection");
    // Composing replaces the selection; cancelling the composition leaves the rest.
    ime(&mut h, egui::ImeEvent::Preedit { text: "か".into(), active_range_chars: None });
    assert_eq!(text(h.state(), id).text, "か");
    ime(&mut h, egui::ImeEvent::Preedit { text: String::new(), active_range_chars: None });
    assert_eq!(text(h.state(), id).text, "");
}

fn type_command() -> Modifiers {
    Modifiers { command: true, ctrl: true, ..Modifiers::NONE }
}

fn hold(h: &mut Harness<'static, PhotocraftApp>, mods: Modifiers) {
    h.event(egui::Event::ModifiersChanged(mods));
    h.run_steps(1);
}

fn transform_press(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool, mods: Modifiers) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: mods });
    h.run_steps(1);
}

fn transform_drag(h: &mut Harness<'static, PhotocraftApp>, a: Pos2, b: Pos2, mods: Modifiers, release_command: bool) {
    hold(h, mods);
    h.hover_at(a);
    h.run_steps(1);
    transform_press(h, a, true, mods);
    for k in 1..=6 {
        if release_command && k == 4 {
            hold(h, Modifiers { command: false, ctrl: false, mac_cmd: false, ..mods });
        }
        h.hover_at(a + (b - a) * (k as f32 / 6.0));
        h.run_steps(1);
    }
    transform_press(h, b, false, if release_command { Modifiers::NONE } else { mods });
    hold(h, Modifiers::NONE);
}

fn transform_app(paragraph: bool, vertical: bool, transform: Affine) -> (PhotocraftApp, LayerId) {
    let mut app = new_app();
    let mut p = json!({"text": "Hello é世界\nSecond line", "size": 32, "x": 300, "y": 420});
    if paragraph {
        p["box"] = json!([300, 300, 240, 180]);
    }
    if vertical {
        p["orientation"] = json!("vertical");
    }
    let id = LayerId(app.run("type.create", p).unwrap()["layer"].as_u64().unwrap());
    app.run("type.edit", json!({"layer": id.0, "transform": transform.m})).unwrap();
    super::edit_active(&mut app).unwrap();
    let ed = app.ui.text_edit.as_mut().unwrap();
    ed.caret = 3;
    ed.anchor = 3;
    (app, id)
}

fn frame_point(app: &PhotocraftApp, p: [f64; 2]) -> Pos2 {
    xf(app).to_screen(p[0] as f32, p[1] as f32)
}

fn assert_affine(a: Affine, b: Affine) {
    for (x, y) in a.m.into_iter().zip(b.m) {
        // Translations around 500 doc px at a scaled ppp cancel to ~2.7e-4 in f32 rounding.
        assert!((x - y).abs() < 1e-3, "{a:?} != {b:?}");
    }
}

/// #1091: modifier changes alone draw/hide the oriented frame without touching editing/history.
#[test]
fn temporary_type_transform_visibility_and_cursors() {
    let (app, id) = transform_app(false, false, Affine::translate(500.0, 350.0));
    let mut h = harness(1.0, app);
    let before = h.state().session.active().unwrap().history.entries().len();
    let original = text(h.state(), id);
    assert!(!crate::type_transform::visible(h.state(), Modifiers::NONE));
    hold(&mut h, type_command());
    assert!(crate::type_transform::visible(h.state(), h.ctx.input(|i| i.modifiers)));
    let frame = crate::type_transform::frame(h.state_mut()).unwrap();
    let mut positions: Vec<[f64; 2]> = frame.quad.to_vec();
    positions.extend((0..4).map(|i| [(frame.quad[i][0] + frame.quad[(i + 1) % 4][0]) / 2.0, (frame.quad[i][1] + frame.quad[(i + 1) % 4][1]) / 2.0]));
    for (i, q) in positions.into_iter().enumerate() {
        let p = frame_point(h.state(), q);
        h.hover_at(p);
        h.run_steps(2);
        let cursor = h.output().platform_output.cursor_icon;
        if i < 4 {
            assert!(matches!(
                cursor,
                egui::CursorIcon::ResizeNwSe | egui::CursorIcon::ResizeNeSw | egui::CursorIcon::ResizeHorizontal | egui::CursorIcon::ResizeVertical
            ));
        } else {
            assert_eq!(cursor, egui::CursorIcon::None, "skew arrowhead");
        }
    }
    let middle = [(frame.quad[0][0] + frame.quad[1][0]) / 2.0, frame.quad[0][1] - 20.0 / f64::from(h.state().current_zoom())];
    h.hover_at(frame_point(h.state(), middle));
    h.run_steps(2);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "curved rotation cursor");
    hold(&mut h, Modifiers::NONE);
    assert!(!crate::type_transform::visible(h.state(), Modifiers::NONE));
    assert_eq!(selection(&h), (3, 3));
    assert_eq!(text(h.state(), id).transform, original.transform);
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), before);
    assert!(h.state().ui.transform.is_none());
    assert!(h.state().ui.type_transform.is_none());
    // macOS's real Control is not the logical Command modifier.
    assert!(!crate::type_transform::visible(h.state(), Modifiers { ctrl: true, command: false, ..Modifiers::NONE }));
    h.state_mut().ui.shell.sticky_command = true;
    let sticky = crate::workspace_ui::sticky_mods(h.state(), Modifiers::NONE);
    assert!(crate::type_transform::visible(h.state(), sticky));
}

/// Real canvas input: layer-local scaling stays correct across orientation, zoom, HiDPI and flip.
#[test]
fn temporary_type_transform_scales_without_reflow_or_leaving_type() {
    for ppp in [1.0, 2.0] {
        for zoom in [0.25, 1.0, 2.0, 8.0] {
            for (paragraph, vertical, flip) in [(false, false, false), (true, false, true), (false, true, true)] {
                let transform = Affine { m: [1.2, 0.25, 0.18, 0.9, 500.0, 360.0] };
                let (app, id) = transform_app(paragraph, vertical, transform);
                let mut h = harness(ppp, app);
                let f = crate::type_transform::frame(h.state_mut()).unwrap();
                let center = f.pivot;
                h.state_mut().ui.view.flip_horizontal = flip;
                let view = &mut h.state_mut().ui.views[0];
                // At high zoom, show the dragged corner; a release beyond the canvas is valid.
                view.center = if zoom >= 2.0 { [f.quad[2][0] as f32, f.quad[2][1] as f32] } else { [center[0] as f32 + 17.0, center[1] as f32 - 9.0] };
                view.zoom = zoom;
                view.fit_pending = false;
                h.run_steps(2);
                let before = text(h.state(), id);
                let count = h.state().session.active().unwrap().history.entries().len();
                let (a, b) = (f.quad[2], [f.quad[0][0] + 1.25 * (f.quad[2][0] - f.quad[0][0]), f.quad[0][1] + 1.25 * (f.quad[2][1] - f.quad[0][1])]);
                let (a, b) = (frame_point(h.state(), a), frame_point(h.state(), b));
                assert!(xf(h.state()).rect.contains(a));
                transform_drag(&mut h, a, b, type_command() | Modifiers::SHIFT, true);
                let after = text(h.state(), id);
                assert_affine(
                    after.transform,
                    transform.mul(&Affine::translate(f.rect[0], f.rect[1])).mul(&Affine::scale(1.25)).mul(&Affine::translate(-f.rect[0], -f.rect[1])),
                );
                assert_eq!(
                    (after.text.as_str(), &after.runs, &after.paragraphs, after.orientation, after.shape),
                    (before.text.as_str(), &before.runs, &before.paragraphs, before.orientation, before.shape)
                );
                assert_eq!(selection(&h), (3, 3));
                assert_eq!(h.state().ui.text_edit.as_ref().unwrap().layer, id.0);
                assert!(h.state().ui.tool.is_type());
                assert!(h.state().ui.transform.is_none());
                assert!(h.state().ui.type_transform.is_none());
                assert_eq!(h.state().session.active().unwrap().history.entries().len(), count + 1);
                h.event(egui::Event::Text("!".into()));
                h.run_steps(2);
                assert!(text(h.state(), id).text.starts_with("Hel!lo"));
                h.key_press(if vertical { egui::Key::ArrowUp } else { egui::Key::ArrowLeft });
                h.run_steps(1);
                assert_eq!(selection(&h), (3, 3));
                assert_eq!(h.state().session.active().unwrap().history.entries().len(), count + 1, "one edit session");
                h.state_mut().run("edit.undo", json!({})).unwrap();
                assert_affine(text(h.state(), id).transform, before.transform);
                assert_eq!(text(h.state(), id).text, before.text);
                h.state_mut().run("edit.redo", json!({})).unwrap();
                assert_eq!(text(h.state(), id).transform, after.transform);
            }
        }
    }
}

#[test]
fn temporary_type_transform_free_scale_center_scale_move_skew_and_rotation() {
    use crate::canvas::{ToolEvent, tool_event};
    for gesture in ["free", "center", "move", "skew", "rotate", "snap"] {
        let initial = Affine::translate(400.0, 350.0).mul(&Affine::rotate(0.2));
        let (mut app, id) = transform_app(true, false, initial);
        let frame = crate::type_transform::frame(&mut app).unwrap();
        let before = text(&app, id);
        let (a, b, mods) = match gesture {
            "move" => (frame.quad[0].map(|v| v + 20.0), frame.quad[0].map(|v| v + 45.0), type_command()),
            "skew" => {
                let p = [(frame.quad[0][0] + frame.quad[1][0]) / 2.0, (frame.quad[0][1] + frame.quad[1][1]) / 2.0];
                (p, [p[0] + 30.0, p[1] + 12.0], type_command())
            }
            "rotate" | "snap" => {
                let a = [frame.quad[1][0] + 20.0, frame.quad[1][1] - 10.0];
                let da: f64 = 0.42;
                let (x, y) = (a[0] - frame.pivot[0], a[1] - frame.pivot[1]);
                let b = [frame.pivot[0] + x * da.cos() - y * da.sin(), frame.pivot[1] + x * da.sin() + y * da.cos()];
                (a, b, if gesture == "snap" { type_command() | Modifiers::SHIFT } else { type_command() })
            }
            _ => {
                let a = frame.quad[2];
                let local = [frame.rect[2] + 60.0, frame.rect[3] + 18.0];
                let p = initial.apply(Point::new(local[0], local[1]));
                (a, [p.x, p.y], if gesture == "center" { type_command() | Modifiers::ALT } else { type_command() })
            }
        };
        let count = app.session.active().unwrap().history.entries().len();
        tool_event(&mut app, ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, mods);
        for k in 1..=5 {
            let f = f64::from(k) / 5.0;
            tool_event(&mut app, ToolEvent::Move { x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, pressure: 1.0 }, mods);
            assert!(crate::type_transform::display_doc(&mut app, 0).is_some());
            assert_eq!(text(&app, id).transform, initial, "preview leaves engine untouched");
        }
        assert_eq!(app.session.active().unwrap().history.entries().len(), count);
        tool_event(&mut app, ToolEvent::Up { x: b[0], y: b[1] }, mods);
        let after = text(&app, id);
        assert_ne!(after.transform, initial, "{gesture}");
        assert_eq!(after.shape, before.shape, "no paragraph reflow");
        assert_eq!(selection(&harness(1.0, app)), (3, 3));
        // Assertions below use the matrix itself to check the chosen gesture.
        if gesture == "move" {
            assert_affine(after.transform, Affine::translate(25.0, 25.0).mul(&initial));
        } else if gesture == "rotate" {
            assert_affine(
                after.transform,
                Affine::translate(frame.pivot[0], frame.pivot[1])
                    .mul(&Affine::rotate(0.42))
                    .mul(&Affine::translate(-frame.pivot[0], -frame.pivot[1]))
                    .mul(&initial),
            );
        } else if gesture == "snap" {
            let angle = after.transform.m[1].atan2(after.transform.m[0]).to_degrees();
            assert!((angle / 15.0 - (angle / 15.0).round()).abs() < 1e-8);
        } else if gesture == "center" {
            let p = after.transform.apply(Point::new((frame.rect[0] + frame.rect[2]) / 2.0, (frame.rect[1] + frame.rect[3]) / 2.0));
            assert!((p.x - frame.pivot[0]).abs() < 1e-8 && (p.y - frame.pivot[1]).abs() < 1e-8);
        } else if gesture == "free" {
            let ratio_x = after.transform.m[0].hypot(after.transform.m[1]);
            let ratio_y = after.transform.m[2].hypot(after.transform.m[3]);
            assert!((ratio_x - ratio_y).abs() > 0.05, "unconstrained corners");
        } else if gesture == "skew" {
            assert_ne!(after.transform.m[2], before.transform.m[2]);
        }
    }
}

#[test]
fn temporary_type_transform_noop_cancel_and_stale_capture() {
    use crate::canvas::{ToolEvent, tool_event};
    for stop in ["noop", "escape", "focus", "tool", "undo", "delete", "document", "close"] {
        let original = Affine::translate(400.0, 350.0);
        let (app, id) = transform_app(false, false, original);
        let mut h = harness(1.0, app);
        let f = crate::type_transform::frame(h.state_mut()).unwrap();
        let a = f.quad[2];
        let count = h.state().session.active().unwrap().history.entries().len();
        tool_event(h.state_mut(), ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
        tool_event(h.state_mut(), ToolEvent::Move { x: a[0] + 50.0, y: a[1] + 20.0, pressure: 1.0 }, type_command());
        let preview = {
            let (shown, _) = crate::type_transform::display_doc(h.state_mut(), 0).unwrap();
            std::sync::Arc::downgrade(&shown)
        };
        assert_eq!(text(h.state(), id).transform, original);
        match stop {
            "noop" => tool_event(h.state_mut(), ToolEvent::Up { x: a[0], y: a[1] }, Modifiers::NONE),
            "escape" => {
                h.key_press(egui::Key::Escape);
                h.run_steps(2);
            }
            "focus" => {
                h.input_mut().focused = false;
                h.run_steps(2);
            }
            "tool" => {
                h.state_mut().ui.tool = crate::state::Tool::Brush;
                h.state_mut().sync_views();
            }
            "undo" => {
                h.state_mut().run("edit.undo", json!({})).unwrap();
            }
            "delete" => {
                h.state_mut().run("layer.delete", json!({"layer": id.0})).unwrap();
            }
            "close" => {
                h.state_mut().run("file.close", json!({})).unwrap();
            }
            _ => {
                h.state_mut().run("file.new", json!({"width": 20, "height": 20})).unwrap();
            }
        }
        assert!(h.state().ui.type_transform.is_none(), "{stop}");
        assert!(preview.upgrade().is_none(), "{stop}: release rendered preview without waiting for another frame");
        if ["noop", "escape", "focus", "tool"].contains(&stop) {
            assert_eq!(text(h.state(), id).transform, original);
            assert_eq!(h.state().session.active().unwrap().history.entries().len(), count);
        }
        if stop == "escape" {
            assert!(h.state().ui.text_edit.is_some());
        }
    }
}

#[test]
fn temporary_type_transform_control_roundtrip_and_input_safety() {
    use crate::control::{ControlRequest, Outcome};
    let initial = Affine::translate(400.0, 350.0);
    let (mut app, id) = transform_app(true, false, initial);
    // A document selection must not restrict a text-layer transformation.
    app.run("select.rect", json!({"x": 1, "y": 1, "width": 2, "height": 2})).unwrap();
    let frame = crate::type_transform::frame(&mut app).unwrap();
    let a = frame.quad[2];
    let ctx = egui::Context::default();
    let call = |app: &mut PhotocraftApp, params| {
        let (req, _) = ControlRequest::new("ui.pointer", params);
        let Outcome::Done(result) = crate::control::handle(app, &ctx, &req) else { panic!("expected immediate pointer reply") };
        assert_eq!(result["ok"], true, "{result}");
    };
    call(&mut app, json!({"command": true, "events": [{"kind": "down", "x": a[0], "y": a[1]}]}));
    let inspected = crate::control::inspect(&app, &ctx);
    assert_eq!(inspected["typeTransform"]["frame"]["layer"], id.0);
    call(&mut app, json!({"command": true, "events": [{"kind": "move", "x": a[0]+60.0, "y": a[1]+30.0}]}));
    call(&mut app, json!({"events": [{"kind": "up", "x": a[0]+60.0, "y": a[1]+30.0}]}));
    assert_ne!(text(&app, id).transform, initial);
    let doc = &app.session.active().unwrap().doc;
    for name in ["type.pcraft", "type.psd"] {
        let out = photocraft_io::export(doc, name, &Default::default()).unwrap();
        let back = photocraft_io::import(name, &out.bytes).unwrap().document;
        let t = back.walk().into_iter().find_map(|(_, _, l)| if let LayerContent::Text(t) = &l.content { Some(t) } else { None }).unwrap();
        assert_affine(t.transform, text(&app, id).transform);
        assert_eq!(t.text, text(&app, id).text);
        assert_eq!(t.shape, text(&app, id).shape);
    }
    for content in ["", "   ", "é👩‍💻世界", "\n\n"] {
        app.run("type.edit", json!({"layer": id.0, "text": content, "point": [400, 350]})).unwrap();
        assert!(crate::type_transform::frame(&mut app).is_some());
        call(
            &mut app,
            json!({"command": true, "events": [{"kind": "down", "x": 400, "y": 350}, {"kind": "move", "x": 1e300, "y": -1e300}, {"kind": "up", "x": 400, "y": 350}]}),
        );
        assert!(text(&app, id).transform.m.iter().all(|v| v.is_finite()));
        assert!(app.ui.type_transform.is_none());
    }
    app.run("layer.lockLayers", json!({"position": true})).unwrap();
    assert!(crate::type_transform::frame(&mut app).is_none());
}

#[test]
fn temporary_type_transform_preserves_depth_and_color_model() {
    use crate::canvas::{ToolEvent, tool_event};
    for (depth, mode) in [(8, "rgb"), (16, "rgb"), (32, "rgb"), (16, "gray"), (8, "cmyk"), (16, "lab")] {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 1200, "height": 900, "depth": depth, "mode": mode})).unwrap();
        let id = LayerId(app.run("type.create", json!({"text": "PhotoCraft é世界", "size": 32, "x": 300, "y": 350})).unwrap()["layer"].as_u64().unwrap());
        super::edit_active(&mut app).unwrap();
        let format = app.session.active().unwrap().doc.pixel_format();
        let a = crate::type_transform::frame(&mut app).unwrap().quad[2];
        tool_event(&mut app, ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
        tool_event(&mut app, ToolEvent::Move { x: a[0] + 30.0, y: a[1] + 15.0, pressure: 1.0 }, type_command());
        let (shown, _) = crate::type_transform::display_doc(&mut app, 0).unwrap();
        assert_eq!(shown.pixel_format(), format);
        assert_eq!(text_layer(&shown, id).unwrap().cache.as_ref().unwrap().format(), format);
        tool_event(&mut app, ToolEvent::Up { x: a[0] + 30.0, y: a[1] + 15.0 }, Modifiers::NONE);
        assert_eq!(text(&app, id).cache.as_ref().unwrap().format(), format);
        assert_eq!(text(&app, id).text, "PhotoCraft é世界");
    }
}

#[test]
fn temporary_type_transform_preserves_paragraph_resize_pivot_ime_and_shortcuts() {
    use crate::canvas::{ToolEvent, tool_event};
    let (app, id) = transform_app(true, false, Affine::translate(350.0, 300.0));
    let mut h = harness(1.0, app);
    let f = crate::type_transform::frame(h.state_mut()).unwrap();
    let before = text(h.state(), id);
    let (a, b) = (frame_point(h.state(), f.quad[2]), frame_point(h.state(), [f.quad[2][0] + 40.0, f.quad[2][1] + 20.0]));
    h.hover_at(a);
    h.run_steps(2);
    assert!(matches!(h.output().platform_output.cursor_icon, egui::CursorIcon::ResizeNwSe | egui::CursorIcon::ResizeNeSw));
    drag(&mut h, a, b);
    let after = text(h.state(), id);
    assert_ne!(after.shape, before.shape);
    assert_eq!(after.transform, before.transform, "plain drag resizes the frame");
    let f = crate::type_transform::frame(h.state_mut()).unwrap();
    let count = h.state().session.active().unwrap().history.entries().len();
    let pivot = [f.pivot[0] + 45.0, f.pivot[1] - 15.0];
    tool_event(h.state_mut(), ToolEvent::Down { x: f.pivot[0], y: f.pivot[1], pressure: 1.0 }, type_command());
    tool_event(h.state_mut(), ToolEvent::Up { x: pivot[0], y: pivot[1] }, type_command());
    assert_eq!(h.state().ui.type_transform_pivot, Some(pivot));
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), count, "moving the pivot is view state");
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
    h.run_steps(2);
    assert_eq!(selection(&h), (0, after.text.chars().count()));
    h.event(egui::Event::Ime(egui::ImeEvent::Preedit { text: "候補".into(), active_range_chars: None }));
    h.run_steps(2);
    let f = crate::type_transform::frame(h.state_mut()).unwrap();
    let a = f.quad[2];
    tool_event(h.state_mut(), ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
    tool_event(h.state_mut(), ToolEvent::Up { x: a[0] + 20.0, y: a[1] + 10.0 }, type_command());
    assert_eq!(h.state().ui.text_edit.as_ref().unwrap().preedit, Some((0, 2)));
    let f = crate::type_transform::frame(h.state_mut()).unwrap();
    let a = f.quad[2];
    tool_event(h.state_mut(), ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
    tool_event(h.state_mut(), ToolEvent::Move { x: a[0] + 15.0, y: a[1] + 10.0, pressure: 1.0 }, type_command());
    let composed_transform = crate::type_transform::current_transform(h.state(), id).unwrap();
    h.event(egui::Event::Ime(egui::ImeEvent::Commit("確定".into())));
    h.run_steps(2);
    assert_eq!(text(h.state(), id).text, "確定");
    assert_affine(text(h.state(), id).transform, composed_transform);
    assert!(h.state().ui.type_transform.is_none(), "composition commits a pending drag before editing");
    assert!(h.state().ui.text_edit.as_ref().unwrap().preedit.is_none());
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::Enter);
    h.run_steps(2);
    assert!(h.state().ui.text_edit.is_none());
    super::edit_active(h.state_mut()).unwrap();
    assert!(h.state().ui.text_edit.is_some());
    super::commit(h.state_mut());
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::T);
    h.run_steps(2);
    assert!(h.state().ui.transform.is_some(), "outside typing: ordinary Free Transform");
}

#[test]
fn temporary_type_transform_captures_press_and_rejects_collapsed_geometry() {
    let (app, id) = transform_app(false, false, Affine::translate(450.0, 350.0));
    let mut h = harness(1.0, app);
    let frame = crate::type_transform::frame(h.state_mut()).unwrap();
    let a = frame_point(h.state(), frame.quad[2]);
    let b = a + vec2(50.0, 20.0);
    hold(&mut h, type_command());
    h.hover_at(a);
    transform_press(&mut h, a, true, type_command());
    assert!(h.state().ui.type_transform.is_some(), "capture before any movement");
    hold(&mut h, Modifiers::NONE);
    h.hover_at(b);
    h.run_steps(2);
    transform_press(&mut h, b, false, Modifiers::NONE);
    assert_ne!(text(h.state(), id).transform, Affine::translate(450.0, 350.0));
    assert_eq!(selection(&h), (3, 3));
    let frame = crate::type_transform::frame(h.state_mut()).unwrap();
    let before = text(h.state(), id).transform;
    use crate::canvas::{ToolEvent, tool_event};
    let (a, b) = (frame.quad[2], frame.quad[0]);
    tool_event(h.state_mut(), ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
    tool_event(h.state_mut(), ToolEvent::Up { x: b[0], y: b[1] }, type_command());
    assert_eq!(text(h.state(), id).transform, before, "collapsed scale is rejected");
    let mut saved = serde_json::to_value(&h.state().ui).unwrap();
    saved["type_transform"] = json!({"invalid": "restored pointer capture"});
    let restored: crate::state::UiState = serde_json::from_value(saved).unwrap();
    assert!(restored.type_transform.is_none());
}

/// Render on demand with the same offscreen wgpu infrastructure as the snapshot example.
#[test]
#[ignore = "writes visual evidence to target/issue-1091; run explicitly for visual QA"]
fn temporary_type_transform_visual_evidence() {
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/issue-1091");
    std::fs::create_dir_all(&out).unwrap();
    for (index, theme) in crate::theme::ThemeKind::ALL.into_iter().enumerate() {
        let (app, _) = transform_app(false, false, Affine::translate(450.0, 350.0));
        let mut h = make_harness(if index == 0 { 2.0 } else { 1.0 }, app, true);
        let ctx = h.ctx.clone();
        h.state_mut().set_theme(&ctx, theme);
        h.run_steps(4);
        h.render().unwrap().save(out.join(format!("{index}-normal.png"))).unwrap();
        hold(&mut h, type_command());
        h.render().unwrap().save(out.join(format!("{index}-handles.png"))).unwrap();
        let f = crate::type_transform::frame(h.state_mut()).unwrap();
        let mut handles = f.quad.to_vec();
        handles.extend((0..4).map(|i| [(f.quad[i][0] + f.quad[(i + 1) % 4][0]) / 2.0, (f.quad[i][1] + f.quad[(i + 1) % 4][1]) / 2.0]));
        for (i, q) in handles.into_iter().enumerate() {
            h.hover_at(frame_point(h.state(), q));
            h.run_steps(2);
            h.render().unwrap().save(out.join(format!("{index}-handle-{i}.png"))).unwrap();
        }
        let a = frame_point(h.state(), f.quad[2]);
        transform_press(&mut h, a, true, type_command());
        h.hover_at(a + vec2(70.0, 35.0));
        h.run_steps(2);
        h.render().unwrap().save(out.join(format!("{index}-scaling.png"))).unwrap();
        crate::type_transform::cancel_drag(h.state_mut());
        transform_press(&mut h, a + vec2(70.0, 35.0), false, type_command());
        h.state_mut().ui.type_transform_pivot = None;
        let ed = h.state().ui.text_edit.as_ref().unwrap().layer;
        h.state_mut().run("type.edit", json!({"layer": ed, "transform": Affine::translate(450.0, 350.0).mul(&Affine::rotate(-0.3)).m})).unwrap();
        h.run_steps(3);
        h.render().unwrap().save(out.join(format!("{index}-rotated.png"))).unwrap();
        let f = crate::type_transform::frame(h.state_mut()).unwrap();
        let q = [f.quad[1][0] + 20.0, f.quad[1][1] - 15.0];
        let a = frame_point(h.state(), q);
        h.hover_at(a);
        h.run_steps(2);
        transform_press(&mut h, a, true, type_command());
        let b = frame_point(h.state(), [q[0] + 35.0, q[1] + 60.0]);
        h.hover_at(b);
        h.run_steps(2);
        h.render().unwrap().save(out.join(format!("{index}-rotating.png"))).unwrap();
        transform_press(&mut h, b, false, type_command());
        h.state_mut().run("type.edit", json!({"layer": ed, "box": [400, 300, 260, 180]})).unwrap();
        crate::type_transform::reset(h.state_mut());
        hold(&mut h, Modifiers::NONE);
        h.run_steps(3);
        h.render().unwrap().save(out.join(format!("{index}-paragraph.png"))).unwrap();
        hold(&mut h, type_command());
        h.render().unwrap().save(out.join(format!("{index}-paragraph-transform.png"))).unwrap();
    }
}

#[test]
fn temporary_type_transform_imported_psd_retains_editability_and_styles() {
    let (mut app, original_id) = transform_app(false, false, Affine { m: [1.5, 0.2, 0.1, 0.8, 400.0, 350.0] });
    app.run("type.setStyle", json!({"layer": original_id.0, "range": [0, 5], "fauxBold": true, "color": "#3366cc", "tracking": 25, "align": "center"}))
        .unwrap();
    app.run("layer.renameLayer", json!({"layer": original_id.0, "name": "Styled caption"})).unwrap();
    app.run("layer.layerStyle.dropShadow", json!({"layer": original_id.0, "distance": 8, "size": 4})).unwrap();
    let exported = photocraft_io::export(&app.session.active().unwrap().doc, "type.psd", &Default::default()).unwrap();
    let imported = photocraft_io::import("type.psd", &exported.bytes).unwrap().document;
    let mut app = new_app();
    app.session.add_document(imported, None);
    app.sync_views();
    let id = app.session.active().unwrap().doc.walk().into_iter().find(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).unwrap().2.id;
    app.session.select_layer(id).unwrap();
    super::edit_active(&mut app).unwrap();
    let before = text(&app, id);
    assert!(before.runs.len() > 1, "mixed character styles were imported");
    let layer_before = app.session.active().unwrap().doc.layer(id).unwrap().clone();
    assert!(!layer_before.effects.items.is_empty());
    let f = crate::type_transform::frame(&mut app).unwrap();
    use crate::canvas::{ToolEvent, tool_event};
    let a = f.quad[2];
    tool_event(&mut app, ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
    tool_event(&mut app, ToolEvent::Up { x: a[0] + 35.0, y: a[1] + 10.0 }, type_command());
    let after = text(&app, id);
    assert_eq!((after.text.as_str(), &after.runs, &after.paragraphs), (before.text.as_str(), &before.runs, &before.paragraphs));
    assert!(after.psd_raw.is_some());
    let layer_after = app.session.active().unwrap().doc.layer(id).unwrap();
    assert_eq!(layer_after.name, layer_before.name);
    assert_eq!(layer_after.effects, layer_before.effects);
    assert!(app.ui.text_edit.is_some());
    assert!(app.ui.transform.is_none());
    super::cancel(&mut app);
    assert_affine(text(&app, id).transform, before.transform);
}

#[test]
#[ignore = "release benchmark for issue #1091: 24 MP, eight text layers, 60 preview updates"]
fn temporary_type_transform_large_document_preview() {
    use crate::canvas::{ToolEvent, tool_event};
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 6000, "height": 4000})).unwrap();
    for i in 0..8 {
        app.run("type.create", json!({"text": "PhotoCraft transform preview", "size": 60, "x": 300, "y": 300 + i * 200})).unwrap();
    }
    super::edit_active(&mut app).unwrap();
    let frame = crate::type_transform::frame(&mut app).unwrap();
    let a = frame.quad[2];
    let revision = app.session.active().unwrap().revision;
    let doc = app.session.active().unwrap().doc.id;
    let index = app.session.active_index().unwrap();
    let count = app.session.active().unwrap().history.entries().len();
    let source = (*app.session.active().unwrap().doc).clone();
    let layer = app.ui.text_edit.as_ref().unwrap().layer;
    tool_event(&mut app, ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, type_command());
    let mut times = Vec::new();
    let mut transforms = Vec::new();
    let mut before = 0;
    for i in 1..=60 {
        let started = std::time::Instant::now();
        tool_event(&mut app, ToolEvent::Move { x: a[0] + f64::from(i) * 2.0, y: a[1] + f64::from(i), pressure: 1.0 }, type_command());
        let (_, after) = crate::type_transform::display_doc(&mut app, index).unwrap();
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        transforms.push(crate::type_transform::current_transform(&app, LayerId(layer)).unwrap());
        let damage = crate::type_transform::damage(&app, doc, revision, before, after).unwrap();
        assert!(i64::from(damage.width()) * i64::from(damage.height()) < 2_000_000, "preview only damages the type layer");
        let (_, same) = crate::type_transform::display_doc(&mut app, index).unwrap();
        assert_eq!(same, after, "stationary pointer reuses the rendered preview");
        before = after;
    }
    assert_eq!(app.session.active().unwrap().revision, revision);
    assert_eq!(app.session.active().unwrap().history.entries().len(), count);
    times.sort_by(f64::total_cmp);
    println!(
        "24 MP / 8 type layers / 60 previews: p50 {:.3} ms, p95 {:.3} ms, max {:.3} ms (render + gesture; composite excluded)",
        times[29], times[56], times[59]
    );
    // Compare the same matrices with the existing live-command path on the same document.
    let mut commands = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    commands.session.add_document(source, None);
    let mut command_times = Vec::new();
    for transform in transforms {
        let started = std::time::Instant::now();
        commands.run("type.edit", json!({"layer": layer, "transform": transform.m, "coalesce": "1091-bench"})).unwrap();
        command_times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    command_times.sort_by(f64::total_cmp);
    println!(
        "existing type.edit per update: p50 {:.3} ms, p95 {:.3} ms, max {:.3} ms (command + render; composite excluded)",
        command_times[29], command_times[56], command_times[59]
    );
    crate::type_transform::cancel_drag(&mut app);
}

#[test]
fn font_styles_keep_metadata_names_and_numeric_labels() {
    let styles = super::styles("Inter");
    assert!(styles.contains(&"Regular".into()));
    assert!(styles.contains(&"SemiBold".into()));
    assert_eq!(super::style_label("20"), "20");
    assert_eq!(super::style_label("30"), "30");
    assert_eq!(super::styles("Missing test family"), ["Regular"]);
}

#[test]
fn postscript_only_style_shows_actual_subfamily() {
    let style = photocraft_doc::text::CharStyle { font_family: "Inter".into(), postscript_name: Some("Inter-SemiBold".into()), ..Default::default() };
    assert_eq!(super::selected_style(&style), "SemiBold");
}

/// A variable face lists every standard weight of its `wght` axis, not only its default instance
/// (Montserrat from Google Fonts: its default instance is Thin).
#[test]
fn variable_faces_list_the_weights_of_their_axis() {
    let face = |weight: f32, italic: bool, axes: Vec<(String, f32, f32, f32)>| {
        let base = match weight as i32 {
            100 => "Thin",
            300 => "Light",
            _ => "Bold",
        };
        let style = if italic { format!("{base} Italic") } else { base.to_string() };
        photocraft_text::FaceInfo { family: "Montserrat".into(), style, postscript_name: None, weight, italic, axes }
    };
    let full = || vec![("wght".to_string(), 100.0, 100.0, 900.0)];
    let names = super::style_names(&[face(100.0, false, full()), face(100.0, true, full())]);
    assert_eq!(names.len(), 18, "{names:?}");
    assert_eq!(names.first().map(String::as_str), Some("Thin"));
    for s in ["Regular", "Italic", "Bold", "Bold Italic", "Black Italic"] {
        assert!(names.iter().any(|n| n == s), "{s}: {names:?}");
    }
    // A narrower axis lists only its range; a static face only itself.
    let names = super::style_names(&[face(300.0, false, vec![("wght".into(), 300.0, 400.0, 700.0)])]);
    assert_eq!(names, ["Light", "Regular", "Medium", "SemiBold", "Bold"]);
    assert_eq!(super::style_names(&[face(700.0, false, Vec::new())]), ["Bold"]);
}

/// A variable font's axes become sliders (optical size excepted: it follows the type size), and
/// each shows the style's value for it, or what the layout draws without one (#2985).
#[test]
fn variable_font_axes_become_sliders_with_the_drawn_value() {
    use photocraft_doc::text::{CharStyle, FontVariation};
    let face = |italic: bool, axes: Vec<super::Axis>| photocraft_text::FaceInfo {
        family: "Roboto Flex".into(),
        style: "Regular".into(),
        postscript_name: None,
        weight: 400.0,
        italic,
        axes,
    };
    let ax = |tag: &str, min: f32, def: f32, max: f32| (tag.to_string(), min, def, max);
    let faces = [
        face(false, vec![ax("opsz", 8.0, 14.0, 144.0), ax("wght", 100.0, 400.0, 1000.0), ax("wdth", 25.0, 100.0, 151.0)]),
        face(true, vec![ax("slnt", -10.0, 0.0, 0.0)]),
    ];
    let tags = |v: Vec<super::Axis>| v.into_iter().map(|a| a.0).collect::<Vec<_>>();
    assert_eq!(tags(super::slider_axes(&faces, false)), ["wght", "wdth"]);
    assert_eq!(tags(super::slider_axes(&faces, true)), ["slnt"]);
    let c = CharStyle { weight: 700, variations: vec![FontVariation { axis: "wdth".into(), value: 300.0 }], ..Default::default() };
    assert_eq!(super::axis_value(&c, &ax("wght", 100.0, 400.0, 1000.0)), 700.0, "the style's weight");
    assert_eq!(super::axis_value(&c, &ax("wdth", 25.0, 100.0, 151.0)), 151.0, "its own value, within the range");
    assert_eq!(super::axis_value(&c, &ax("GRAD", -200.0, 0.0, 150.0)), 0.0, "the axis default");
}

/// Families the host serves are in the font menu before they are fetched.
#[test]
fn the_font_menu_lists_served_families() {
    photocraft_text::served::add_families(["Served Menu Test Serif".to_string()]);
    assert!(super::families().iter().any(|f| f == "Served Menu Test Serif"));
}

/// #2236: formatting applies to the selected type layers, while an active text edit stays local.
mod multi_layer_formatting {
    use super::*;
    use egui::accesskit::Role;
    use photocraft_doc::text::{AntiAlias, CharStyle, Orientation, TextAlign};

    fn selected(app: &mut PhotocraftApp, ids: &[LayerId]) {
        for (i, id) in ids.iter().enumerate() {
            app.run("layer.select", json!({"layer": id.0, "mode": if i == 0 { "replace" } else { "add" }})).unwrap();
        }
    }

    fn fixture() -> (PhotocraftApp, [LayerId; 3]) {
        let mut app = new_app();
        let ids = std::array::from_fn(|i| {
            LayerId(
                app.run("type.create", json!({"text": "AéBCD", "font": "Inter", "size": 12 + i * 4, "color": "#000000"})).unwrap()["layer"].as_u64().unwrap(),
            )
        });
        app.run("type.setStyle", json!({"layer": ids[0].0, "fontStyle": "Bold Italic", "tracking": 20})).unwrap();
        selected(&mut app, &ids[..2]);
        (app, ids)
    }

    fn style(app: &PhotocraftApp, id: LayerId, ci: usize) -> CharStyle {
        let t = text(app, id);
        let byte = t.text.char_indices().nth(ci).unwrap().0;
        let mut end = 0;
        t.char_runs()
            .into_iter()
            .find(|r| {
                end += r.len;
                byte < end
            })
            .unwrap()
            .style
    }

    fn controls(app: PhotocraftApp, properties: bool) -> Harness<'static, PhotocraftApp> {
        let mut h = Harness::builder().with_size(vec2(1100.0, 800.0)).build_ui_state(
            move |ui, app| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    if properties {
                        ui.set_width(340.0);
                        super::super::type_properties(app, ui);
                    } else {
                        ui.horizontal(|ui| super::super::options_bar(app, ui));
                    }
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, Default::default());
        h.run_steps(4);
        h
    }

    fn number(h: &mut Harness<'static, PhotocraftApp>, label: &str, value: &str) {
        // Variable-font axes add numeric fields before size/leading. Select the setting by
        // its accessible name, so a font scan cannot shift this action onto another control.
        h.get_by_role_and_label(Role::SpinButton, label).click();
        h.run();
        h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
        h.event(egui::Event::Text(value.into()));
        h.key_press(egui::Key::Enter);
        h.run();
    }

    fn editing(app: &mut PhotocraftApp, id: LayerId) {
        app.ui.text_edit = Some(crate::state::TextEdit {
            layer: id.0,
            caret: 3,
            anchor: 1,
            session: "batch-type-edit".into(),
            created: false,
            mask: false,
            dragging: false,
            resize: None,
            preedit: None,
        });
    }

    #[test]
    fn options_font_family_preserves_peer_faces_and_batches_layer_flags() {
        let (app, ids) = fixture();
        let before = ids.map(|id| style(&app, id, 0));
        let mut h = controls(app, false);
        h.get_all_by_role(Role::ComboBox).next().unwrap().click();
        h.run();
        h.get_by_role(Role::TextInput).type_text("JetBrains");
        h.key_press(egui::Key::ArrowDown);
        h.run();
        h.key_press(egui::Key::Enter);
        h.run();
        for i in 0..2 {
            let mut expected = before[i].clone();
            expected.font_family = "JetBrains Mono".into();
            // #3556: the new family's installed face at the peer's weight and italic is named (no
            // name without such a face), so a PSD names a real font.
            expected.postscript_name = photocraft_text::shared()
                .lock()
                .unwrap()
                .fonts
                .faces("JetBrains Mono")
                .into_iter()
                .find(|f| f.italic == expected.italic && f.weight.round() as u16 == expected.weight)
                .and_then(|f| f.postscript_name);
            assert_eq!(style(h.state(), ids[i], 0), expected, "family changes preserve each peer's face and metrics");
        }
        assert_eq!(style(h.state(), ids[2], 0), before[2]);
        h.get_all_by_role(Role::ComboBox).nth(2).unwrap().click();
        h.run();
        h.get_by_label("Sharp").click();
        h.run();
        h.get_by_role_and_label(Role::Button, "Toggle text orientation").click();
        h.run();
        for id in &ids[..2] {
            let t = text(h.state(), *id);
            assert_eq!(t.antialias, AntiAlias::Sharp);
            assert_eq!(t.orientation, Orientation::Vertical);
        }
        assert_eq!(text(h.state(), ids[2]).orientation, Orientation::Horizontal);
        assert_ne!(text(h.state(), ids[2]).antialias, AntiAlias::Sharp);
    }

    /// Other engines registering fonts and the served catalog growing (tests running in parallel)
    /// used to move the key of the font menu's preview cache, so its samples re-rendered and asked
    /// for a repaint every pass and `Harness::run` never settled (a flaky `exceeded max_steps`).
    /// Here both happen on every pass.
    #[test]
    fn the_font_menu_settles_while_other_engines_register_fonts() {
        let (app, _) = fixture();
        let mut pass = 0u32;
        let mut h = Harness::builder().with_size(vec2(1100.0, 800.0)).build_ui_state(
            move |ui, app| {
                photocraft_text::TextEngine::new();
                // The served catalog is process-wide: names that sort last.
                photocraft_text::served::add_families([format!("zz Churn Test {pass:03}")]);
                pass = pass.saturating_add(1);
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    ui.horizontal(|ui| super::super::options_bar(app, ui));
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, Default::default());
        h.run_steps(4);
        h.get_all_by_role(Role::ComboBox).next().unwrap().click();
        h.run();
        h.run();
    }

    #[test]
    fn properties_edit_shown_metrics_and_paragraphs_with_a_non_type_primary() {
        let (mut app, ids) = fixture();
        let untouched = style(&app, ids[2], 0);
        for (id, scale) in [(ids[0], 2.0), (ids[1], 4.0)] {
            app.run("type.edit", json!({"layer": id.0, "transform": [scale, 0.0, 0.0, scale, 0.0, 0.0]})).unwrap();
        }
        let raster = LayerId(app.run("layer.new.layer", json!({"name": "Selected raster"})).unwrap()["layer"].as_u64().unwrap());
        selected(&mut app, &[ids[0], ids[1], raster]);
        let mut h = controls(app, true);
        number(&mut h, "Font size", "60");
        // Differ from the automatic 72 pt leading so this is an actual property change.
        number(&mut h, "Leading (set to the font size × auto-leading when Auto)", "84");
        h.get_by_label("Faux Bold").click();
        h.run();
        h.get_by_label("Center text").click();
        h.run();
        h.get_by_label("Hyphenate").click();
        h.run();
        number(&mut h, "Indent left margin", "9");
        for (id, scale) in [(ids[0], 2.0), (ids[1], 4.0)] {
            let c = style(h.state(), id, 0);
            assert_eq!(c.size_pt * scale, 60.0);
            assert_eq!(c.leading_pt.unwrap() * scale, 84.0);
            assert!(c.faux_bold);
            let p = text(h.state(), id).paragraph_runs()[0].style.clone();
            assert_eq!(p.align, TextAlign::Center);
            assert!(p.hyphenate);
            assert_eq!(p.start_indent_pt, 9.0);
        }
        assert_eq!(style(h.state(), ids[2], 0), untouched);
        assert_eq!(h.state().session.active().unwrap().active_layer, Some(raster));
        // The same controls edit only the selected Unicode span during a text session.
        let peer = text(h.state(), ids[1]).char_runs();
        editing(h.state_mut(), ids[0]);
        h.run();
        number(&mut h, "Font size", "40");
        assert_eq!((size_at(h.state(), ids[0], 0), size_at(h.state(), ids[0], 1), size_at(h.state(), ids[0], 3)), (30.0, 20.0, 30.0));
        assert_eq!(text(h.state(), ids[1]).char_runs(), peer);
    }

    #[test]
    fn formatting_menus_isolate_the_edit_layer_and_range_but_honor_explicit_targets() {
        let (mut app, ids) = fixture();
        let ctx = egui::Context::default();
        let peer = text(&app, ids[1]);
        editing(&mut app, ids[0]);
        for command in ["type.antiAlias.sharp", "type.orientation.vertical", "type.openType.discretionaryLigatures"] {
            crate::menus::invoke(&mut app, &ctx, command, json!({})).unwrap();
        }
        assert_eq!(text(&app, ids[0]).antialias, AntiAlias::Sharp);
        assert_eq!(text(&app, ids[0]).orientation, Orientation::Vertical);
        assert!(!style(&app, ids[0], 0).discretionary_ligatures);
        assert!(style(&app, ids[0], 1).discretionary_ligatures);
        assert!(!style(&app, ids[0], 3).discretionary_ligatures);
        assert_eq!(text(&app, ids[1]).char_runs(), peer.char_runs());
        assert_eq!(text(&app, ids[1]).antialias, peer.antialias);
        assert_eq!(text(&app, ids[1]).orientation, peer.orientation);
        crate::menus::invoke(&mut app, &ctx, "type.antiAlias.crisp", json!({"layer": ids[1].0})).unwrap();
        crate::menus::invoke(&mut app, &ctx, "type.orientation.vertical", json!({"layers": [ids[2].0]})).unwrap();
        assert_eq!(text(&app, ids[1]).antialias, AntiAlias::Crisp);
        assert_eq!(text(&app, ids[2]).orientation, Orientation::Vertical);
        assert_eq!(text(&app, ids[0]).antialias, AntiAlias::Sharp);
    }

    /// The Character panel's OpenType buttons (TYPE-218-5, #3752): Photoshop's eight, in its
    /// order; each shows pressed while its feature is on and toggles it on the selected type
    /// layers, or on the selected characters while typing.
    #[test]
    fn character_panel_opentype_buttons_toggle_and_show_the_feature() {
        use egui::accesskit::Toggled;
        use egui_kittest::kittest::NodeT;
        use photocraft_engine::type_extra_cmds::feature_on;
        let (app, ids) = fixture();
        let untouched = style(&app, ids[2], 0);
        let mut h = controls(app, true);
        let shown = |h: &Harness<'static, PhotocraftApp>, label: &str| h.get_by_label(label).accesskit_node().toggled();
        let order: Vec<String> = h
            .get_all_by_role(Role::Button)
            .filter_map(|n| n.accesskit_node().label())
            .filter(|l| super::super::OPENTYPE_BUTTONS.iter().any(|(_, _, tip)| tip == l))
            .collect();
        let expected: Vec<&str> = super::super::OPENTYPE_BUTTONS.iter().map(|(_, _, tip)| *tip).collect();
        assert_eq!(order, expected, "every OpenType button, in Photoshop's order");
        assert_eq!(shown(&h, "Standard Ligatures"), Some(Toggled::True), "ligatures are on by default");
        assert_eq!(shown(&h, "Fractions"), Some(Toggled::False));
        h.get_by_label("Fractions").click();
        h.run();
        for id in &ids[..2] {
            assert!(feature_on(&style(h.state(), *id, 0), "frac"), "selected layers get the feature");
        }
        assert_eq!(style(h.state(), ids[2], 0), untouched, "an unselected layer is untouched");
        h.run();
        assert_eq!(shown(&h, "Fractions"), Some(Toggled::True), "the button shows the feature on");
        h.get_by_label("Standard Ligatures").click();
        h.run();
        h.run();
        assert!(!style(h.state(), ids[0], 0).ligatures && !style(h.state(), ids[1], 0).ligatures);
        assert_eq!(shown(&h, "Standard Ligatures"), Some(Toggled::False));
        // While typing, only the selected characters (1..3) change.
        editing(h.state_mut(), ids[0]);
        h.run();
        h.get_by_label("Swash").click();
        h.run();
        let swash = |h: &Harness<'static, PhotocraftApp>, ci| feature_on(&style(h.state(), ids[0], ci), "swsh");
        assert!(!swash(&h, 0) && swash(&h, 1) && swash(&h, 2) && !swash(&h, 3));
        h.run();
        assert_eq!(shown(&h, "Swash"), Some(Toggled::True), "shown at the selection start");
    }

    /// Type › Anti-Alias and Type › Orientation check the edited (else selected) layer's value.
    #[test]
    fn type_menu_checks_the_current_anti_alias_and_orientation() {
        let (mut app, ids) = fixture();
        app.run("type.antiAlias.none", json!({"layer": ids[0].0})).unwrap();
        app.run("type.orientation.vertical", json!({"layer": ids[0].0})).unwrap();
        app.run("type.antiAlias.crisp", json!({"layer": ids[1].0})).unwrap();
        app.run("type.orientation.horizontal", json!({"layer": ids[1].0})).unwrap();
        let checks = |app: &PhotocraftApp| -> Vec<String> {
            let items = crate::menus::menu_items(app);
            for id in ["type.antiAlias.none", "type.antiAlias.crisp", "type.orientation.horizontal", "type.orientation.vertical"] {
                let it = items.iter().find(|i| i.id == id).unwrap();
                assert!(it.checked.is_some(), "{id} is a check item");
            }
            items
                .iter()
                .filter(|i| (i.id.starts_with("type.antiAlias.") || i.id.starts_with("type.orientation.")) && i.checked == Some(true))
                .map(|i| i.id.clone())
                .collect()
        };
        editing(&mut app, ids[0]);
        assert_eq!(checks(&app), ["type.antiAlias.none", "type.orientation.vertical"]);
        app.ui.text_edit = None;
        app.session.active_mut().unwrap().active_layer = Some(ids[1]);
        assert_eq!(checks(&app), ["type.antiAlias.crisp", "type.orientation.horizontal"]);
    }

    #[test]
    fn warp_dialog_captures_layer_selection_or_inline_edit_before_confirmation() {
        for inline in [false, true] {
            let (mut app, ids) = fixture();
            if inline {
                editing(&mut app, ids[0]);
            }
            let styles = ids.map(|id| text(&app, id).char_runs());
            let steps = app.session.active().unwrap().history.entries().len();
            crate::menus::invoke(&mut app, &egui::Context::default(), "type.warpText", json!({})).unwrap();
            let dialog = app.ui.dialogs.last().unwrap().id;
            let fields = &app.ui.dialogs.last().unwrap().fields;
            if inline {
                assert_eq!(fields["layer"], ids[0].0);
                assert_eq!(fields["coalesce"], "batch-type-edit");
            } else {
                let targets = fields["layers"].as_array().unwrap();
                assert_eq!(targets.len(), 2);
                assert!(targets.contains(&json!(ids[0].0)) && targets.contains(&json!(ids[1].0)));
            }
            let mut h = Harness::builder().with_size(vec2(900.0, 700.0)).build_ui_state(
                |ui, app| {
                    if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                        crate::dialogs::show(app, ui.ctx());
                    }
                },
                app,
            );
            PhotocraftApp::setup_context(&h.ctx, Default::default());
            h.run_steps(4);
            assert!(h.query_by_label("Bend").is_some(), "the actual Warp form is visible");
            for label in ["Layer", "Layers", "Range", "Coalesce"] {
                assert!(h.query_by_label(label).is_none(), "target metadata must not become a form field: {label}");
            }
            h.state_mut().ui.dialog_mut(dialog).unwrap().fields.insert("style".into(), json!("arc"));
            h.state_mut().ui.dialog_mut(dialog).unwrap().fields.insert("bend".into(), json!(25.0));
            selected(h.state_mut(), &[ids[2]]);
            h.get_by_label("OK").click();
            h.run_steps(3);
            assert!(h.state().ui.dialogs.is_empty());
            for (i, id) in ids.into_iter().enumerate() {
                let t = text(h.state(), id);
                assert_eq!(t.warp.as_ref().map(|w| w.value), (i == 0 || (!inline && i == 1)).then_some(25.0), "inline={inline}, layer={i}");
                assert_eq!(t.char_runs(), styles[i], "Warp leaves character formatting intact");
            }
            assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps + 1);
            if inline {
                assert_eq!(h.state().session.active().unwrap().coalesce.as_deref(), Some("batch-type-edit"));
            }
        }
    }

    #[test]
    fn text_swatch_captures_selected_targets_and_cancel_does_not_edit() {
        let (app, ids) = fixture();
        let originals = ids.map(|id| style(&app, id, 0));
        let mut h = controls(app, false);
        let steps = h.state().session.active().unwrap().history.entries().len();
        for confirm in [false, true] {
            selected(h.state_mut(), &ids[..2]);
            h.run();
            h.get_by_label("Set the text color").click();
            h.run();
            let dialog = h.state().ui.dialogs.last().unwrap().id;
            h.state_mut().ui.dialog_mut(dialog).unwrap().fields.insert("color".into(), json!("#00ff00"));
            selected(h.state_mut(), &[ids[2]]);
            if confirm {
                crate::dialogs::confirm(h.state_mut(), dialog).unwrap();
            } else {
                h.state_mut().ui.close_dialog(dialog).unwrap();
            }
            for i in 0..3 {
                let mut expected = originals[i].clone();
                if confirm && i < 2 {
                    expected.color = photocraft_color::Color::rgb(0.0, 1.0, 0.0);
                }
                assert_eq!(style(h.state(), ids[i], 0), expected);
            }
            assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps + usize::from(confirm));
        }
        assert!(h.state_mut().session.undo());
        for (id, expected) in ids.into_iter().zip(originals) {
            assert_eq!(style(h.state(), id, 0), expected);
        }
    }
}

/// #1132: with the Move tool, double-clicking type on the canvas edits it (Photoshop CC 2019+):
/// the Type tool takes over with the caret on the glyph under the pointer, on whichever type
/// layer was double-clicked, even when another layer is active. Elsewhere it does nothing.
#[test]
fn move_tool_double_click_on_type_edits_it() {
    let mut app = new_app();
    let id = LayerId(app.run("type.create", json!({"text": "HOHOHO", "size": 120, "x": 300, "y": 420})).unwrap()["layer"].as_u64().unwrap());
    app.run("layer.new.layer", json!({})).unwrap();
    app.ui.tool = crate::state::Tool::Move;
    let mut h = fast_harness(app);
    let p = glyph(&mut h, id, 2, 0.2);
    let steps = h.state().session.active().unwrap().history.entries().len();
    double_click(&mut h, p);
    assert_eq!(h.state().ui.tool, crate::state::Tool::Type, "the Type tool takes over");
    assert_eq!(h.state().ui.text_edit.as_ref().map(|e| e.layer), Some(id.0));
    assert_eq!(selection(&h), (2, 2), "the caret sits on the double-clicked glyph");
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(id));
    assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps, "nothing moved");
    super::insert(h.state_mut(), "x");
    assert_eq!(text(h.state(), id).text, "HOxHOHO");
    // Off the text, a double-click stays with the Move tool.
    super::commit(h.state_mut());
    h.state_mut().ui.tool = crate::state::Tool::Move;
    h.run_steps(2);
    let empty = screen(&mut h, id, -400.0, -300.0);
    double_click(&mut h, empty);
    assert_eq!(h.state().ui.tool, crate::state::Tool::Move);
    assert!(h.state().ui.text_edit.is_none());
}

fn selection_bounds(app: &PhotocraftApp) -> Option<photocraft_geom::Rect> {
    app.session.active().unwrap().doc.selection.as_ref().map(|m| m.content_bounds())
}

/// Click with `tool` at (200, 300), type `text` and commit.
fn type_with(app: &mut PhotocraftApp, tool: crate::state::Tool, text: &str) {
    use crate::canvas::{ToolEvent, tool_event};
    app.ui.tool = tool;
    tool_event(app, ToolEvent::Down { x: 200.0, y: 300.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(app, ToolEvent::Up { x: 200.0, y: 300.0 }, Modifiers::NONE);
    super::insert(app, text); // replaces the Type tool's selected placeholder
    super::commit(app);
}

/// #1055: the Type Mask tools type a selection in the shape of the glyphs, the same pixels the
/// Type tool's layer covers, as one history step that adds no layer. Vertical runs down a column.
#[test]
fn type_mask_commits_the_glyphs_as_the_selection_in_one_step() {
    use crate::state::Tool;
    let mut app = new_app();
    type_with(&mut app, Tool::Type, "I");
    let glyphs = app.session.active().unwrap().doc.layers.last().unwrap().surface().unwrap().content_bounds();

    let mut app = new_app();
    let layers = app.session.active().unwrap().doc.layers.len();
    let steps = app.session.active().unwrap().history.entries().len();
    type_with(&mut app, Tool::HorizontalTypeMask, "I");
    assert!(app.ui.text_edit.is_none());
    assert_eq!(selection_bounds(&app), Some(glyphs));
    let st = app.session.active().unwrap();
    assert_eq!(st.doc.layers.len(), layers, "no layer is left behind");
    assert_eq!(st.history.entries().len(), steps + 1);
    assert_eq!(st.history.entries().last().map(String::as_str), Some("Type Mask"));
    app.run("edit.undo", json!({})).unwrap();
    assert_eq!(selection_bounds(&app), None, "one undo restores the selection");
    assert_eq!(app.session.active().unwrap().doc.layers.len(), layers);

    let mut app = new_app();
    type_with(&mut app, Tool::VerticalTypeMask, "HELLO");
    let v = selection_bounds(&app).unwrap();
    assert!(v.height() > 2 * v.width(), "{v:?}");
}

/// #1055: Escape cancels a Type Mask session whatever "Use Escape to Commit" says: the selection
/// and the layers stay as they were.
#[test]
fn type_mask_escape_leaves_the_selection_unchanged() {
    let mut app = new_app();
    app.run("select.all", json!({})).unwrap();
    let (before, layers) = (selection_bounds(&app), app.session.active().unwrap().doc.layers.len());
    app.ui.tool = crate::state::Tool::HorizontalTypeMask;
    let mut h = harness(1.0, app);
    let p = xf(h.state()).to_screen(200.0, 300.0);
    click(&mut h, p);
    assert!(h.state().ui.text_edit.as_ref().is_some_and(|ed| ed.mask), "typing a mask");
    super::insert(h.state_mut(), "I");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);
    assert!(h.state().ui.text_edit.is_none());
    assert_eq!(selection_bounds(h.state()), before);
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), layers);
}

/// A harness with short frames, so a run of clicks stays inside egui's double-click delay.
fn quick_harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1200.0, 800.0)).with_step_dt(0.02).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

/// `n` quick clicks at `p`, then a pause long enough that the next click starts a new run.
fn clicks(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, n: usize) -> (usize, usize) {
    h.hover_at(p);
    h.run_steps(1);
    for _ in 0..n {
        press(h, p, true);
        press(h, p, false);
    }
    let sel = selection(h);
    h.run_steps(40);
    sel
}

/// Centre the view on layer `id` at 50 % zoom.
fn centre_on(h: &mut Harness<'static, PhotocraftApp>, id: LayerId) {
    let (l, aff, _) = layout(h.state_mut(), id).unwrap();
    let b = l.bounds().unwrap();
    let m = aff.apply(Point::new(f64::from(b[0] + b[2]) / 2.0, f64::from(b[1] + b[3]) / 2.0));
    let v = &mut h.state_mut().ui.views[0];
    (v.zoom, v.center, v.fit_pending) = (0.5, [m.x as f32, m.y as f32], false);
    h.run_steps(2);
}

/// Laid-out line (character range) holding character `i`.
fn line_chars(h: &mut Harness<'static, PhotocraftApp>, id: LayerId, i: usize) -> (usize, usize) {
    let (l, _, text) = layout(h.state_mut(), id).unwrap();
    let b = text.char_indices().nth(i).unwrap().0;
    let c = l.clusters.iter().find(|c| c.range.start == b).unwrap();
    let ln = &l.lines[c.line];
    (photocraft_text::char_index(&text, ln.range.start), photocraft_text::char_index(&text, ln.range.end))
}

/// #3493: like Photoshop, two clicks select the word, three the line, four the paragraph (up
/// to the hard return) and five all text; the sixth starts over with a caret.
#[test]
fn multi_click_selects_word_line_paragraph_and_all_in_point_text() {
    let mut app = new_app();
    app.ui.extras.snap = false;
    let s = "alpha beta\ngamma delta\nepsilon";
    let id = LayerId(app.run("type.create", json!({"text": s, "size": 60, "x": 300, "y": 200})).unwrap()["layer"].as_u64().unwrap());
    let mut h = quick_harness(app);
    centre_on(&mut h, id);
    // Glyph 19 is the 'l' of "delta" (characters 17..22), on the line "gamma delta" (11..22).
    let p = glyph(&mut h, id, 19, 0.2);
    assert_eq!(clicks(&mut h, p, 1), (19, 19), "one click: caret");
    assert_eq!(clicks(&mut h, p, 2), (17, 22), "two clicks: word");
    assert_eq!(clicks(&mut h, p, 3), (11, 22), "three clicks: line");
    assert_eq!(clicks(&mut h, p, 4), (11, 22), "four clicks: paragraph without its return");
    assert_eq!(clicks(&mut h, p, 5), (0, s.chars().count()), "five clicks: all text");
    assert_eq!(clicks(&mut h, p, 6), (19, 19), "a sixth click starts over with a caret");
    assert_eq!(clicks(&mut h, p, 7), (17, 22), "and a seventh selects the word again");
    // After a pause the run restarts: a single click places the caret.
    assert_eq!(clicks(&mut h, p, 1), (19, 19), "a click after a pause is a caret");
    // Paste replaces the selected word; ⌘A then selects all of the edited text.
    assert_eq!(clicks(&mut h, p, 2), (17, 22));
    h.event(egui::Event::Paste("omega\nzeta".into()));
    h.run_steps(2);
    let edited = "alpha beta\ngamma omega\nzeta\nepsilon";
    assert_eq!(text(h.state(), id).text, edited, "paste replaces the selection");
    assert_eq!(selection(&h), (27, 27), "caret after the pasted text");
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
    h.run_steps(2);
    assert_eq!(selection(&h), (0, edited.chars().count()), "⌘A selects all text");
}

/// Paragraph text: three clicks select only the visual (soft-wrapped) line, four the whole
/// paragraph; vertical type selects the column the same way.
#[test]
fn multi_click_separates_wrapped_lines_from_paragraphs() {
    let mut app = new_app();
    app.ui.extras.snap = false;
    let text = "one two three four five six\nseven";
    let id = LayerId(app.run("type.create", json!({"text": text, "size": 40, "box": [200, 150, 260, 600]})).unwrap()["layer"].as_u64().unwrap());
    let mut h = quick_harness(app);
    centre_on(&mut h, id);
    // Glyph 9 is inside "three"; the first paragraph wraps, so its line is only part of it.
    let line = line_chars(&mut h, id, 9);
    assert!(line.0 > 0 || line.1 < 27, "the paragraph wraps: line {line:?}");
    let p = glyph(&mut h, id, 9, 0.5);
    assert_eq!(clicks(&mut h, p, 2), (8, 13), "two clicks: the word \"three\"");
    assert_eq!(clicks(&mut h, p, 3), line, "three clicks: the visual line");
    assert_eq!(clicks(&mut h, p, 4), (0, 27), "four clicks: the wrapped paragraph");
    let p = glyph(&mut h, id, 30, 0.5);
    assert_eq!(clicks(&mut h, p, 4), (28, 33), "four clicks in the last paragraph");

    let mut app = new_app();
    app.ui.extras.snap = false;
    let id = LayerId(app.run("type.create", json!({"text": "HOHOHO\nHOHO", "size": 60, "x": 800, "y": 120})).unwrap()["layer"].as_u64().unwrap());
    app.run("type.orientation.vertical", json!({"layer": id.0})).unwrap();
    let mut h = quick_harness(app);
    centre_on(&mut h, id);
    let p = vglyph(&mut h, id, 9, 0.5);
    assert_eq!(clicks(&mut h, p, 3), (7, 11), "three clicks in vertical type: the column");
    assert_eq!(clicks(&mut h, p, 5), (0, 11), "five clicks: all text");
}

#[test]
fn click_counts_cycle_and_restart() {
    use super::next_click_count as next;
    // egui counts 1, 2, 3, 3, 3, ... for a quick run; ours goes on to 5 and then starts over.
    let mut n = 0;
    let mut seen = vec![];
    for egui in [1, 2, 3, 3, 3, 3, 3] {
        n = next(n, egui);
        seen.push(n);
    }
    assert_eq!(seen, [1, 2, 3, 4, 5, 1, 2]);
    // A slow or distant click (egui: 1) restarts at a caret, whatever came before.
    assert_eq!(next(4, 1), 1);
    assert_eq!(next(4, 0), 1);
    // No previous count (another layer, a fresh session): egui's count stands, capped.
    assert_eq!(next(0, 2), 2);
    assert_eq!(next(0, 3), 3);
    assert_eq!(next(0, u32::MAX), 5);
    assert_eq!(next(u32::MAX, 2), 1, "a corrupt count wraps without overflow");
}

#[test]
fn click_selection_without_layout_and_on_hostile_input() {
    use super::click_selection as sel;
    let text = "ab cd\nef gh\r\nij";
    // Without a layout a line falls back to its paragraph.
    assert_eq!(sel(None, text, 7, None, 2), (6, 8));
    assert_eq!(sel(None, text, 7, None, 3), (6, 11));
    assert_eq!(sel(None, text, 7, None, 4), (6, 11), "the CR LF is not selected");
    assert_eq!(sel(None, text, 13, None, 4), (13, 15));
    assert_eq!(sel(None, text, 7, None, 5), (0, 15));
    // Empty layer, caret past the end, absurd counts and line indices: no panic, clamped.
    for count in [0, 1, 2, 3, 4, 5, 6, u32::MAX] {
        assert_eq!(sel(None, "", 99, Some(usize::MAX), count), (0, 0), "empty text, count {count}");
        let (a, b) = sel(None, text, usize::MAX, Some(usize::MAX), count);
        assert!(a <= b && b <= 15, "caret past the end, count {count}: {a}..{b}");
        let empty = photocraft_text::TextLayout::default();
        let (a, b) = sel(Some(&empty), text, usize::MAX, Some(usize::MAX), count);
        assert!(a <= b && b <= 15, "empty layout, count {count}: {a}..{b}");
    }
    // A trailing return ends in an empty paragraph: four clicks there select nothing.
    assert_eq!(sel(None, "ab\n", 3, None, 4), (3, 3));
    // Multi-byte text keeps character offsets.
    assert_eq!(sel(None, "é ü\nñø", 4, None, 2), (4, 6));
    assert_eq!(sel(None, "é ü\nñø", 1, None, 4), (0, 3));
}

/// Clicks with no edit session, or one on a layer that is gone, never panic or select.
#[test]
fn multi_click_without_a_session_does_nothing() {
    let mut app = new_app();
    super::multi_click(&mut app, 3, 10.0, 10.0);
    assert!(app.ui.text_edit.is_none() && app.type_clicks.is_none());
    app.ui.text_edit = Some(crate::state::TextEdit {
        layer: 999_999,
        caret: 5,
        anchor: 5,
        session: "stale".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });
    for egui in [1, 2, 3, 3, 3, 3] {
        super::multi_click(&mut app, egui, f64::NAN, f64::INFINITY);
    }
    let ed = app.ui.text_edit.as_ref().unwrap();
    assert_eq!((ed.anchor, ed.caret), (5, 5));
}

/// #3036: ⇧ on the click that commits a Type Mask adds the glyphs to the selection and ⌥
/// subtracts them, as in Photoshop; while typing, the working text has no Layers panel row.
#[test]
fn type_mask_shift_adds_and_alt_subtracts_without_a_layer_row() {
    use crate::canvas::{ToolEvent, tool_event};
    use crate::state::Tool;
    use photocraft_geom::Rect;
    let mut app = new_app();
    type_with(&mut app, Tool::HorizontalTypeMask, "I");
    let glyphs = selection_bounds(&app).unwrap();
    let mut app = new_app();
    app.ui.extras.snap = false; // the second text would snap to the first one's selected edge
    let rect = Rect::new(1000, 100, 1100, 200);
    app.run("select.rect", json!({"x": 1000, "y": 100, "width": 100, "height": 100})).unwrap();
    // Type, then commit with a click outside the text holding `m`.
    let type_and_commit = |app: &mut PhotocraftApp, m: Modifiers| {
        app.ui.tool = Tool::HorizontalTypeMask;
        tool_event(app, ToolEvent::Down { x: 200.0, y: 300.0, pressure: 1.0 }, Modifiers::NONE);
        tool_event(app, ToolEvent::Up { x: 200.0, y: 300.0 }, Modifiers::NONE);
        super::insert(app, "I");
        let doc = app.session.active().unwrap().doc.clone();
        let rows = crate::layer_tree_ui::panel_rows(app, &doc);
        assert_eq!(rows.len(), doc.layers.len() - 1, "the working text is not listed");
        tool_event(app, ToolEvent::Down { x: 1500.0, y: 800.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: 1500.0, y: 800.0 }, m);
        assert!(app.ui.text_edit.is_none());
    };
    let at = |app: &PhotocraftApp, x: i32, y: i32| app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |s| s.sample_channel(x, y, 0));
    let stem = ((glyphs.x0 + glyphs.x1) / 2, (glyphs.y0 + glyphs.y1) / 2);
    type_and_commit(&mut app, Modifiers::SHIFT);
    assert_eq!(selection_bounds(&app), Some(rect.union(&glyphs)), "⇧ adds");
    assert!(at(&app, stem.0, stem.1) > 0.99 && at(&app, 1050, 150) > 0.99);
    type_and_commit(&mut app, Modifiers::ALT);
    assert!(at(&app, stem.0, stem.1) < 0.01, "⌥ subtracts the glyphs");
    assert!(at(&app, 1050, 150) > 0.99, "and keeps the rest");
}
