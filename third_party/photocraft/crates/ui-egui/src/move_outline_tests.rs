//! #2413: View › Show › Layer Edges while the Move tool drags. As in Photoshop, the outline belongs
//! to the layer being moved (the one Auto-Select picks on the press, not the one selected before)
//! and moves in lockstep with its pixels, however fast the pointer goes.

use egui::{Event, PointerButton, Pos2};
use egui_kittest::Harness;
use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{LAYER_EDGES, ViewXform};
use crate::state::{Tool, View};

/// A 64×64 document with two painted layers: A covers (8..24)², B covers (40..56)². B is active,
/// the Move tool (Auto-Select on) is current and View › Show › Layer Edges is on.
fn app() -> (PhotocraftApp, LayerId, LayerId) {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    app.sync_views();
    let mut ids = Vec::new();
    for at in [8, 40] {
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(at, at, at + 16, at + 16), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        ids.push(app.session.active().unwrap().active_layer.unwrap());
    }
    app.ui.tool = Tool::Move;
    assert!(app.ui.tool_options.move_auto_select);
    // Snapping would shift the expected offsets.
    app.ui.extras.snap = false;
    app.ui.view.show.smart_guides = false;
    app.ui.extras.rulers = false;
    app.ui.view.extras = true;
    app.ui.view.show.layer_edges = true;
    // 4 screen px per document px, the document centred.
    app.ui.views = vec![View { zoom: 4.0, center: [32.0, 32.0], fit_pending: false, fill_pending: false, doc_size: [64, 64], rotation: 0.0 }];
    (app, ids[0], ids[1])
}

fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(egui::vec2(400.0, 340.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            app.last_canvas_rect = ui.max_rect();
            let v = app.ui.views[0].clone();
            crate::canvas::canvas_view(app, ui, 0, ui.max_rect(), v, true);
        },
        app,
    );
    h.run_steps(2);
    h
}

fn xf(app: &PhotocraftApp) -> ViewXform {
    let v = &app.ui.views[0];
    ViewXform { rect: app.last_canvas_rect, zoom: v.zoom / app.ppp, center: v.center, flip: false, rotation: 0.0, aspect: 1.0 }
}

/// The screen point of document point `p`.
fn at(h: &Harness<'_, PhotocraftApp>, p: [f32; 2]) -> Pos2 {
    xf(h.state()).to_screen(p[0], p[1])
}

/// The Layer Edges outlines the last frame drew.
fn edges(h: &Harness<'_, PhotocraftApp>) -> Vec<egui::Rect> {
    h.output()
        .shapes
        .iter()
        .filter_map(|c| match &c.shape {
            egui::Shape::Rect(r) if r.stroke.color == LAYER_EDGES && r.fill == egui::Color32::TRANSPARENT => Some(r.rect),
            _ => None,
        })
        .collect()
}

/// The outline Layer Edges draws around document rect `r`.
fn outline(h: &Harness<'_, PhotocraftApp>, r: Rect) -> Vec<egui::Rect> {
    vec![xf(h.state()).doc_rect(r)]
}

fn button(h: &mut Harness<'_, PhotocraftApp>, pos: Pos2, pressed: bool) {
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
}

fn active(h: &Harness<'_, PhotocraftApp>) -> Option<LayerId> {
    h.state().session.active().unwrap().active_layer
}

const A: Rect = Rect::new(8, 8, 24, 24);
const B: Rect = Rect::new(40, 40, 56, 56);

#[test]
fn the_outline_belongs_to_the_layer_auto_select_picked() {
    let (app, a, b) = app();
    let mut h = harness(app);
    assert_eq!(active(&h), Some(b));
    assert_eq!(edges(&h), outline(&h, B), "B is selected: its edges show");
    // Press on A and drag it 5 px to the right (20 screen px, past egui's click distance).
    let p0 = at(&h, [16.0, 16.0]);
    h.event(Event::PointerMoved(p0));
    button(&mut h, p0, true);
    h.step();
    h.event(Event::PointerMoved(p0 + egui::vec2(20.0, 0.0)));
    h.step();
    // The frame the drag starts in picks A: it outlines A, never B, which isn't moving.
    assert_eq!(active(&h), Some(a), "Auto-Select picked A");
    assert_eq!(edges(&h), outline(&h, A), "the frame that picks A outlines A, not the previously selected B");
    h.step();
    assert_eq!(edges(&h), outline(&h, A.translate(5, 0)), "A's outline follows it");
    button(&mut h, p0 + egui::vec2(20.0, 0.0), false);
    h.step();
    h.step();
    assert_eq!(edges(&h), outline(&h, A.translate(5, 0)), "after the release the outline stays on A where it landed");
    let st = h.state().session.active().unwrap();
    assert_eq!(st.doc.layer(a).unwrap().surface().unwrap().content_bounds(), A.translate(5, 0));
}

#[test]
fn the_outline_moves_in_lockstep_with_a_fast_drag() {
    let (app, _, b) = app();
    let mut h = harness(app);
    let p0 = at(&h, [48.0, 48.0]);
    h.event(Event::PointerMoved(p0));
    button(&mut h, p0, true);
    h.step();
    h.event(Event::PointerMoved(p0 + egui::vec2(-20.0, 0.0)));
    h.step();
    h.step();
    assert_eq!(active(&h), Some(b));
    assert_eq!(edges(&h), outline(&h, B.translate(-5, 0)));
    // Big jumps, one per frame. A frame draws the layer where the previous frame's events left it
    // (the image is painted before this frame's events are read), and the outline goes with it.
    let mut shown = (-5, 0);
    for (dx, dy) in [(-30, -10), (-36, -30), (-12, -36), (-40, 0)] {
        h.event(Event::PointerMoved(p0 + egui::vec2(dx as f32 * 4.0, dy as f32 * 4.0)));
        h.step();
        assert_eq!(edges(&h), outline(&h, B.translate(shown.0, shown.1)), "drawn with the layer at {shown:?} while the pointer moved to {dx},{dy}");
        // The next frame shows both at the pointer.
        h.step();
        assert_eq!(edges(&h), outline(&h, B.translate(dx, dy)), "the outline is where the layer is shown, at {dx},{dy}");
        shown = (dx, dy);
    }
    button(&mut h, p0 + egui::vec2(-160.0, 0.0), false);
    h.step();
    h.step();
    assert_eq!(edges(&h), outline(&h, B.translate(-40, 0)), "dropped where it was shown");
}
