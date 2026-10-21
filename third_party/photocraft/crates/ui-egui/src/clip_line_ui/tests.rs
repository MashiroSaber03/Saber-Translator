//! #967: ⌥-click the line between two layers clips / releases the upper one.

use egui::{Modifiers, PointerButton, Pos2, Rect, pos2, vec2};
use egui_kittest::Harness;
use photocraft_doc::{Document, LayerContent, LayerId};
use serde_json::json;

use super::at;
use crate::PhotocraftApp;
use crate::layer_row_ui::{RowRects, recorded};

/// Background, `a` and `b` above it.
fn two_layers() -> (photocraft_engine::Session, LayerId, LayerId) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let a = s.execute("layer.new.layer", json!({"name": "a"})).unwrap()["layer"].as_u64().unwrap();
    let b = s.execute("layer.new.layer", json!({"name": "b"})).unwrap()["layer"].as_u64().unwrap();
    (s, LayerId(a), LayerId(b))
}

fn harness(session: photocraft_engine::Session) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(session, crate::Services::default())
    });
    h.run_steps(8);
    h
}

/// A real click (press and release on separate frames), modifiers held throughout.
fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, modifiers: Modifiers) {
    h.event(egui::Event::ModifiersChanged(modifiers));
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers });
    h.run_steps(1);
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
}

fn clipped(h: &Harness<'_, PhotocraftApp>, id: LayerId) -> bool {
    h.state().session.active().unwrap().doc.layer(id).unwrap().clipped
}

#[test]
fn alt_click_between_two_layers_clips_and_releases_the_upper_one() {
    let (s, a, b) = two_layers();
    let mut h = harness(s);
    let rows = recorded(&h.ctx);
    let below = rows.iter().find(|r| r.layer == a.0).unwrap().row;
    // On the line between `b` and `a`, right of the eye column.
    let line = pos2(below.left() + 80.0, below.top());
    let undo = h.state().session.active().unwrap().history.entries().len();
    click(&mut h, line, Modifiers::ALT);
    assert!(clipped(&h, b), "⌥-click clips the upper layer to the one below");
    assert!(!clipped(&h, a));
    let st = h.state().session.active().unwrap();
    assert_eq!(st.history.entries().len(), undo + 1, "one history step");
    assert_eq!(st.active_layer, Some(b), "the layer selection is unchanged");
    click(&mut h, line, Modifiers::ALT);
    assert!(!clipped(&h, b), "⌥-click on a clipped layer's line releases it");
    // A plain click there selects a row as before.
    click(&mut h, line, Modifiers::NONE);
    assert!(!clipped(&h, b));
}

#[test]
fn the_clip_cursor_shows_only_with_alt_over_a_line() {
    let (s, a, _) = two_layers();
    let mut h = harness(s);
    let below = recorded(&h.ctx).iter().find(|r| r.layer == a.0).unwrap().row;
    let line = pos2(below.left() + 80.0, below.top());
    h.hover_at(line);
    h.run_steps(2);
    assert_ne!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "no ⌥: the normal pointer");
    h.event(egui::Event::ModifiersChanged(Modifiers::ALT));
    h.run_steps(2);
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "the clip cursor replaces the pointer");
    h.hover_at(below.center());
    h.run_steps(2);
    assert_ne!(h.output().platform_output.cursor_icon, egui::CursorIcon::None, "mid-row: no line");
}

fn row(layer: LayerId, top: f32) -> RowRects {
    RowRects { layer: layer.0, row: Rect::from_min_size(pos2(0.0, top), vec2(240.0, 32.0)), name: None, indicators: Vec::new() }
}

fn named(doc: &Document, name: &str) -> LayerId {
    doc.walk().into_iter().find(|(_, _, l)| l.name == name).unwrap().2.id
}

#[test]
fn only_siblings_next_to_each_other_meet_at_a_line() {
    // Background, `d`, and the group `g { c }` above it.
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({"name": "d"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "c"})).unwrap();
    s.execute("layer.groupLayers", json!({"name": "g"})).unwrap();
    let mut doc: Document = (*s.active().unwrap().doc).clone();
    let [g, c, d, bg] = ["g", "c", "d", "Background"].map(|n| named(&doc, n));
    let open = [row(g, 0.0), row(c, 32.0), row(d, 64.0), row(bg, 96.0)];
    let x = 100.0;
    assert_eq!(at(&doc, &open, pos2(x, 32.0)), None, "a group and its first child");
    assert_eq!(at(&doc, &open, pos2(x, 64.0)), None, "a group's last child and the layer below the group");
    let line = at(&doc, &open, pos2(x, 96.0)).expect("two top-level layers");
    assert_eq!((line.upper, line.release), (d, false));
    assert_eq!(line.commands(&doc, "k"), vec![("layer.createClippingMask".to_string(), json!({"layer": d.0}))]);
    assert_eq!(at(&doc, &open, pos2(10.0, 96.0)), None, "the eye column");
    assert_eq!(at(&doc, &open, pos2(x, 90.0)), None, "inside a row");
    // A closed group sits directly above the layer below it, and can be clipped to it.
    if let Some(LayerContent::Group(grp)) = doc.layer_mut(g).map(|l| &mut l.content) {
        grp.expanded = false;
    }
    let closed = [row(g, 0.0), row(d, 32.0), row(bg, 64.0)];
    assert_eq!(at(&doc, &closed, pos2(x, 32.0)).map(|l| l.upper), Some(g));
    // A filter hiding `d` leaves `g` over the Background: not neighbours in the stack.
    assert_eq!(at(&doc, &[row(g, 0.0), row(bg, 32.0)], pos2(x, 32.0)), None);
    doc.layer_mut(d).unwrap().clipped = true;
    let line = at(&doc, &closed, pos2(x, 64.0)).unwrap();
    assert_eq!(
        line.commands(&doc, "k"),
        vec![
            ("layer.releaseClippingMask".to_string(), json!({"layer": d.0, "coalesce": "k"})),
            ("layer.select".to_string(), json!({"layer": d.0, "mode": "replace"})),
        ]
    );
}

#[test]
fn the_band_reaches_three_above_and_five_below_the_line() {
    // Photoshop 25.4: the line takes the pointer from 3 px above it to 5 px below it.
    let (s, a, b) = two_layers();
    let doc = s.active().unwrap().doc.clone();
    let bg = doc.layers[0].id;
    let rows = [row(b, 0.0), row(a, 32.0), row(bg, 64.0)];
    let x = 100.0;
    for y in [61.0, 64.0, 68.9] {
        assert_eq!(at(&doc, &rows, pos2(x, y)).map(|l| l.upper), Some(a), "y {y}");
    }
    for y in [60.0, 69.5] {
        assert_eq!(at(&doc, &rows, pos2(x, y)), None, "y {y}");
    }
}

/// Background, Layer 1, CB (Color Balance), Top; Top active.
fn chain() -> (photocraft_engine::Session, [LayerId; 3]) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let l1 = s.execute("layer.new.layer", json!({"name": "Layer 1"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.newAdjustmentLayer.colorBalance", json!({})).unwrap();
    let cb = s.active().unwrap().active_layer.unwrap().0;
    let top = s.execute("layer.new.layer", json!({"name": "Top"})).unwrap()["layer"].as_u64().unwrap();
    (s, [LayerId(l1), LayerId(cb), LayerId(top)])
}

/// A point on the line along the top of `layer`'s row, `dy` below it, right of the eye column.
fn on_line(h: &Harness<'_, PhotocraftApp>, layer: LayerId, dy: f32) -> Pos2 {
    let r = recorded(&h.ctx).into_iter().find(|r| r.layer == layer.0).unwrap().row;
    pos2(r.left() + 80.0, r.top() + dy)
}

fn active(h: &Harness<'_, PhotocraftApp>) -> Option<LayerId> {
    h.state().session.active().unwrap().active_layer
}

fn steps(h: &Harness<'_, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().history.entries().len()
}

#[test]
fn releasing_frees_the_clipped_layers_above_too_and_selects_the_layer() {
    // Photoshop 25.4: a release in the middle of a clipping chain frees the layer above the line
    // and every clipped layer stacked above it, in one history step, and makes it active.
    let (s, [l1, cb, top]) = chain();
    let mut h = harness(s);
    // Clip CB onto Layer 1 (from inside the band, 4 px below the line), then Top onto CB.
    let p = on_line(&h, l1, 4.0);
    click(&mut h, p, Modifiers::ALT);
    assert!(clipped(&h, cb) && !clipped(&h, top));
    assert_eq!(active(&h), Some(top), "clipping keeps the active layer");
    let p = on_line(&h, cb, -2.5);
    click(&mut h, p, Modifiers::ALT);
    assert!(clipped(&h, top) && clipped(&h, cb));
    let before = steps(&h);
    let p = on_line(&h, l1, 2.0);
    click(&mut h, p, Modifiers::ALT);
    assert!(!clipped(&h, cb) && !clipped(&h, top), "the clipped layer above is released too");
    assert_eq!(active(&h), Some(cb), "the released layer becomes active");
    assert_eq!(steps(&h), before + 1, "one history step");
    assert!(h.state_mut().session.undo());
    assert!(clipped(&h, cb) && clipped(&h, top), "one Undo brings the chain back");
    // Releasing the top of a chain leaves the layers below it clipped.
    let p = on_line(&h, cb, 0.0);
    click(&mut h, p, Modifiers::ALT);
    assert!(!clipped(&h, top) && clipped(&h, cb));
}

#[test]
fn outside_the_band_an_alt_click_does_not_clip() {
    let (s, [l1, cb, _]) = chain();
    let mut h = harness(s);
    for dy in [-4.5, 6.0] {
        let p = on_line(&h, l1, dy);
        click(&mut h, p, Modifiers::ALT);
        assert!(!clipped(&h, cb), "{dy} px from the line");
    }
}
