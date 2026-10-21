use photocraft_geom::Rect;
use serde_json::json;

use super::*;

/// An 80×60 transparent document whose layer is painted red over (10..30)²; (10..30)² selected.
fn session() -> (Session, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 80, "height": 60, "background": "transparent"})).unwrap();
    let id = s.active().unwrap().active_layer.unwrap();
    s.edit("paint", |doc, _| {
        doc.layer_mut(id).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 10, 30, 30), &[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    (s, id)
}

fn alpha(d: &Document, id: LayerId, x: i32, y: i32) -> f32 {
    d.layer(id).unwrap().surface().unwrap().rgba(x, y)[3]
}

#[test]
fn float_moves_a_cut_piece_without_touching_the_document_until_dropped() {
    let (mut s, id) = session();
    let steps = s.active().unwrap().history.past_len();
    assert_eq!(s.execute("select.float", json!({"dx": 15, "dy": 0})).unwrap()["offset"], json!([15, 0]));
    // Moving it again doesn't cut anything new.
    s.execute("select.float", json!({"dx": 0, "dy": 10})).unwrap();
    let st = s.active().unwrap();
    assert_eq!(floating(st).unwrap().offset, (15, 10));
    assert_eq!(alpha(&st.doc, id, 12, 12), 1.0, "the document is untouched while floating");
    assert_eq!(st.history.past_len(), steps);
    let shown = displayed(st, (0, 0)).unwrap();
    assert!(alpha(&shown, id, 12, 12) == 0.0 && alpha(&shown, id, 27, 22) == 1.0, "shown cut and moved");
    // Drop: one history step, pixels and selection where they were shown.
    s.execute("select.drop", json!({})).unwrap();
    let st = s.active().unwrap();
    assert!(floating(st).is_none());
    assert_eq!(st.history.past_len(), steps + 1);
    for (x, y) in [(12, 12), (27, 22), (44, 39), (40, 40)] {
        assert_eq!(alpha(&st.doc, id, x, y), alpha(&shown, id, x, y), "dropped as shown at ({x}, {y})");
    }
    assert_eq!(st.doc.selection.as_ref().unwrap().content_bounds(), Rect::new(25, 20, 45, 40));
    // Undo takes the whole move back.
    s.undo();
    assert_eq!(alpha(&s.active().unwrap().doc, id, 12, 12), 1.0);
}

#[test]
fn any_other_command_drops_it_and_undo_puts_it_back() {
    let (mut s, id) = session();
    let steps = s.active().unwrap().history.past_len();
    s.execute("select.float", json!({"dx": 20, "dy": 0})).unwrap();
    // Undo while floating: the piece goes back; nothing in the document or history changed.
    assert_eq!(s.execute("edit.undo", json!({})).unwrap()["floating"], json!("returned"));
    let st = s.active().unwrap();
    assert!(floating(st).is_none() && alpha(&st.doc, id, 12, 12) == 1.0 && st.history.past_len() == steps);
    // Float again, then deselect: dropped first (where it was shown), then deselected.
    s.execute("select.float", json!({"dx": 20, "dy": 0})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let st = s.active().unwrap();
    assert!(st.doc.selection.is_none() && floating(st).is_none());
    assert!(alpha(&st.doc, id, 12, 12) == 0.0 && alpha(&st.doc, id, 35, 20) == 1.0);
}

#[test]
fn needs_a_selection_and_an_unlocked_pixel_layer() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 20, "height": 20})).unwrap();
    assert!(!s.is_enabled("select.float"), "no selection");
    s.execute("select.all", json!({})).unwrap();
    assert!(s.is_enabled("select.float"), "the Background's selected pixels move");
    s.execute("layer.new.layer", json!({})).unwrap();
    assert!(s.is_enabled("select.float"));
    s.execute("layer.lockLayers", json!({"position": true})).unwrap();
    assert!(!s.is_enabled("select.float"), "a position-locked layer");
}

/// On the Background, which can't hold transparency, the hole takes the background colour.
#[test]
fn on_the_background_the_hole_takes_the_background_colour() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 40})).unwrap();
    s.tools.background = [0.0, 0.0, 1.0, 1.0];
    let id = s.active().unwrap().active_layer.unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
    s.execute("select.float", json!({"dx": 15, "dy": 0})).unwrap();
    s.execute("select.drop", json!({})).unwrap();
    let px = |x, y| s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().rgba(x, y);
    assert_eq!(px(12, 12), [0.0, 0.0, 1.0, 1.0], "the hole");
    assert_eq!(px(27, 12), [1.0, 1.0, 1.0, 1.0], "the moved pixels");
    assert_eq!(px(5, 5), [1.0, 1.0, 1.0, 1.0], "the rest");
}

/// Copying leaves the original pixels, at every depth; the move is one undo step.
#[test]
fn a_copy_leaves_the_original_at_every_depth_in_one_undo_step() {
    for depth in [8, 16, 32] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 80, "height": 60, "background": "transparent", "depth": depth})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        let steps = s.active().unwrap().history.past_len();
        s.execute("select.float", json!({"dx": 30, "dy": 0, "copy": true})).unwrap();
        s.execute("select.drop", json!({})).unwrap();
        let st = s.active().unwrap();
        assert!(alpha(&st.doc, id, 12, 12) == 1.0 && alpha(&st.doc, id, 42, 12) == 1.0, "{depth}-bit: both");
        assert_eq!((st.history.past_len(), st.history.undo_label()), (steps + 1, Some("Move")), "{depth}-bit");
        s.undo();
        let st = s.active().unwrap();
        assert!(alpha(&st.doc, id, 12, 12) == 1.0 && alpha(&st.doc, id, 42, 12) == 0.0, "{depth}-bit: undone");
    }
}

/// `cargo test --release -p photocraft-engine float_drag_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn float_drag_bench() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 6000, "height": 4000, "background": "transparent"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("paint", |doc, a| {
        doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 6000, 4000), &[0.2, 0.4, 0.6, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("select.rect", json!({"x": 1000, "y": 1000, "width": 500, "height": 400})).unwrap();
    let t = std::time::Instant::now();
    s.execute("select.float", json!({})).unwrap();
    let cut = t.elapsed();
    let t = std::time::Instant::now();
    for i in 0..20 {
        displayed(s.active().unwrap(), (i * 7, i * 3)).unwrap();
    }
    eprintln!("6000×4000 layer, 500×400 piece: cut {cut:?}, then {:?} per pointer move", t.elapsed() / 20);
}

#[test]
fn copy_floats_a_duplicate_and_leaves_the_original() {
    let (mut s, id) = session();
    s.execute("select.float", json!({"dx": 30, "dy": 0, "copy": true})).unwrap();
    let shown = displayed(s.active().unwrap(), (0, 0)).unwrap();
    assert!(alpha(&shown, id, 12, 12) == 1.0 && alpha(&shown, id, 42, 12) == 1.0, "original kept, copy shown");
    s.execute("select.drop", json!({})).unwrap();
    let doc = &s.active().unwrap().doc;
    assert!(alpha(doc, id, 12, 12) == 1.0 && alpha(doc, id, 42, 12) == 1.0, "dropped: both");
}

#[test]
fn float_with_the_mask_target_moves_the_mask_not_the_layer() {
    let (mut s, id) = session();
    s.edit("mask", |doc, _| {
        let l = doc.layer_mut(id).unwrap();
        let mut m = photocraft_doc::LayerMask::reveal_all();
        m.surface.fill_rect(Rect::new(10, 10, 30, 30), &[0.0]);
        l.mask = Some(m);
        Ok(())
    })
    .unwrap();
    let mask_at = |s: &Session, x, y| s.active().unwrap().doc.layer(id).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0);
    s.execute("select.float", json!({"dx": 25, "dy": 0, "copy": true, "target": "mask"})).unwrap();
    s.execute("select.drop", json!({})).unwrap();
    assert_eq!(mask_at(&s, 12, 12), 0.0, "the copy leaves the mask where it was");
    assert_eq!(mask_at(&s, 37, 12), 0.0, "the mask's selected part moved");
    assert_eq!(mask_at(&s, 60, 12), 1.0);
    assert_eq!(alpha(&s.active().unwrap().doc, id, 37, 12), 0.0, "the layer's pixels did not move");
    assert_eq!(alpha(&s.active().unwrap().doc, id, 12, 12), 1.0);
}

#[test]
fn cutting_an_untouched_part_of_the_mask_moves_it_and_leaves_the_background_colour() {
    let (mut s, id) = session();
    s.edit("mask", |doc, _| {
        let mut m = photocraft_doc::LayerMask::hide_all();
        m.surface.fill_rect(Rect::new(50, 10, 70, 30), &[1.0]);
        doc.layer_mut(id).unwrap().mask = Some(m);
        Ok(())
    })
    .unwrap();
    s.tools.background = [0.5, 0.5, 0.5, 1.0];
    let mask_at = |s: &Session, x, y| s.active().unwrap().doc.layer(id).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0);
    // (10..30)² is the mask's untouched black; it moves over the white at (50..70, 10..30).
    s.execute("select.float", json!({"dx": 40, "dy": 0, "target": "mask"})).unwrap();
    s.execute("select.drop", json!({})).unwrap();
    assert_eq!(mask_at(&s, 55, 12), 0.0, "the untouched black moved");
    assert!((mask_at(&s, 12, 12) - 0.5).abs() < 0.01, "the hole takes the background colour");
    assert_eq!(mask_at(&s, 40, 12), 0.0);
}
