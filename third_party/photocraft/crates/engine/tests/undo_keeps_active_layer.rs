//! Undoing a paint stroke keeps the painted layer active (#1356), as in Photoshop: selecting a
//! layer is not a history step, so the state a step leaves behind targets what was selected just
//! before the step, not what was selected when the state before it was made.

use photocraft_doc::LayerId;
use photocraft_engine::Session;
use serde_json::json;

fn target(s: &Session) -> (Option<LayerId>, Vec<LayerId>) {
    let st = s.active().unwrap();
    (st.active_layer, st.selected_layers.clone())
}

fn one(id: LayerId) -> (Option<LayerId>, Vec<LayerId>) {
    (Some(id), vec![id])
}

#[test]
fn undoing_a_stroke_on_a_rasterized_shape_keeps_it_active() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    let bg = s.active().unwrap().doc.layers[0].id;
    let id = LayerId(s.execute("shape.create", json!({"kind": "ellipse", "rect": [4, 4, 40, 30]})).unwrap()["layer"].as_u64().unwrap());
    // Rasterize the shape layer while the Background is the active layer (Layers panel context
    // menu), then select the rasterized layer and paint on it.
    s.execute("layer.select", json!({"layer": bg.0})).unwrap();
    s.execute("layer.rasterize.layer", json!({"layer": id.0})).unwrap();
    s.execute("layer.select", json!({"layer": id.0})).unwrap();
    s.execute("paint.stroke", json!({"points": [[4, 40], [60, 40]], "size": 6})).unwrap();
    assert!(s.undo());
    assert_eq!(target(&s), one(id), "undo of the stroke left the painted layer (background is {bg:?})");
    assert!(s.redo());
    assert_eq!(target(&s), one(id), "redo targets the painted layer");
    // Undoing the stroke and then the rasterize goes back to what was active before each step.
    assert!(s.undo());
    assert!(s.undo());
    assert_eq!(target(&s), one(bg), "the rasterize was taken with the background active");
}

#[test]
fn undoing_a_fill_keeps_the_filled_layer_active() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    let bg = s.active().unwrap().doc.layers[0].id;
    s.execute("layer.new.layer", json!({})).unwrap();
    let new = target(&s).0.unwrap();
    s.execute("layer.select", json!({"layer": bg.0})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    assert!(s.undo());
    assert_eq!(target(&s), one(bg), "the fill was on the background");
    // Undoing New Layer still goes back to the layer active before it.
    assert!(s.undo());
    assert_eq!(target(&s), one(bg));
    // Redo returns New Layer's state as it was left: the background had been selected in it.
    assert!(s.redo());
    assert_eq!(target(&s), one(bg));
    assert!(s.active().unwrap().doc.layer(new).is_some());
}

#[test]
fn undo_preserves_intentionally_empty_layer_selection() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    let background = s.active().unwrap().doc.layers[0].id;
    s.execute("select.deselectLayers", json!({})).unwrap();
    assert_eq!(target(&s), (None, Vec::new()));

    s.execute("layer.new.layer", json!({"name": "A"})).unwrap();
    let added = target(&s).0.unwrap();
    assert!(s.undo());
    assert_eq!(target(&s), (None, Vec::new()), "undo restores the empty pre-edit target");
    assert!(s.redo());
    assert_eq!(target(&s), one(added), "redo selects the created layer");
    assert!(s.undo());
    assert_eq!(target(&s), (None, Vec::new()), "repeated undo keeps the empty target");

    // A real selected target must still be restored normally.
    s.execute("layer.select", json!({"layer": background.0})).unwrap();
    s.execute("layer.new.layer", json!({"name": "B"})).unwrap();
    assert!(s.undo());
    assert_eq!(target(&s), one(background));
}
