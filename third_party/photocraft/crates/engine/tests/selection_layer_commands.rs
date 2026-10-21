//! Adding a layer mask, Paste Layer Style and Clear Layer Style act on every selected layer that can
//! take them (one history step), not only on the active one. An explicit `layer`, or a single
//! selected layer, behaves exactly as before.

use photocraft_doc::LayerId;
use photocraft_engine::Session;
use serde_json::{Value, json};

fn id_of(r: Value) -> LayerId {
    LayerId(r["layer"].as_u64().unwrap())
}

struct Scene {
    s: Session,
    bg: LayerId,
    r1: LayerId,
    /// Already has a layer mask.
    r2: LayerId,
    /// Has a drop shadow.
    fx: LayerId,
    /// Has a drop shadow.
    fx2: LayerId,
}

fn scene() -> Scene {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    let bg = s.active().unwrap().doc.layers[0].id;
    let r1 = id_of(s.execute("layer.new.layer", json!({"name": "R1"})).unwrap());
    let r2 = id_of(s.execute("layer.new.layer", json!({"name": "R2"})).unwrap());
    s.execute("layer.layerMask.hideAll", json!({"layer": r2.0})).unwrap();
    let fx = id_of(s.execute("layer.new.layer", json!({"name": "FX"})).unwrap());
    s.execute("layer.layerStyle.dropShadow", json!({"layer": fx.0})).unwrap();
    let fx2 = id_of(s.execute("layer.new.layer", json!({"name": "FX2"})).unwrap());
    s.execute("layer.layerStyle.dropShadow", json!({"layer": fx2.0})).unwrap();
    Scene { s, bg, r1, r2, fx, fx2 }
}

impl Scene {
    fn select(&mut self, ids: &[LayerId]) {
        self.s.execute("layer.select", json!({"layer": ids[0].0})).unwrap();
        for id in &ids[1..] {
            self.s.execute("layer.select", json!({"layer": id.0, "mode": "add"})).unwrap();
        }
    }
    fn has_mask(&self, id: LayerId) -> bool {
        self.s.active().unwrap().doc.layer(id).unwrap().mask.is_some()
    }
    fn fx_count(&self, id: LayerId) -> usize {
        self.s.active().unwrap().doc.layer(id).unwrap().effects.items.len()
    }
    fn revision(&self) -> u64 {
        self.s.active().unwrap().revision
    }
}

#[test]
fn add_layer_mask_goes_to_every_selected_layer_that_can_take_one() {
    for cmd in ["layer.layerMask.revealAll", "layer.layerMask.hideAll"] {
        let mut sc = scene();
        // Background (cannot be masked here), a masked layer (keeps its mask) and two more.
        sc.select(&[sc.bg, sc.r1, sc.r2, sc.fx]);
        let masked_before = sc.s.active().unwrap().doc.layer(sc.r2).unwrap().mask.clone();
        sc.s.execute(cmd, json!({})).unwrap();
        assert!(sc.has_mask(sc.r1) && sc.has_mask(sc.fx), "{cmd}: the layers without a mask get one");
        assert!(!sc.has_mask(sc.bg), "{cmd}: the Background is skipped");
        assert_eq!(sc.s.active().unwrap().doc.layer(sc.r2).unwrap().mask, masked_before, "{cmd}: an existing mask is not replaced");
        // One history step brings it all back.
        assert!(sc.s.undo());
        assert!(!sc.has_mask(sc.r1) && !sc.has_mask(sc.fx));
    }
}

#[test]
fn hide_selection_masks_every_selected_layer() {
    let mut sc = scene();
    sc.select(&[sc.r1, sc.fx]);
    sc.s.execute("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 10})).unwrap();
    sc.s.execute("layer.layerMask.hideSelection", json!({})).unwrap();
    assert!(sc.has_mask(sc.r1) && sc.has_mask(sc.fx));
}

#[test]
fn add_layer_mask_with_nothing_to_mask_is_an_error_not_a_step() {
    let mut sc = scene();
    sc.s.execute("layer.layerMask.hideAll", json!({"layer": sc.fx.0})).unwrap();
    sc.select(&[sc.r2, sc.fx]);
    let revision = sc.revision();
    let err = sc.s.execute("layer.layerMask.revealAll", json!({})).unwrap_err().to_string();
    assert!(err.contains("none of the selected layers"), "{err}");
    assert_eq!(sc.revision(), revision);
}

#[test]
fn add_layer_mask_with_a_layer_param_or_one_selected_layer_is_unchanged() {
    let mut sc = scene();
    sc.select(&[sc.r1, sc.fx]);
    sc.s.execute("layer.layerMask.revealAll", json!({"layer": sc.r1.0})).unwrap();
    assert!(sc.has_mask(sc.r1) && !sc.has_mask(sc.fx), "an explicit layer is the only target");
    // A single layer that already has a mask is still replaced, as before.
    sc.select(&[sc.r2]);
    sc.s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    assert!(sc.has_mask(sc.r2));
}

#[test]
fn paste_layer_style_goes_to_every_selected_layer_but_the_background() {
    let mut sc = scene();
    sc.s.execute("layer.layerStyle.copyLayerStyle", json!({"layer": sc.fx.0})).unwrap();
    sc.select(&[sc.bg, sc.r1, sc.r2]);
    sc.s.execute("layer.layerStyle.pasteLayerStyle", json!({})).unwrap();
    assert_eq!((sc.fx_count(sc.r1), sc.fx_count(sc.r2), sc.fx_count(sc.bg)), (1, 1, 0));
    assert!(sc.s.undo());
    assert_eq!((sc.fx_count(sc.r1), sc.fx_count(sc.r2)), (0, 0), "one history step");
    // An explicit layer is the only target.
    sc.select(&[sc.r1, sc.r2]);
    sc.s.execute("layer.layerStyle.pasteLayerStyle", json!({"layer": sc.r1.0})).unwrap();
    assert_eq!((sc.fx_count(sc.r1), sc.fx_count(sc.r2)), (1, 0));
}

#[test]
fn clear_layer_style_goes_to_every_selected_layer_that_has_one() {
    let mut sc = scene();
    sc.select(&[sc.fx, sc.fx2, sc.r1]);
    sc.s.execute("layer.layerStyle.clear", json!({})).unwrap();
    assert_eq!((sc.fx_count(sc.fx), sc.fx_count(sc.fx2)), (0, 0));
    assert!(sc.s.undo());
    assert_eq!((sc.fx_count(sc.fx), sc.fx_count(sc.fx2)), (1, 1), "one history step");
    // Nothing selected has a style: an error and no step.
    sc.select(&[sc.r1, sc.r2]);
    let revision = sc.revision();
    let err = sc.s.execute("layer.layerStyle.clear", json!({})).unwrap_err().to_string();
    assert!(err.contains("none of the selected layers"), "{err}");
    assert_eq!(sc.revision(), revision);
    // One layer, or an explicit layer, as before (clearing a layer without a style is allowed).
    sc.select(&[sc.fx, sc.fx2]);
    sc.s.execute("layer.layerStyle.clear", json!({"layer": sc.fx.0})).unwrap();
    assert_eq!((sc.fx_count(sc.fx), sc.fx_count(sc.fx2)), (0, 1));
    sc.select(&[sc.r1]);
    sc.s.execute("layer.layerStyle.clear", json!({})).unwrap();
}

#[test]
fn bad_targets_fail_gracefully() {
    let mut s = Session::new();
    for id in ["layer.layerMask.revealAll", "layer.layerMask.hideAll", "layer.layerStyle.pasteLayerStyle", "layer.layerStyle.clear"] {
        assert!(s.execute(id, json!({})).is_err(), "{id}: no document");
    }
    let mut sc = scene();
    for id in ["layer.layerMask.revealAll", "layer.layerStyle.pasteLayerStyle", "layer.layerStyle.clear"] {
        assert!(sc.s.execute(id, json!({"layer": u64::MAX})).is_err(), "{id}: no such layer");
    }
}
