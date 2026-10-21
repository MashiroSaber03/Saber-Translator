//! Delete Layer never leaves a document without layers (#1107): the single-layer path refuses
//! like the multi-select path does, and changes nothing.

use photocraft_engine::Session;
use serde_json::json;

fn one_layer_doc() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 10, "height": 5, "background": "transparent"})).unwrap();
    s
}

#[test]
fn deleting_the_only_layer_is_refused() {
    let mut s = one_layer_doc();
    let id = s.active().unwrap().doc.layers[0].id;
    let undo_before = s.active().unwrap().history.can_undo();
    let r = s.execute("layer.delete", json!({"layer": id.0}));
    assert!(r.is_err(), "deleted the last layer ({r:?})");
    let st = s.active().unwrap();
    assert_eq!(st.doc.layers.len(), 1);
    assert_eq!(st.active_layer, Some(id));
    assert_eq!(st.history.can_undo(), undo_before, "a refused delete left a history step");
}

#[test]
fn deleting_the_only_layer_without_a_param_is_refused() {
    let mut s = one_layer_doc();
    assert!(s.execute("layer.delete", json!({})).is_err());
    assert_eq!(s.active().unwrap().doc.layers.len(), 1);
}

#[test]
fn deleting_a_group_that_holds_every_layer_is_refused() {
    let mut s = one_layer_doc();
    let r = s.execute("layer.groupLayers", json!({})).unwrap();
    let group = r["layer"].as_u64().unwrap_or_else(|| s.active().unwrap().active_layer.unwrap().0);
    assert_eq!(s.active().unwrap().doc.layers.len(), 1, "the group should be the only top-level layer");
    assert!(s.execute("layer.delete", json!({"layer": group})).is_err());
    assert_eq!(s.active().unwrap().doc.layers.len(), 1);
}

#[test]
fn deleting_one_of_two_layers_still_works() {
    let mut s = one_layer_doc();
    s.execute("layer.new.layer", json!({})).unwrap();
    let top = s.active().unwrap().active_layer.unwrap();
    s.execute("layer.delete", json!({"layer": top.0})).unwrap();
    let st = s.active().unwrap();
    assert_eq!(st.doc.layers.len(), 1);
    assert!(st.doc.layer(top).is_none());
    assert_eq!(st.active_layer, Some(st.doc.layers[0].id));
}
