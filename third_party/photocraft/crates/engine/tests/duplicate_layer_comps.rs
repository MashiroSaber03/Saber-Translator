//! Layer comps must continue targeting their layers after a copied document gets fresh ids.
//! Cover current/old history states, nested layers, Last Document State and merged-only copies.

use photocraft_doc::{CompLayerState, Layer, LayerId};
use photocraft_engine::Session;
use serde_json::json;

fn scene() -> (Session, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "white"})).unwrap();
    let id = LayerId(s.execute("layer.new.layer", json!({"name": "L"})).unwrap()["layer"].as_u64().unwrap());
    (s, id)
}

fn copied_layer(s: &Session, name: &str) -> LayerId {
    s.active().unwrap().doc.walk().into_iter().find(|(_, _, l)| l.name == name).unwrap().2.id
}

#[test]
fn image_duplicate_and_history_copy_keep_saved_comp_visibility() {
    for command in ["image.duplicate", "history.newDocument"] {
        let (mut s, source) = scene();
        let original = s.active().unwrap().doc.id;
        s.execute("layerComp.new", json!({"name": "Shown"})).unwrap();
        s.execute("layer.setProps", json!({"layer": source.0, "visible": false})).unwrap();
        s.execute(command, json!({})).unwrap();

        let id = copied_layer(&s, "L");
        let doc = &s.active().unwrap().doc;
        assert_ne!(doc.id, original);
        assert_ne!(id, source, "{command}: layers must get new identities");
        assert!(!doc.layer(id).unwrap().visible, "{command}: copied current layout");
        let comp = doc.layer_comps.iter().find(|c| c.name == "Shown").unwrap();
        assert!(comp.missing_layers(doc).is_empty(), "{command}: all captured layers still exist");
        assert_eq!(comp.state(id).unwrap().visible, Some(true), "{command}: preserve recorded state");

        let applied = s.execute("layerComp.apply", json!({"comp": "Shown"})).unwrap();
        assert_eq!(applied["missingLayers"], 0, "{command}: no spurious missing layers");
        assert!(s.active().unwrap().doc.layer(id).unwrap().visible, "{command}: recorded visibility restored");

        s.set_active(0);
        assert!(!s.active().unwrap().doc.layer(source).unwrap().visible, "{command}: source untouched");
        assert_eq!(s.execute("layerComp.apply", json!({"comp": "Shown"})).unwrap()["missingLayers"], 0);
        assert!(s.active().unwrap().doc.layer(source).unwrap().visible, "{command}: source comp still works");
    }
}

#[test]
fn a_historical_state_uses_its_own_saved_comp_and_fresh_layer_ids() {
    let (mut s, source) = scene();
    s.execute("layerComp.new", json!({"name": "Shown"})).unwrap();
    s.execute("layer.setProps", json!({"layer": source.0, "visible": false})).unwrap();
    let saved_state = s.active().unwrap().history.past_len();
    s.execute("layer.setProps", json!({"layer": source.0, "visible": true})).unwrap();

    s.execute("history.newDocument", json!({"state": saved_state, "name": "From History"})).unwrap();
    assert_eq!(s.active().unwrap().doc.name, "From History");
    let id = copied_layer(&s, "L");
    assert_ne!(id, source);
    assert!(!s.active().unwrap().doc.layer(id).unwrap().visible, "the historical hidden layout survives");
    assert_eq!(s.execute("layerComp.apply", json!({"comp": "Shown"})).unwrap()["missingLayers"], 0);
    assert!(s.active().unwrap().doc.layer(id).unwrap().visible, "historical comp can restore visibility");
    s.set_active(0);
    assert!(s.active().unwrap().doc.layer(source).unwrap().visible, "current source state remains unchanged");
}

#[test]
fn nested_group_states_are_remapped_but_genuinely_missing_layers_stay_missing() {
    let (mut s, _) = scene();
    let (group, child) = s
        .edit("Add nested group", |doc, _| {
            let child = Layer::raster("Child", doc.pixel_format());
            let child_id = child.id;
            let group = Layer::group("Group", vec![child]);
            let group_id = group.id;
            doc.layers.push(group);
            Ok((group_id, child_id))
        })
        .unwrap();
    s.execute("layerComp.new", json!({"name": "Nested"})).unwrap();
    // An old comp may already refer to a deleted layer before duplication.
    // Copying should not silently rewrite that warning to an unrelated new id.
    let missing = LayerId(u64::MAX);
    s.edit("Record deleted layer", |doc, _| {
        let comp = doc.layer_comps.iter_mut().find(|c| c.name == "Nested").unwrap();
        comp.states.push(CompLayerState { layer: missing, visible: Some(false), position: None, appearance: None });
        Ok(())
    })
    .unwrap();
    s.execute("layer.setProps", json!({"layer": child.0, "visible": false})).unwrap();
    s.execute("image.duplicate", json!({})).unwrap();

    let new_group = copied_layer(&s, "Group");
    let new_child = copied_layer(&s, "Child");
    assert_ne!(new_group, group);
    assert_ne!(new_child, child);
    let doc = &s.active().unwrap().doc;
    let comp = doc.layer_comps.iter().find(|c| c.name == "Nested").unwrap();
    assert!(comp.state(new_group).is_some());
    assert_eq!(comp.state(new_child).unwrap().visible, Some(true));
    assert_eq!(comp.missing_layers(doc), vec![missing]);
    assert_eq!(s.execute("layerComp.apply", json!({"comp": "Nested"})).unwrap()["missingLayers"], 1);
    assert!(s.active().unwrap().doc.layer(new_child).unwrap().visible);
}

#[test]
fn duplicated_last_document_state_restores_the_copied_layout() {
    for command in ["image.duplicate", "history.newDocument"] {
        let (mut s, source) = scene();
        s.execute("layerComp.new", json!({"name": "Shown"})).unwrap();
        s.execute("layer.setProps", json!({"layer": source.0, "visible": false})).unwrap();
        s.execute("layerComp.apply", json!({"comp": "Shown"})).unwrap();
        assert!(s.active().unwrap().doc.last_document_state.is_some());
        s.execute(command, json!({})).unwrap();

        let new_id = copied_layer(&s, "L");
        let doc = &s.active().unwrap().doc;
        assert!(doc.layer(new_id).unwrap().visible);
        let backup = doc.last_document_state.as_ref().unwrap();
        assert_eq!(backup.state(new_id).unwrap().visible, Some(false));
        assert!(backup.missing_layers(doc).is_empty(), "{command}: backup references copied ids");

        s.execute("layerComp.restoreLastDocumentState", json!({})).unwrap();
        assert!(!s.active().unwrap().doc.layer(new_id).unwrap().visible, "{command}: backup restores hidden layout");
        s.set_active(0);
        assert!(s.active().unwrap().doc.layer(source).unwrap().visible, "{command}: source remains visible");
    }
}

#[test]
fn merged_only_duplicate_does_not_carry_comps_for_removed_source_layers() {
    let (mut s, source) = scene();
    s.execute("layerComp.new", json!({"name": "Original"})).unwrap();
    s.execute("layer.setProps", json!({"layer": source.0, "visible": false})).unwrap();
    s.execute("layerComp.apply", json!({})).unwrap();
    assert!(s.active().unwrap().doc.last_document_state.is_some());

    s.execute("image.duplicate", json!({"mergedOnly": true})).unwrap();
    let d = &s.active().unwrap().doc;
    assert_eq!(d.layers.len(), 1, "merged-only copy is a single composite layer");
    assert!(d.layer_comps.is_empty(), "old comps cannot address discarded layers");
    assert_eq!(d.last_applied_comp, None);
    assert!(d.last_document_state.is_none(), "no stale saved layout on a flattened copy");
    assert!(!s.is_enabled("layerComp.apply"));

    s.set_active(0);
    assert_eq!(s.active().unwrap().doc.layer_comps.len(), 1, "original comps are preserved");
    assert!(s.active().unwrap().doc.last_document_state.is_some(), "original backup is preserved");
}
