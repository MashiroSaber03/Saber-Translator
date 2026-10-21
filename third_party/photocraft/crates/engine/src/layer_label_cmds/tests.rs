use super::*;
use photocraft_color::SampleType;
use photocraft_doc::{Layer, LayerContent};

fn fixture(depth: u8) -> (Session, [LayerId; 4]) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 4, "height": 4, "depth": depth})).unwrap();
    let ids = s
        .edit("Fixture", |doc, _| {
            let mut leaf = Layer::raster("Hidden, locked", doc.pixel_format());
            leaf.visible = false;
            leaf.locks.all = true;
            leaf.label = LabelColor::Blue;
            let leaf_id = leaf.id;
            let mut inner = Layer::group("Inner", vec![leaf]);
            if let LayerContent::Group(g) = &mut inner.content {
                g.expanded = false;
            }
            let inner_id = inner.id;
            let outer = Layer::group("Outer", vec![inner]);
            let outer_id = outer.id;
            let other = Layer::raster("Other", doc.pixel_format());
            let other_id = other.id;
            doc.layers = vec![outer, other];
            Ok([outer_id, inner_id, leaf_id, other_id])
        })
        .unwrap();
    s.select_layer(ids[0]).unwrap();
    (s, ids)
}

#[test]
fn layer_color_groups_and_overlapping_selection_are_one_undo_step() {
    for depth in [8, 16, 32] {
        let (mut s, [outer, inner, leaf, other]) = fixture(depth);
        crate::layer_multi_cmds::set_selection(&mut s, vec![outer, leaf], Some(leaf), Some(outer)).unwrap();
        let before = s.active().unwrap().doc.clone();
        let selected = s.active().unwrap().selected_layers();
        let past = s.active().unwrap().history.past_len();
        let r = s.execute(ID, json!({"color": "seafoam"})).unwrap();
        assert_eq!(r["layers"], json!([outer.0, inner.0, leaf.0]));
        assert_eq!(s.active().unwrap().history.past_len(), past + 1);
        assert_eq!(s.active().unwrap().selected_layers(), selected);
        assert_eq!(s.active().unwrap().last_damage, Some(photocraft_geom::Rect::EMPTY));
        let after = s.active().unwrap().doc.clone();
        for id in [outer, inner, leaf] {
            assert_eq!(after.layer(id).unwrap().label, LabelColor::Seafoam);
        }
        assert_eq!(after.layer(other), before.layer(other));
        let mut expected = (*before).clone();
        for id in [outer, inner, leaf] {
            expected.layer_mut(id).unwrap().label = LabelColor::Seafoam;
        }
        assert_eq!(*after, expected, "only labels change at depth {depth}");
        assert_eq!(crate::inspect::layer(after.layer(leaf).unwrap())["labelColor"], "seafoam");
        assert!(s.undo());
        assert_eq!(s.active().unwrap().doc, before);
        assert!(s.redo());
        assert_eq!(s.active().unwrap().doc, after);
        let past = s.active().unwrap().history.past_len();
        let revision = s.active().unwrap().revision;
        assert_eq!(s.execute(ID, json!({"color": "seafoam"})).unwrap()["changed"], false);
        assert_eq!(s.active().unwrap().history.past_len(), past);
        assert_eq!(s.active().unwrap().revision, revision);
    }
}

#[test]
fn layer_color_every_color_and_reset_target_only_the_requested_layer() {
    let (mut s, [outer, _, leaf, other]) = fixture(8);
    for color in LabelColor::ALL {
        s.execute(ID, json!({"color": color.id(), "layer": other.0})).unwrap();
        let d = &s.active().unwrap().doc;
        assert_eq!(d.layer(other).unwrap().label, color);
        assert_eq!(d.layer(outer).unwrap().label, LabelColor::None);
        assert_eq!(d.layer(leaf).unwrap().label, LabelColor::Blue);
    }
    s.execute(ID, json!({"color": "none", "layer": outer.0})).unwrap();
    assert!(s.active().unwrap().doc.walk().iter().all(|(_, _, l)| l.label == LabelColor::None || l.id == other));
    s.execute(ID, json!({"color": "green", "layer": outer.0})).unwrap();
    s.select_layer(leaf).unwrap();
    s.execute("layer.new.layer", json!({"name": "Created later"})).unwrap();
    let st = s.active().unwrap();
    assert_eq!(st.doc.layer(st.active_layer.unwrap()).unwrap().label, LabelColor::None);
    assert!(matches!(st.doc.layer(outer).unwrap().content, LayerContent::Group(_)));
    assert_eq!(st.doc.depth, SampleType::U8);
}

#[test]
fn layer_color_invalid_params_never_change_the_document_or_history() {
    let (mut s, _) = fixture(8);
    for params in [
        Value::Null,
        json!([]),
        json!({}),
        json!({"color": "unknown"}),
        json!({"color": 1}),
        json!({"color": "red", "layer": "1"}),
        json!({"color": "red", "layer": -1}),
        json!({"color": "red", "layer": 1.5}),
        json!({"color": "red", "layer": null}),
        json!({"color": "red", "layer": u64::MAX}),
    ] {
        let st = s.active().unwrap();
        let before = (st.doc.clone(), st.revision, st.history.past_len());
        assert!(s.execute(ID, params).is_err());
        let st = s.active().unwrap();
        assert_eq!((st.doc.clone(), st.revision, st.history.past_len()), before);
    }
    let mut empty = Session::new();
    assert!(!empty.is_enabled(ID));
    assert!(empty.execute(ID, json!({"color": "red"})).is_err());
    empty.execute("file.new", json!({"width": 2, "height": 2})).unwrap();
    empty
        .edit("Empty", |doc, active| {
            doc.layers.clear();
            *active = None;
            Ok(())
        })
        .unwrap();
    assert!(!empty.is_enabled(ID));
    assert!(empty.execute(ID, json!({"color": "red"})).is_err());
}
