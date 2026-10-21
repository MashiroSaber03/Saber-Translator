use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, LayerContent, LayerId, Pattern};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::Session;

/// Document 0 (`w`×`h`, `depth`) with a red 10×10 square at (4, 6) on layer "paint" above the
/// Background, and document 1 (60×50, 8-bit RGB). Document 0 is active.
fn two_docs(w: u32, h: u32, depth: u32) -> (Session, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
    let paint = LayerId(s.execute("layer.new.layer", json!({"name": "paint"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("select.rect", json!({"x": 4, "y": 6, "width": 10, "height": 10})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("file.new", json!({"width": 60, "height": 50})).unwrap();
    s.set_active(0);
    (s, paint)
}

fn doc(s: &Session, i: usize) -> &Document {
    &s.documents()[i].doc
}

fn copied(r: &Value) -> LayerId {
    LayerId(r["layers"][0].as_u64().unwrap())
}

fn bounds(s: &Session, i: usize, id: LayerId) -> Rect {
    crate::layer_multi_cmds::layer_bounds(doc(s, i).layer(id).unwrap()).unwrap()
}

#[test]
fn copies_the_selected_layers_above_the_destinations_active_layer_in_one_step() {
    let (mut s, paint) = two_docs(40, 30, 8);
    let source_before = doc(&s, 0).clone();
    let r = s.execute("layer.copyToDocument", json!({"document": 1})).unwrap();
    assert_eq!(r["document"], 1);
    assert_eq!(s.active_index(), Some(1), "the destination becomes the active document");
    let d = doc(&s, 1);
    let id = copied(&r);
    assert_ne!(id, paint, "a copy has its own id");
    assert_eq!(d.layers.len(), 2);
    assert_eq!(d.layers[1].id, id, "above the Background, the destination's active layer");
    assert_eq!(d.layers[1].name, "paint");
    assert_eq!(s.active().unwrap().active_layer, Some(id));
    assert_eq!(bounds(&s, 1, id), Rect::new(4, 6, 14, 16), "same canvas position by default");
    assert_eq!(d.layers[1].surface().unwrap().read_region(Rect::new(5, 7, 6, 8)), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(*doc(&s, 0), source_before, "the source is untouched");
    // One history step in the destination.
    assert!(s.undo());
    assert_eq!(doc(&s, 1).layers.len(), 1);
}

#[test]
fn converts_to_the_destinations_depth_and_colour_mode() {
    for depth in [8, 16, 32] {
        let (mut s, _) = two_docs(40, 30, depth);
        s.set_active(1);
        s.execute("image.mode.grayscale", json!({})).unwrap();
        s.execute("image.mode.bits16", json!({})).unwrap();
        s.set_active(0);
        let r = s.execute("layer.copyToDocument", json!({"document": 1})).unwrap();
        let id = copied(&r);
        let surf = doc(&s, 1).layer(id).unwrap().surface().unwrap();
        assert_eq!(surf.format(), PixelFormat::new(ColorMode::Grayscale, SampleType::U16, true), "from {depth}-bit RGB");
        let px = surf.read_region(Rect::new(5, 7, 6, 8));
        assert!(px[0] > 0.1 && px[0] < 0.9 && px[1] == 1.0, "red as a mid grey, opaque: {px:?}");
    }
}

#[test]
fn places_the_copies_centred_on_the_canvas_on_a_point_or_offset() {
    let (mut s, paint) = two_docs(40, 30, 8);
    let place = |s: &mut Session, p: Value| {
        s.set_active(0);
        let mut p = p;
        p["document"] = json!(1);
        p["layers"] = json!([paint.0]);
        let r = s.execute("layer.copyToDocument", p).unwrap();
        bounds(s, 1, copied(&r))
    };
    assert_eq!(place(&mut s, json!({"center": true})), Rect::new(25, 20, 35, 30));
    assert_eq!(place(&mut s, json!({"at": [10.0, 40.0]})), Rect::new(5, 35, 15, 45));
    assert_eq!(place(&mut s, json!({"offset": [-4, 20]})), Rect::new(0, 26, 10, 36));
    assert_eq!(doc(&s, 1).layers.len(), 4);
}

#[test]
fn a_copied_background_is_an_ordinary_layer_and_several_layers_keep_their_order() {
    let (mut s, paint) = two_docs(40, 30, 8);
    let bg = doc(&s, 0).layers[0].id;
    let r = s.execute("layer.copyToDocument", json!({"document": 1, "layers": [paint.0, bg]})).unwrap();
    let d = doc(&s, 1);
    let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Background", "Layer 1", "paint"]);
    assert!(!d.layers[1].locks.transparency && !d.layers[1].locks.position);
    assert_eq!(r["layers"].as_array().unwrap().len(), 2);
    assert_eq!(s.active().unwrap().selected_layers().len(), 2, "the copies are selected");
}

#[test]
fn the_sources_patterns_come_along_once() {
    let (mut s, _) = two_docs(40, 30, 8);
    let mut tile = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::U8, true));
    tile.fill_rect(Rect::new(0, 0, 2, 2), &[0.0, 1.0, 0.0, 1.0]);
    let pat = Pattern::new("dots", tile, 2, 2);
    s.edit("add pattern", |d, _| {
        d.patterns.push(pat.clone());
        Ok(())
    })
    .unwrap();
    s.execute("layer.copyToDocument", json!({"document": 1})).unwrap();
    assert_eq!(doc(&s, 1).patterns.iter().filter(|p| p.id == pat.id).count(), 1);
    s.set_active(0);
    s.execute("layer.copyToDocument", json!({"document": 1})).unwrap();
    assert_eq!(doc(&s, 1).patterns.iter().filter(|p| p.id == pat.id).count(), 1, "not twice");
}

#[test]
fn bad_params_and_states_fail_without_changing_anything() {
    let (mut s, _) = two_docs(40, 30, 8);
    let cmd = "layer.copyToDocument";
    for p in [
        json!({}),
        json!({"document": 0}),
        json!({"document": 7}),
        json!({"document": -1}),
        json!({"document": "1"}),
        json!({"document": 1, "source": 9}),
        json!({"document": 1, "layers": [u64::MAX]}),
        json!({"document": 1, "layers": "all"}),
        json!({"document": 1, "layers": []}),
        json!({"document": 1, "at": [1.0]}),
        json!({"document": 1, "at": [f64::MAX, 0.0]}),
        json!({"document": 1, "offset": ["a", 2]}),
        Value::Null,
    ] {
        assert!(s.execute(cmd, p.clone()).is_err(), "{p}");
        assert_eq!(s.active_index(), Some(0), "{p}: the source stays active");
        assert_eq!(doc(&s, 1).layers.len(), 1, "{p}");
    }
    // An Indexed destination has no layers to add to.
    s.set_active(1);
    s.edit("indexed", |d, _| {
        d.mode = ColorMode::Indexed;
        Ok(())
    })
    .unwrap();
    s.set_active(0);
    assert!(s.execute(cmd, json!({"document": 1})).is_err());
    // A single open document: nothing to copy to.
    s.close(1);
    assert!(!s.is_enabled(cmd));
}

#[test]
fn copying_linked_layers_does_not_join_an_unrelated_destination_group() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    let src_a = LayerId(s.execute("layer.new.layer", json!({"name": "Source A"})).unwrap()["layer"].as_u64().unwrap());
    let src_b = LayerId(s.execute("layer.new.layer", json!({"name": "Source B"})).unwrap()["layer"].as_u64().unwrap());
    s.edit("link source", |d, _| {
        d.layer_mut(src_a).unwrap().link_group = Some(1);
        d.layer_mut(src_b).unwrap().link_group = Some(1);
        Ok(())
    })
    .unwrap();

    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    let dst_a = LayerId(s.execute("layer.new.layer", json!({"name": "Destination A"})).unwrap()["layer"].as_u64().unwrap());
    let dst_b = LayerId(s.execute("layer.new.layer", json!({"name": "Destination B"})).unwrap()["layer"].as_u64().unwrap());
    s.edit("link destination", |d, _| {
        d.layer_mut(dst_a).unwrap().link_group = Some(1);
        d.layer_mut(dst_b).unwrap().link_group = Some(1);
        Ok(())
    })
    .unwrap();
    s.set_active(0);
    let result = s.execute("layer.copyToDocument", json!({"document": 1, "layers": [src_a.0, src_b.0]})).unwrap();
    let ids: Vec<LayerId> = result["layers"].as_array().unwrap().iter().map(|id| LayerId(id.as_u64().unwrap())).collect();
    assert_eq!(ids.len(), 2);
    let d = doc(&s, 1);
    assert_eq!(d.layer(dst_a).unwrap().link_group, Some(1));
    assert_eq!(d.layer(dst_b).unwrap().link_group, Some(1));
    let first_group = d.layer(ids[0]).unwrap().link_group;
    assert_eq!(first_group, d.layer(ids[1]).unwrap().link_group, "the copied pair stays linked");
    assert!(first_group.is_some());
    assert_ne!(first_group, Some(1), "copied links must not merge with existing ones");

    assert!(s.undo());
    assert!(doc(&s, 1).layer(ids[0]).is_none());
    assert!(s.redo());
    assert_eq!(doc(&s, 1).layer(ids[0]).unwrap().link_group, first_group);
}

#[test]
fn copied_link_groups_are_distinct_from_each_other_and_destination() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    let a = LayerId(s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap());
    let b = LayerId(s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap());
    s.edit("link independent source groups", |d, _| {
        d.layer_mut(a).unwrap().link_group = Some(1);
        d.layer_mut(b).unwrap().link_group = Some(2);
        Ok(())
    })
    .unwrap();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    s.set_active(0);
    let copied = s.execute("layer.copyToDocument", json!({"document": 1, "layers": [a.0, b.0]})).unwrap();
    let ids: Vec<LayerId> = copied["layers"].as_array().unwrap().iter().map(|id| LayerId(id.as_u64().unwrap())).collect();
    let first = doc(&s, 1).layer(ids[0]).unwrap().link_group;
    let second = doc(&s, 1).layer(ids[1]).unwrap().link_group;
    assert!(first.is_some() && second.is_some());
    assert_ne!(first, second);
}

/// One painted layer, `depth`-bit RGB, as the only layer above the Background.
fn one_doc(depth: u32) -> (Session, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "depth": depth})).unwrap();
    let paint = LayerId(s.execute("layer.new.layer", json!({"name": "paint"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("select.rect", json!({"x": 4, "y": 6, "width": 10, "height": 10})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    (s, paint)
}

#[test]
fn duplicate_to_a_new_document_keeps_size_pixels_and_place_at_every_depth() {
    for depth in [8, 16, 32] {
        let (mut s, paint) = one_doc(depth);
        let source_before = doc(&s, 0).clone();
        let r = s.execute("layer.duplicate", json!({"document": "new"})).unwrap();
        assert_eq!(r["document"], 1);
        assert_eq!(s.active_index(), Some(1), "the new document becomes active");
        let (src, dst) = (doc(&s, 0), doc(&s, 1));
        assert_eq!(*src, source_before, "the source is untouched");
        assert_eq!(dst.size, src.size, "same canvas, as Photoshop");
        assert_eq!((dst.mode, dst.depth, dst.resolution_dpi), (src.mode, src.depth, src.resolution_dpi));
        assert_eq!(dst.name, format!("{} copy", src.name));
        let layers = r["layers"].as_array().unwrap();
        assert_eq!(layers.len(), 1);
        let id = LayerId(layers[0].as_u64().unwrap());
        assert_eq!(r["layer"], id.0);
        assert_ne!(id, paint, "a copy has its own id");
        assert_eq!(dst.layers.len(), 1, "just the duplicated layer");
        assert_eq!(dst.layers[0].name, "paint");
        assert_eq!(bounds(&s, 1, id), Rect::new(4, 6, 14, 16), "same canvas position");
        assert_eq!(
            dst.layers[0].surface().unwrap().read_region(Rect::new(4, 6, 14, 16)),
            src.layer(paint).unwrap().surface().unwrap().read_region(Rect::new(4, 6, 14, 16)),
            "depth {depth}: identical pixels"
        );
        assert_eq!(s.active().unwrap().selected_layers(), vec![id]);
    }
}

#[test]
fn duplicate_to_a_new_document_names_the_copy_and_the_document() {
    let (mut s, _) = one_doc(8);
    let r = s.execute("layer.duplicate", json!({"document": "new", "name": "Ink", "documentName": "Card"})).unwrap();
    assert_eq!(doc(&s, 1).name, "Card");
    assert_eq!(doc(&s, 1).layers[0].name, "Ink");
    assert_eq!(r["layer"].as_u64(), r["layers"][0].as_u64());
}

#[test]
fn duplicate_to_a_new_document_carries_every_selected_layer_in_order() {
    let (mut s, paint) = one_doc(8);
    let bg = doc(&s, 0).layers[0].id;
    s.execute("layer.select", json!({"layer": paint.0, "mode": "add"})).unwrap();
    s.execute("layer.select", json!({"layer": bg, "mode": "add"})).unwrap();
    let r = s.execute("layer.duplicate", json!({"document": "new"})).unwrap();
    let names: Vec<&str> = doc(&s, 1).layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Background", "paint"]);
    assert_eq!(r["layers"].as_array().unwrap().len(), 2);
    assert!(r.get("layer").is_none(), "no single layer to report");
    // A duplicated Background stays a Background: the new document has no destination one.
    let back = &doc(&s, 1).layers[0];
    assert!(back.locks.transparency && back.locks.position);
}

#[test]
fn duplicate_to_a_new_document_cuts_smart_objects_loose_as_a_batch() {
    let (mut s, _) = one_doc(8);
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    let original = s.active().unwrap().active_layer.unwrap();
    let sibling = LayerId(s.execute("layer.duplicate", json!({})).unwrap()["layer"].as_u64().unwrap());
    let contents = |s: &Session, i: usize, id: LayerId| {
        let LayerContent::Smart(sm) = &doc(s, i).layer(id).unwrap().content else { panic!("smart object") };
        sm.contents_id
    };
    assert_eq!(contents(&s, 0, original), contents(&s, 0, sibling), "instances share contents");
    s.execute("layer.select", json!({"layer": original.0, "mode": "add"})).unwrap();
    s.execute("layer.select", json!({"layer": sibling.0, "mode": "add"})).unwrap();
    let r = s.execute("layer.duplicate", json!({"document": "new"})).unwrap();
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|id| LayerId(id.as_u64().unwrap())).collect();
    assert_eq!(contents(&s, 1, ids[0]), contents(&s, 1, ids[1]), "sharing survives within the batch");
    assert_ne!(contents(&s, 1, ids[0]), contents(&s, 0, original), "independent of the source");
}

#[test]
fn duplicate_destinations_that_are_not_new_fail_cleanly() {
    let (mut s, _) = one_doc(8);
    for p in [
        json!({"document": 1}),
        json!({"document": 7}),
        json!({"document": true}),
        json!({"document": 1.5}),
        json!({"document": []}),
        json!({"document": {}}),
        json!({"document": "other"}),
        json!({"document": "new", "layer": u64::MAX}),
    ] {
        let before = (doc(&s, 0).clone(), s.documents().len(), s.active_index());
        let err = s.execute("layer.duplicate", p.clone()).unwrap_err().to_string();
        if p == json!({"document": 1}) {
            assert!(err.contains("layer.copyToDocument"), "{p}: {err}");
        }
        assert_eq!(s.active_index(), before.2, "{p}: the source stays active");
        assert_eq!(s.documents().len(), before.1, "{p}: no document created");
        assert_eq!(*doc(&s, 0), before.0, "{p}: the source is untouched");
    }
    // One open document: same-document duplicates still work, and stay enabled.
    assert!(s.is_enabled("layer.duplicate"));
    s.execute("layer.duplicate", json!({})).unwrap();
}

#[test]
fn a_named_duplicate_takes_the_name_in_one_step() {
    let (mut s, paint) = two_docs(40, 30, 8);
    let before = s.active().unwrap().history.past_len();
    let same = LayerId(s.execute("layer.duplicate", json!({"name": "Ink"})).unwrap()["layer"].as_u64().unwrap());
    assert_eq!(doc(&s, 0).layer(same).unwrap().name, "Ink");
    assert_eq!(s.active().unwrap().history.past_len(), before + 1);
    s.undo();
    s.execute("layer.select", json!({"layer": paint.0})).unwrap();
    let r = s.execute("layer.copyToDocument", json!({"document": 1, "name": "Ink"})).unwrap();
    assert_eq!(doc(&s, 1).layer(copied(&r)).unwrap().name, "Ink");
}
