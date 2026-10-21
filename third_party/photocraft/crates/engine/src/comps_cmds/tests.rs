use super::*;
use photocraft_color::BlendMode;
use photocraft_doc::Layer;
use photocraft_geom::Rect;

fn session(depth: u32) -> (Session, LayerId, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48, "depth": depth})).unwrap();
    let (a, b) = s
        .edit("setup", |doc, _| {
            let fmt = doc.pixel_format();
            let mut a = Layer::raster("A", fmt);
            a.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 14, 14), &photocraft_raster::from_rgba(&fmt, [1.0, 0.0, 0.0, 1.0]));
            let mut b = Layer::raster("B", fmt);
            b.surface_mut().unwrap().fill_rect(Rect::new(30, 20, 40, 30), &photocraft_raster::from_rgba(&fmt, [0.0, 0.0, 1.0, 1.0]));
            let (ia, ib) = (a.id, b.id);
            doc.layers.push(a);
            doc.layers.push(b);
            Ok((ia, ib))
        })
        .unwrap();
    (s, a, b)
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn pos(s: &Session, id: LayerId) -> Option<(i32, i32)> {
    layer_position(doc(s).layer(id).unwrap())
}

#[test]
fn capture_apply_visibility_position_appearance_independently() {
    for depth in [8, 16, 32] {
        let (mut s, a, b) = session(depth);
        let c1 = s.execute("layerComp.new", json!({"name": "Start"})).unwrap()["comp"].as_u64().unwrap();
        // Change all three properties.
        s.execute("layer.translate", json!({"layer": a.0, "dx": 10, "dy": 5})).unwrap();
        s.edit("t", |d, _| {
            let l = d.layer_mut(b).unwrap();
            l.visible = false;
            l.blend = BlendMode::Multiply;
            l.opacity = 0.5;
            Ok(())
        })
        .unwrap();
        assert_eq!(pos(&s, a), Some((14, 9)));

        // Visibility only.
        s.execute("layerComp.setOptions", json!({"comp": c1, "position": false, "appearance": false})).unwrap();
        s.execute("layerComp.apply", json!({"comp": c1})).unwrap();
        assert!(doc(&s).layer(b).unwrap().visible, "{depth}");
        assert_eq!(pos(&s, a), Some((14, 9)));
        assert_eq!(doc(&s).layer(b).unwrap().blend, BlendMode::Multiply);

        // Position only.
        s.edit("t", |d, _| {
            d.layer_mut(b).unwrap().visible = false;
            Ok(())
        })
        .unwrap();
        s.execute("layerComp.setOptions", json!({"comp": c1, "visibility": false, "position": true})).unwrap();
        s.execute("layerComp.apply", json!({})).unwrap();
        assert_eq!(pos(&s, a), Some((4, 4)));
        assert!(!doc(&s).layer(b).unwrap().visible);

        // Appearance only.
        s.execute("layerComp.setOptions", json!({"comp": "Start", "position": false, "appearance": true})).unwrap();
        s.execute("layerComp.apply", json!({})).unwrap();
        let lb = doc(&s).layer(b).unwrap();
        assert_eq!((lb.blend, lb.opacity, lb.visible), (BlendMode::Normal, 1.0, false));
        // Pixels really moved (not just bounds bookkeeping).
        let surf = doc(&s).layer(a).unwrap().surface().unwrap();
        assert!(surf.pixel(5, 5)[3] > 0.9 && surf.pixel(16, 11)[3] < 0.1);
    }
}

#[test]
fn undo_redo_and_last_document_state() {
    let (mut s, a, _) = session(8);
    s.execute("layerComp.new", json!({})).unwrap();
    assert_eq!(doc(&s).layer_comps[0].name, "Layer Comp 1");
    s.execute("layer.translate", json!({"layer": a.0, "dx": 3, "dy": 0})).unwrap();
    // A fresh document state (not a comp) is remembered when a comp is applied over it.
    s.execute("layerComp.apply", json!({})).unwrap();
    assert_eq!(pos(&s, a), Some((4, 4)));
    assert!(doc(&s).last_document_state.is_some());
    s.execute("layerComp.restoreLastDocumentState", json!({})).unwrap();
    assert_eq!(pos(&s, a), Some((7, 4)));
    assert_eq!(doc(&s).last_applied_comp, None);
    // Each step is one undo.
    assert!(s.undo());
    assert_eq!(pos(&s, a), Some((4, 4)));
    assert!(s.undo());
    assert_eq!(pos(&s, a), Some((7, 4)));
    assert!(s.redo());
    assert_eq!(pos(&s, a), Some((4, 4)));
    s.execute("layerComp.delete", json!({})).unwrap();
    assert!(doc(&s).layer_comps.is_empty());
    assert!(s.undo());
    assert_eq!(doc(&s).layer_comps.len(), 1);
}

#[test]
fn ordinary_layout_edits_become_the_last_document_state() {
    for depth in [8, 16, 32] {
        for (command, mut params) in [
            ("layer.translate", json!({"dx": 9, "dy": 3})),
            ("layer.setProps", json!({"visible": false})),
            ("layer.setProps", json!({"opacity": 0.25})),
            ("layer.setProps", json!({"blend": "multiply"})),
        ] {
            let (mut s, a, _) = session(depth);
            s.execute("layerComp.new", json!({})).unwrap();
            params["layer"] = json!(a.0);
            s.execute(command, params).unwrap();
            let edited = capture_states(doc(&s));
            s.execute("layerComp.apply", json!({})).unwrap();
            assert_ne!(capture_states(doc(&s)), edited);
            assert!(s.is_enabled("layerComp.restoreLastDocumentState"));
            s.execute("layerComp.restoreLastDocumentState", json!({})).unwrap();
            assert_eq!(capture_states(doc(&s)), edited, "{depth}-bit");
        }
    }
}

#[test]
fn cycling_comps_keeps_the_latest_edited_layout() {
    let (mut s, a, _) = session(8);
    let one = s.execute("layerComp.new", json!({"name": "One"})).unwrap()["comp"].clone();
    s.execute("layer.translate", json!({"layer": a.0, "dx": 10, "dy": 0})).unwrap();
    let two = s.execute("layerComp.new", json!({"name": "Two"})).unwrap()["comp"].clone();
    for dx in [7, 13] {
        s.execute("layer.translate", json!({"layer": a.0, "dx": dx, "dy": 0})).unwrap();
        let edited = capture_states(doc(&s));
        s.execute("layerComp.apply", json!({"comp": one})).unwrap();
        s.execute("layerComp.apply", json!({"comp": two})).unwrap();
        s.execute("layerComp.apply", json!({"comp": two})).unwrap();
        s.execute("layerComp.restoreLastDocumentState", json!({})).unwrap();
        assert_eq!(capture_states(doc(&s)), edited);
    }
}

#[test]
fn unrecorded_properties_do_not_replace_the_backup_on_reapply() {
    let (mut s, a, _) = session(8);
    s.execute("layerComp.new", json!({"position": false, "appearance": false})).unwrap();
    s.execute("layer.translate", json!({"layer": a.0, "dx": 10, "dy": 0})).unwrap();
    s.execute("layer.setProps", json!({"layer": a.0, "visible": false, "opacity": 0.5})).unwrap();
    let edited = capture_states(doc(&s));
    s.execute("layerComp.apply", json!({})).unwrap();
    assert!(doc(&s).layer(a).unwrap().visible);
    s.execute("layerComp.apply", json!({})).unwrap();
    s.execute("layerComp.restoreLastDocumentState", json!({})).unwrap();
    assert_eq!(capture_states(doc(&s)), edited);
}

#[test]
fn manage_comps_and_cycle() {
    let (mut s, _, b) = session(8);
    let c1 = s.execute("layerComp.new", json!({"name": "One", "comment": "first"})).unwrap()["comp"].as_u64().unwrap();
    s.edit("hide", |d, _| {
        d.layer_mut(b).unwrap().visible = false;
        Ok(())
    })
    .unwrap();
    let c2 = s.execute("layerComp.new", json!({"name": "Two"})).unwrap()["comp"].as_u64().unwrap();
    let c3 = s.execute("layerComp.duplicate", json!({"comp": c1})).unwrap()["comp"].as_u64().unwrap();
    let names: Vec<_> = doc(&s).layer_comps.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, ["One", "One copy", "Two"]);
    s.execute("layerComp.rename", json!({"comp": c3, "name": "Three"})).unwrap();
    s.execute("layerComp.setComment", json!({"comp": "Three", "comment": "dup"})).unwrap();
    assert_eq!(doc(&s).comp(c3 as u32).unwrap().comment, "dup");
    // Next/previous cycle through the list order and wrap.
    s.execute("layerComp.apply", json!({"comp": c2})).unwrap();
    s.execute("layerComp.next", json!({})).unwrap();
    assert_eq!(doc(&s).last_applied_comp, Some(c1 as u32));
    assert!(doc(&s).layer(b).unwrap().visible);
    s.execute("layerComp.previous", json!({})).unwrap();
    assert_eq!(doc(&s).last_applied_comp, Some(c2 as u32));
    assert!(!doc(&s).layer(b).unwrap().visible);
    let l = s.execute("layerComp.list", json!({})).unwrap();
    assert_eq!(l["comps"].as_array().unwrap().len(), 3);
    // Update one property: the visibility recorded in "One" follows the document again.
    s.execute("layerComp.update", json!({"comp": c1, "what": "visibility"})).unwrap();
    assert_eq!(doc(&s).comp(c1 as u32).unwrap().state(b).unwrap().visible, Some(false));
    s.execute("layerComp.update", json!({"comp": "*"})).unwrap();
}

#[test]
fn warnings_for_deleted_layers() {
    let (mut s, a, _) = session(8);
    s.execute("layerComp.new", json!({})).unwrap();
    assert_eq!(s.execute("layerComp.updateWarnings", json!({})).unwrap()["count"], 0);
    s.edit("del", |d, _| {
        d.remove(a);
        Ok(())
    })
    .unwrap();
    let w = s.execute("layerComp.updateWarnings", json!({})).unwrap();
    assert_eq!(w["count"], 1);
    assert_eq!(w["warnings"][0]["missingLayers"][0], a.0);
    // Applying still works (missing layers are skipped) and reports them.
    assert_eq!(s.execute("layerComp.apply", json!({})).unwrap()["missingLayers"], 1);
    s.execute("layerComp.updateWarnings", json!({"clear": true})).unwrap();
    assert_eq!(s.execute("layerComp.updateWarnings", json!({})).unwrap()["count"], 0);
}

#[test]
fn disabled_states_and_bad_params() {
    let mut s = Session::new();
    assert!(!s.is_enabled("layerComp.new"));
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    assert!(s.is_enabled("layerComp.new"));
    for id in
        ["layerComp.apply", "layerComp.update", "layerComp.next", "layerComp.delete", "file.export.layerCompsToFiles", "layerComp.restoreLastDocumentState"]
    {
        assert!(!s.is_enabled(id), "{id}");
    }
    s.execute("layerComp.new", json!({})).unwrap();
    assert!(s.execute("layerComp.apply", json!({"comp": 999})).is_err());
    assert!(s.execute("layerComp.apply", json!({"comp": "nope"})).is_err());
    assert!(s.execute("layerComp.update", json!({"what": "colour"})).is_err());
    assert!(s.execute("layerComp.rename", json!({})).is_err());
    assert!(s.execute("layerComp.setOptions", json!({})).is_err());
}

#[test]
fn export_comps_to_files_naming() {
    let (mut s, _, b) = session(8);
    s.execute("layerComp.new", json!({"name": "With B"})).unwrap();
    s.edit("hide", |d, _| {
        d.layer_mut(b).unwrap().visible = false;
        Ok(())
    })
    .unwrap();
    s.execute("layerComp.new", json!({"name": "No/B"})).unwrap();
    let dir = std::env::temp_dir().join(format!("photocraft-comps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let d = dir.to_string_lossy().into_owned();
    let r = s.execute("file.export.layerCompsToFiles", json!({"dir": d, "prefix": "doc"})).unwrap();
    let files: Vec<String> = r["files"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(files.len(), 2);
    assert!(files[0].ends_with("doc_0000_With B.png"), "{files:?}");
    assert!(files[1].ends_with("doc_0001_No_B.png"), "{files:?}");
    // Each file shows its comp: B's blue square only in the first.
    let blue = |path: &str| {
        let doc = photocraft_io::import("x.png", &std::fs::read(path).unwrap()).unwrap().document;
        photocraft_compose::render(&doc, Rect::new(35, 25, 36, 26)).px[0]
    };
    assert!(blue(&files[0])[2] > 0.9 && blue(&files[0])[0] < 0.1);
    assert!(blue(&files[1])[2] > 0.9 && blue(&files[1])[0] > 0.9);
    let r = s.execute("file.export.layerCompsToFiles", json!({"dir": d, "selectedOnly": true, "format": "jpg"})).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn out_of_range_comp_ids_are_rejected_not_wrapped() {
    // #914: 2^32 + id used to wrap to a real comp.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    let c = s.execute("layerComp.new", json!({"name": "Start"})).unwrap()["comp"].as_u64().unwrap();
    let wrapped = (1u64 << 32) + c;
    assert!(s.execute("layerComp.delete", json!({"comp": wrapped})).is_err());
    assert!(s.execute("layerComp.setOptions", json!({"comp": wrapped, "name": "renamed"})).is_err());
    let doc = &s.active().unwrap().doc;
    assert_eq!(doc.layer_comps.len(), 1);
    assert_eq!(doc.layer_comps[0].name, "Start");
    let dir = std::env::temp_dir().join(format!("photocraft-comps-wrap-{}", std::process::id()));
    let r = s.execute("file.export.layerCompsToFiles", json!({"dir": dir.to_string_lossy(), "comps": [wrapped]}));
    assert!(r.is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn unreachable_comp_positions_leave_the_layer_in_place() {
    // #1017: a recorded position whose distance from the layer's current one leaves i32 range
    // (e.g. from a file) panicked on apply. The move is skipped; visibility still applies.
    for target in [(i32::MIN, 4), (4, i32::MIN)] {
        let (mut s, a, b) = session(8);
        let comp = LayerComp {
            id: 1,
            name: "Far".into(),
            comment: String::new(),
            apply_visibility: true,
            apply_position: true,
            apply_appearance: false,
            states: vec![
                CompLayerState { layer: a, visible: Some(false), position: Some(target), appearance: None },
                // Control: an ordinary position on another layer still moves it.
                CompLayerState { layer: b, visible: None, position: Some((31, 22)), appearance: None },
            ],
        };
        s.edit("far", |d, _| {
            apply_comp(d, &comp, false);
            Ok(())
        })
        .unwrap();
        assert_eq!(pos(&s, a), Some((4, 4)), "{target:?}");
        assert!(!doc(&s).layer(a).unwrap().visible);
        assert_eq!(pos(&s, b), Some((31, 22)));
    }
}

#[test]
fn clearing_one_comps_warning_leaves_the_others() {
    // Regression for #4197: the per-comp warning icon cleared every comp's warnings.
    let (mut s, a, _) = session(8);
    let c1 = s.execute("layerComp.new", json!({"name": "One"})).unwrap()["comp"].as_u64().unwrap();
    let c2 = s.execute("layerComp.new", json!({"name": "Two"})).unwrap()["comp"].as_u64().unwrap();
    s.edit("del", |d, _| {
        d.remove(a);
        Ok(())
    })
    .unwrap();
    // Reporting can be limited to one comp too.
    let w = s.execute("layerComp.updateWarnings", json!({"comp": c2})).unwrap();
    assert_eq!(w["count"], 1);
    assert_eq!(w["warnings"][0]["comp"], c2);
    s.execute("layerComp.updateWarnings", json!({"clear": true, "comp": c1})).unwrap();
    assert!(doc(&s).comp(c1 as u32).unwrap().missing_layers(doc(&s)).is_empty());
    assert_eq!(doc(&s).comp(c2 as u32).unwrap().missing_layers(doc(&s)), vec![a]);
    let w = s.execute("layerComp.updateWarnings", json!({})).unwrap();
    assert_eq!(w["count"], 1);
    assert_eq!(w["warnings"][0]["comp"], c2);
    // Without "comp" every comp is cleared (Clear All Layer Comp Warnings).
    s.execute("layerComp.updateWarnings", json!({"clear": true})).unwrap();
    assert_eq!(s.execute("layerComp.updateWarnings", json!({})).unwrap()["count"], 0);
}

#[test]
fn clearing_warnings_of_an_unknown_comp_is_an_error() {
    let (mut s, a, _) = session(8);
    let c1 = s.execute("layerComp.new", json!({})).unwrap()["comp"].as_u64().unwrap();
    s.edit("del", |d, _| {
        d.remove(a);
        Ok(())
    })
    .unwrap();
    let before = doc(&s).layer_comps.clone();
    for bad in [json!(c1 + 99), json!("nope"), json!(-1), json!(1.5), json!(true), json!([c1])] {
        assert!(s.execute("layerComp.updateWarnings", json!({"clear": true, "comp": bad})).is_err(), "{bad}");
    }
    assert_eq!(doc(&s).layer_comps, before);
    assert_eq!(s.execute("layerComp.updateWarnings", json!({})).unwrap()["count"], 1);
}
