use super::*;
use crate::brush_cmds::LiveStroke;
use photocraft_doc::{Layer, LayerMask};

fn session(depth: u8, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width":81,"height":61,"depth":depth,"mode":mode,"background":"white"})).unwrap();
    s
}
#[test]
fn presets_validate_atomically_and_leave_document_paths_and_history_untouched() {
    let mut s = session(8, "rgb");
    s.execute("path.set", json!({"name":"work","path":{"subpaths":[{"knots":[[2,3],[9,15]]}]}})).unwrap();
    let before = s.active().unwrap().doc.clone();
    let history = s.active().unwrap().history.past_len();
    let revision = s.active().unwrap().revision;
    for mode in SymmetryMode::ALL {
        s.execute("paint.setSymmetry", json!({"mode":mode.id()})).unwrap();
        let p = s.active().unwrap().symmetry.as_ref().unwrap().preset().unwrap();
        assert_eq!(p.center, [40.5, 30.5]);
        assert_eq!(p.rotation, 0.0);
        assert_eq!(crate::inspect::document(s.active().unwrap())["symmetry"]["mode"], mode.id());
        assert_eq!(s.active().unwrap().doc, before);
        assert_eq!((s.active().unwrap().history.past_len(), s.active().unwrap().revision), (history, revision));
    }
    s.execute("paint.setSymmetry", json!({"mode":"dual","center":[17,29],"rotation":45})).unwrap();
    let original = s.active().unwrap().symmetry.clone();
    for bad in [
        Value::Null,
        json!([]),
        json!({}),
        json!({"mode":"radial"}),
        json!({"mode":7}),
        json!({"mode":"dual","center":[1]}),
        json!({"mode":"dual","center":[1,2,3]}),
        json!({"mode":"dual","center":null}),
        json!({"mode":"dual","center":["x",2]}),
        json!({"mode":"dual","center":[1e8,0]}),
        json!({"mode":"dual","rotation":"x"}),
        json!({"mode":"dual","rotation":null}),
    ] {
        assert!(s.execute("paint.setSymmetry", bad).is_err());
        assert_eq!(s.active().unwrap().symmetry, original);
        assert_eq!(s.active().unwrap().doc, before);
    }
    s.execute("paint.setSymmetry", json!({"mode":"dual"})).unwrap();
    assert_eq!(s.active().unwrap().symmetry.as_ref().unwrap().preset().unwrap().center, [40.5, 30.5]);
    s.execute("paint.symmetryFromPath", json!({"name":"work"})).unwrap();
    assert_eq!(s.active().unwrap().symmetry.as_ref().unwrap().path_source(), Some("work"));
    s.execute("paint.symmetryDisable", json!({})).unwrap();
    assert!(crate::inspect::document(s.active().unwrap())["symmetry"].is_null());
    s.execute("paint.setSymmetry", json!({"mode":"vertical"})).unwrap();
    s.execute("file.new", json!({"width":10,"height":20})).unwrap();
    assert!(s.active().unwrap().symmetry.is_none());
    assert!(s.set_active(0));
    assert!(s.active().unwrap().symmetry.is_some());
    let mut empty = Session::new();
    assert!(!empty.is_enabled("paint.setSymmetry"));
    assert!(empty.execute("paint.setSymmetry", json!({"mode":"vertical"})).is_err());
}
fn pixel(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
}
#[test]
fn every_preset_live_preview_matches_committed_pixels_and_undo_at_every_depth() {
    for depth in [8, 16, 32] {
        for mode in ["rgb", "gray"] {
            for preset in SymmetryMode::ALL {
                for command in ["paint.stroke", "paint.pencil"] {
                    let mut s = session(depth, mode);
                    s.execute("paint.setSymmetry", json!({"mode":preset.id(),"center":[40,30]})).unwrap();
                    let before = s.active().unwrap().doc.clone();
                    let params = json!({"points":[[20,10],[24,14]],"size":5,"hardness":1.0,"color":"#000000","opacity":0.5,"smoothing":0.0,"seed":7});
                    let live = LiveStroke::begin_with(&s, command, &params).unwrap();
                    let past = s.active().unwrap().history.past_len();
                    s.execute(command, params).unwrap();
                    let st = s.active().unwrap();
                    assert_eq!(st.history.past_len(), past + 1);
                    let id = st.active_layer.unwrap();
                    let actual = st.doc.layer(id).unwrap().surface().unwrap();
                    let preview = live.doc.layer(id).unwrap().surface().unwrap();
                    let original = StrokePoint::new(20.0, 10.0, 1.0);
                    let p = st.symmetry.as_ref().unwrap().preset().unwrap();
                    let positions = std::iter::once(original).chain(p.reflected_passes(&[original]).into_iter().map(|p| p[0]));
                    for p in positions {
                        assert!(actual.rgba(p.x.round() as i32, p.y.round() as i32)[0] < 0.9, "{depth} {mode} {preset:?}");
                    }
                    for y in 0..61 {
                        for x in 0..81 {
                            let a = actual.rgba(x, y);
                            let b = preview.rgba(x, y);
                            assert!((0..4).all(|i| (a[i] - b[i]).abs() < 0.02));
                        }
                    }
                    let after = st.doc.clone();
                    assert!(s.undo());
                    assert_eq!(s.active().unwrap().doc, before);
                    assert!(s.redo());
                    assert_eq!(s.active().unwrap().doc, after);
                }
            }
        }
    }
}
#[test]
fn dual_centre_overlap_has_one_opacity_ceiling_and_eraser_uses_the_same_copies() {
    for depth in [8, 16, 32] {
        let mut plain = session(depth, "rgb");
        let mut symmetric = session(depth, "rgb");
        symmetric.execute("paint.setSymmetry", json!({"mode":"dual","center":[40,30]})).unwrap();
        let params = json!({"points":[[40,30]],"size":9,"hardness":1.0,"opacity":0.5,"color":"#000000","smoothing":0.0});
        plain.execute("paint.stroke", params.clone()).unwrap();
        symmetric.execute("paint.stroke", params).unwrap();
        let a = pixel(&plain, 40, 30);
        let b = pixel(&symmetric, 40, 30);
        assert!((0..4).all(|i| (a[i] - b[i]).abs() < 0.02));
        symmetric.execute("layer.new.layer", json!({})).unwrap();
        symmetric.execute("edit.fill", json!({"color":"#000000"})).unwrap();
        let params = json!({"points":[[20,10],[25,15]],"size":7,"hardness":1.0,"erase":true,"smoothing":0.0});
        let mut live = LiveStroke::begin(&symmetric, &json!({"points":[[20,10]],"size":7,"hardness":1.0,"erase":true,"smoothing":0.0})).unwrap();
        live.push(&[StrokePoint::new(25.0, 15.0, 1.0)]).unwrap();
        symmetric.execute("paint.stroke", params).unwrap();
        let st = symmetric.active().unwrap();
        let id = st.active_layer.unwrap();
        for (x, y) in [(20, 10), (60, 10), (20, 50), (60, 50)] {
            let a = pixel(&symmetric, x, y);
            let b = live.doc.layer(id).unwrap().surface().unwrap().rgba(x, y);
            assert!(a[3] < 0.2);
            assert!((0..4).all(|i| (a[i] - b[i]).abs() < 0.02));
        }
    }
}
#[test]
fn selection_transparency_lock_and_mask_targets_apply_to_all_passes() {
    let mut s = session(8, "rgb");
    s.execute("paint.setSymmetry", json!({"mode":"dual","center":[40,30]})).unwrap();
    s.execute("select.rect", json!({"x":0,"y":0,"width":40,"height":30})).unwrap();
    s.execute("paint.stroke", json!({"points":[[20,10]],"size":5,"hardness":1.0,"color":"#000000"})).unwrap();
    assert!(pixel(&s, 20, 10)[0] < 0.2);
    assert!(pixel(&s, 60, 10)[0] > 0.9);
    s.execute("select.deselect", json!({})).unwrap();
    s.edit("Mask fixture", |doc, active| {
        let mut layer = Layer::raster("Mask", doc.pixel_format());
        layer.mask = Some(LayerMask::reveal_all());
        layer.locks.transparency = true;
        *active = Some(layer.id);
        doc.layers.push(layer);
        Ok(())
    })
    .unwrap();
    s.execute("paint.stroke", json!({"points":[[20,10]],"size":5,"color":"#000000","target":"mask"})).unwrap();
    let st = s.active().unwrap();
    let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
    for (x, y) in [(20, 10), (60, 10), (20, 50), (60, 50)] {
        assert!(l.mask.as_ref().unwrap().surface.rgba(x, y)[0] < 0.8);
    }
    assert!(l.surface().unwrap().content_bounds().is_empty());
    s.execute("paint.stroke", json!({"points":[[20,10]],"size":5,"color":"#ff0000"})).unwrap();
    assert!(s.active().unwrap().doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap().surface().unwrap().content_bounds().is_empty());
}

#[test]
fn streaming_across_dual_axes_keeps_one_opacity_and_matches_commit() {
    let points = [[40.0, 30.0], [40.0, 20.0], [35.0, 20.0], [45.0, 40.0]];
    for depth in [8, 16, 32] {
        let mut s = session(depth, "rgb");
        s.execute("paint.setSymmetry", json!({"mode":"dual","center":[40,30]})).unwrap();
        let params = json!({"points":[points[0]],"size":9,"hardness":1.0,"opacity":0.5,"color":"#000000","smoothing":0.0});
        let mut live = LiveStroke::begin(&s, &params).unwrap();
        for [x, y] in points.into_iter().skip(1) {
            live.push(&[StrokePoint::new(x, y, 1.0)]).unwrap();
        }
        let mut params = params;
        params["points"] = json!(points);
        s.execute("paint.stroke", params).unwrap();
        let st = s.active().unwrap();
        let id = st.active_layer.unwrap();
        for y in 0..61 {
            for x in 0..81 {
                let a = pixel(&s, x, y);
                let b = live.doc.layer(id).unwrap().surface().unwrap().rgba(x, y);
                assert!(a[0] >= 0.49, "opacity applied more than once at {x},{y}");
                assert!((0..4).all(|i| (a[i] - b[i]).abs() < 0.02), "preview differs at {x},{y}");
            }
        }
    }
}
