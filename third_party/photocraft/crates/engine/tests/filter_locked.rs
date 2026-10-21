//! Filters refuse a pixel-locked (or fully locked) layer and leave it unchanged, as painting
//! does (#1102).

use photocraft_doc::LayerId;
use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::json;

fn layered() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[5, 10], [20, 10]], "size": 8, "hardness": 1.0, "color": "#ff0000"})).unwrap();
    s
}

fn px(s: &Session) -> [f32; 4] {
    let st = s.active().unwrap();
    st.active_layer.and_then(|id| st.doc.layer(id)).unwrap().surface().unwrap().rgba(12, 6)
}

#[test]
fn filters_refuse_a_locked_layer() {
    for lock in [json!({"pixels": true}), json!({"all": true})] {
        let mut s = layered();
        s.execute("layer.lockLayers", lock.clone()).unwrap();
        let before = px(&s);
        let r = s.execute("filter.blur.gaussianBlur", json!({"radius": 4}));
        assert!(r.is_err(), "blurred a layer locked with {lock}: {r:?}");
        assert_eq!(px(&s), before);
    }
}

#[test]
fn filters_still_run_on_an_unlocked_layer() {
    let mut s = layered();
    let before = px(&s);
    s.execute("filter.blur.gaussianBlur", json!({"radius": 4})).unwrap();
    assert_ne!(px(&s), before);
}

#[test]
fn render_filters_and_fill_path_refuse_a_locked_layer() {
    for (cmd, p) in [("filter.render.tree", json!({})), ("filter.render.pictureFrame", json!({})), ("path.fill", json!({"color": "#0000ff"}))] {
        let mut s = layered();
        s.execute("path.set", json!({"path": {"subpaths": [{"knots": [[2, 2], [38, 2], [38, 28], [2, 28]]}]}})).unwrap();
        s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
        let before = px(&s);
        let r = s.execute(cmd, p);
        assert!(r.is_err(), "{cmd} drew on a locked layer: {r:?}");
        assert_eq!(px(&s), before);
    }
}

#[test]
fn stroke_path_refuses_a_locked_layer_or_group() {
    for tool in ["brush", "pencil", "eraser"] {
        for lock in [json!({"pixels": true}), json!({"all": true})] {
            for on_group in [false, true] {
                let mut s = layered();
                s.execute("path.set", json!({"path": {"subpaths": [{"knots": [[2, 6], [38, 6]]}]}})).unwrap();
                let layer = s.active().unwrap().active_layer.unwrap().0;
                if on_group {
                    s.execute("layer.new.groupFromLayers", json!({})).unwrap();
                }
                s.execute("layer.lockLayers", lock.clone()).unwrap();
                let row = |s: &Session| s.active().unwrap().doc.layer(LayerId(layer)).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 40, 30));
                let before = row(&s);
                let r = s.execute("path.stroke", json!({"layer": layer, "tool": tool, "size": 6, "color": "#0000ff"}));
                assert!(r.is_err(), "{tool} stroked a layer locked with {lock} (group: {on_group}): {r:?}");
                assert_eq!(row(&s), before, "{tool} {lock} group: {on_group}");
            }
        }
    }
}

#[test]
fn stroke_path_still_runs_on_an_unlocked_layer() {
    let mut s = layered();
    s.execute("path.set", json!({"path": {"subpaths": [{"knots": [[2, 6], [38, 6]]}]}})).unwrap();
    let before = px(&s);
    s.execute("path.stroke", json!({"tool": "pencil", "size": 6, "color": "#0000ff"})).unwrap();
    assert_ne!(px(&s), before);
}

#[test]
fn filters_keep_alpha_on_a_transparency_locked_layer() {
    let mut s = layered();
    s.execute("layer.lockLayers", json!({"transparency": true})).unwrap();
    let alpha = |s: &Session, x, y| {
        let st = s.active().unwrap();
        st.active_layer.and_then(|id| st.doc.layer(id)).unwrap().surface().unwrap().rgba(x, y)[3]
    };
    let (edge, outside) = (alpha(&s, 12, 6), alpha(&s, 12, 2));
    s.execute("filter.blur.gaussianBlur", json!({"radius": 4})).unwrap();
    assert_eq!(alpha(&s, 12, 6), edge, "the blur changed alpha at the stroke edge");
    assert_eq!(alpha(&s, 12, 2), outside, "the blur spread into the transparent area");
}
