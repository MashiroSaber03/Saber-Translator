//! Eraser on the Background (or any transparency-locked layer) paints the background colour
//! (Photoshop); on a normal layer it erases to transparency (#76).

use photocraft_engine::Session;
use serde_json::{Value, json};

fn doc(mode: &str, depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "mode": mode, "depth": depth})).unwrap();
    s.execute("tools.setColors", json!({"foreground": "#000000", "background": "#ff0000"})).unwrap();
    s
}

fn layer_px(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let st = s.active().unwrap();
    let l = st.active_layer.and_then(|id| st.doc.layer(id)).unwrap();
    l.surface().unwrap().rgba(x, y)
}

fn erase(s: &mut Session, extra: Value) -> photocraft_engine::Result<Value> {
    let mut p = json!({"points": [[2, 10], [38, 10]], "size": 8, "hardness": 1.0, "erase": true});
    if let (Some(o), Some(e)) = (p.as_object_mut(), extra.as_object()) {
        o.extend(e.clone());
    }
    s.execute("paint.stroke", p)
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.02)
}

fn undo_depth(s: &Session) -> usize {
    s.active().unwrap().history.past_len()
}

#[test]
fn eraser_on_background_paints_background_colour_at_every_depth_and_mode() {
    for mode in ["rgb", "cmyk", "gray"] {
        for depth in [8, 16, 32] {
            let mut s = doc(mode, depth);
            let before = layer_px(&s, 20, 25);
            let undo = undo_depth(&s);
            erase(&mut s, json!({})).unwrap_or_else(|e| panic!("{mode}/{depth}: {e}"));
            let got = layer_px(&s, 20, 10);
            assert_eq!(got[3], 1.0, "{mode}/{depth}: the Background stays opaque");
            // Exactly what the Brush paints with the background colour (CMYK and Gray documents
            // hold the colour converted into their mode).
            let mut brush = doc(mode, depth);
            brush.execute("paint.stroke", json!({"points": [[2, 10], [38, 10]], "size": 8, "hardness": 1.0, "color": "#ff0000"})).unwrap();
            assert!(close(got, layer_px(&brush, 20, 10)), "{mode}/{depth}: got {got:?}");
            assert!(!close(got, before), "{mode}/{depth}: the stroke shows");
            if mode == "rgb" {
                assert!(close(got, [1.0, 0.0, 0.0, 1.0]), "{mode}/{depth}: got {got:?}");
            }
            assert!(close(layer_px(&s, 20, 25), before), "{mode}/{depth}: outside the stroke untouched");
            assert_eq!(undo_depth(&s), undo + 1, "{mode}/{depth}: one undo step");
            s.execute("edit.undo", json!({})).unwrap();
            assert!(close(layer_px(&s, 20, 10), before), "{mode}/{depth}: undo restores");
        }
    }
}

#[test]
fn eraser_on_background_respects_the_selection() {
    let mut s = doc("rgb", 8);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 30})).unwrap();
    erase(&mut s, json!({})).unwrap();
    assert!(close(layer_px(&s, 10, 10), [1.0, 0.0, 0.0, 1.0]));
    assert!(close(layer_px(&s, 30, 10), [1.0, 1.0, 1.0, 1.0]), "outside the selection stays white");
}

#[test]
fn eraser_on_normal_layer_erases_to_transparency() {
    for depth in [8, 16, 32] {
        let mut s = doc("rgb", depth);
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("paint.stroke", json!({"points": [[2, 10], [38, 10]], "size": 12, "hardness": 1.0, "color": "#00ff00"})).unwrap();
        assert_eq!(layer_px(&s, 20, 10)[3], 1.0);
        erase(&mut s, json!({})).unwrap();
        assert_eq!(layer_px(&s, 20, 10)[3], 0.0, "{depth}: erased to transparency");
    }
}

#[test]
fn eraser_on_transparency_locked_layer_paints_background_colour() {
    let mut s = doc("rgb", 16);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[2, 10], [38, 10]], "size": 12, "hardness": 1.0, "color": "#00ff00"})).unwrap();
    s.execute("layer.lockLayers", json!({"transparency": true})).unwrap();
    erase(&mut s, json!({})).unwrap();
    assert!(close(layer_px(&s, 20, 10), [1.0, 0.0, 0.0, 1.0]), "{:?}", layer_px(&s, 20, 10));
    assert_eq!(layer_px(&s, 20, 25)[3], 0.0, "transparent pixels stay transparent");
}

#[test]
fn stroke_path_with_the_eraser_paints_background_colour_on_locked_layers() {
    let path = json!({"path": {"subpaths": [{"knots": [[5, 10], [35, 10]]}]}});
    // The Background: background colour, still opaque.
    let mut s = doc("rgb", 8);
    s.execute("path.set", path.clone()).unwrap();
    s.execute("path.stroke", json!({"tool": "eraser", "size": 6, "hardness": 1.0, "opacity": 100})).unwrap();
    assert!(close(layer_px(&s, 20, 10), [1.0, 0.0, 0.0, 1.0]), "{:?}", layer_px(&s, 20, 10));
    // A transparency-locked layer: the same, and transparent pixels stay transparent.
    let mut s = doc("rgb", 16);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[2, 10], [38, 10]], "size": 12, "hardness": 1.0, "color": "#00ff00"})).unwrap();
    s.execute("layer.lockLayers", json!({"transparency": true})).unwrap();
    s.execute("path.set", path.clone()).unwrap();
    s.execute("path.stroke", json!({"tool": "eraser", "size": 6, "hardness": 1.0, "opacity": 100})).unwrap();
    assert!(close(layer_px(&s, 20, 10), [1.0, 0.0, 0.0, 1.0]), "{:?}", layer_px(&s, 20, 10));
    assert_eq!(layer_px(&s, 20, 25)[3], 0.0);
    // Unlocked: erased to transparency.
    s.execute("layer.lockLayers", json!({"transparency": false})).unwrap();
    s.execute("paint.stroke", json!({"points": [[2, 10], [38, 10]], "size": 12, "hardness": 1.0, "color": "#00ff00"})).unwrap();
    s.execute("path.stroke", json!({"tool": "eraser", "size": 6, "hardness": 1.0, "opacity": 100})).unwrap();
    assert_eq!(layer_px(&s, 20, 10)[3], 0.0);
}

#[test]
fn eraser_bad_params_fail_gracefully() {
    let mut s = doc("rgb", 8);
    assert!(s.execute("paint.stroke", json!({"erase": true})).is_err());
    assert!(s.execute("paint.stroke", json!({"erase": true, "points": []})).is_err());
    assert!(s.execute("paint.stroke", json!({"erase": true, "points": "x"})).is_err());
    assert!(erase(&mut s, json!({"layer": u64::MAX})).is_err());
    assert!(erase(&mut s, json!({"target": "mask"})).is_err(), "no mask");
    // Absurd coordinates are refused up front (a stroke that long would run billions of dabs).
    assert!(erase(&mut s, json!({"points": [[0, 0], [f64::MAX, 0.0]]})).is_err());
    assert!(erase(&mut s, json!({"points": [[1e30, -1e30]]})).is_err());
    assert!(s.execute("paint.pencil", json!({"erase": true, "points": [[0, 0], [1e12, 0]]})).is_err());
    // Degenerate brushes must not panic.
    let _ = erase(&mut s, json!({"size": -5}));
    assert!(erase(&mut s, json!({"brush": {"size": "big"}})).is_err());
}
