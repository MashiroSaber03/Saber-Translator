//! Paint Bucket and Gradient keep alpha on a transparency-locked layer, as Edit › Fill and the
//! brushes do (#1104).

use photocraft_engine::Session;
use serde_json::json;

/// A transparent layer with a red stroke across it, transparency locked.
fn locked() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[5, 10], [20, 10]], "size": 8, "hardness": 1.0, "color": "#ff0000"})).unwrap();
    s.execute("layer.lockLayers", json!({"transparency": true})).unwrap();
    s
}

fn px(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let st = s.active().unwrap();
    st.active_layer.and_then(|id| st.doc.layer(id)).unwrap().surface().unwrap().rgba(x, y)
}

#[test]
fn bucket_keeps_alpha() {
    let mut s = locked();
    s.execute("paint.bucket", json!({"x": 30, "y": 25, "color": "#0000ff", "antiAlias": false})).unwrap();
    assert_eq!(px(&s, 30, 25)[3], 0.0, "transparent pixel became opaque");
    s.execute("paint.bucket", json!({"x": 12, "y": 10, "color": "#0000ff", "antiAlias": false})).unwrap();
    assert_eq!(px(&s, 12, 10), [0.0, 0.0, 1.0, 1.0], "the opaque stroke is still recoloured");
}

#[test]
fn gradient_keeps_alpha() {
    let mut s = locked();
    s.execute("paint.gradient", json!({"from": [0, 0], "to": [40, 30]})).unwrap();
    assert_eq!(px(&s, 30, 25)[3], 0.0, "transparent pixel became opaque");
    assert_eq!(px(&s, 12, 10)[3], 1.0);
    assert_ne!(px(&s, 12, 10), [1.0, 0.0, 0.0, 1.0], "the opaque stroke is still painted");
}
