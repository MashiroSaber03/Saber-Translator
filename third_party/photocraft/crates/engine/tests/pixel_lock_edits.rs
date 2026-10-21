//! Clear, Cut and Image › Adjustments refuse a pixel-locked layer and leave it unchanged, as
//! painting does (#1103).

use photocraft_engine::Session;
use serde_json::json;

/// A layer with a red stroke across it, pixels locked.
fn locked() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[5, 10], [20, 10]], "size": 8, "hardness": 1.0, "color": "#ff0000"})).unwrap();
    s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
    s
}

fn px(s: &Session) -> [f32; 4] {
    let st = s.active().unwrap();
    st.active_layer.and_then(|id| st.doc.layer(id)).unwrap().surface().unwrap().rgba(12, 10)
}

#[test]
fn locked_layer_edits_are_refused() {
    for (cmd, p) in [
        ("edit.clear", json!({})),
        ("edit.cut", json!({})),
        ("image.adjustments.invert", json!({})),
        ("image.autoTone", json!({})),
        ("image.autoContrast", json!({})),
        ("image.autoColor", json!({})),
    ] {
        let mut s = locked();
        let before = px(&s);
        let r = s.execute(cmd, p);
        assert!(r.is_err(), "{cmd} edited a locked layer: {r:?}");
        assert_eq!(px(&s), before, "{cmd} changed a locked layer's pixels");
    }
}

#[test]
fn a_refused_cut_leaves_the_clipboard_alone() {
    let mut s = locked();
    // Copy the whole canvas first, then select a smaller area: a Cut that went through would
    // replace the clipboard with a smaller clip.
    s.execute("edit.copy", json!({})).unwrap();
    let before = s.clipboard.as_ref().map(|c| c.bounds);
    assert!(before.is_some());
    s.execute("select.rect", json!({"x": 2, "y": 2, "width": 10, "height": 10})).unwrap();
    assert!(s.execute("edit.cut", json!({})).is_err());
    assert_eq!(s.clipboard.as_ref().map(|c| c.bounds), before, "a refused Cut replaced the clipboard");
}

#[test]
fn unlocked_layer_edits_still_work() {
    let mut s = locked();
    s.execute("layer.lockLayers", json!({"pixels": false})).unwrap();
    s.execute("image.adjustments.invert", json!({})).unwrap();
    assert_eq!(px(&s), [0.0, 1.0, 1.0, 1.0]);
    s.execute("edit.clear", json!({})).unwrap();
    assert_eq!(px(&s)[3], 0.0);
}
