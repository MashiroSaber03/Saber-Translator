//! Background-layer invariants through merge, duplication and canvas extension.
//! These three commands must agree with Layer From Background and ordinary layer locks.

use photocraft_doc::{Layer, LayerId};
use photocraft_engine::Session;
use serde_json::json;

fn background(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 12, "depth": depth, "background": "white"})).unwrap();
    s
}

fn layer<'a>(s: &'a Session, name: &str) -> &'a Layer {
    s.active().unwrap().doc.layers.iter().find(|l| l.name == name).unwrap()
}

fn active(s: &Session) -> &Layer {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap()
}

fn pixel(s: &Session, x: i32, y: i32) -> [f32; 4] {
    active(s).surface().unwrap().rgba(x, y)
}

fn select(s: &mut Session, id: LayerId, mode: &str) {
    s.execute("layer.select", json!({"layer": id.0, "mode": mode})).unwrap();
}

#[test]
fn merge_down_keeps_background_locked_and_clear_or_cut_fills_instead_of_erasing() {
    for depth in [8, 16, 32] {
        for command in ["edit.clear", "edit.cut"] {
            let mut s = background(depth);
            let id = active(&s).id;
            s.execute("layer.new.layer", json!({"name": "Top"})).unwrap();
            s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
            s.execute("layer.mergeDown", json!({})).unwrap();

            assert_eq!(s.active().unwrap().doc.layers.len(), 1, "{depth}/{command}");
            assert_eq!(active(&s).id, id, "{depth}/{command}: retain lower layer id");
            assert_eq!(active(&s).name, "Background");
            assert!(active(&s).locks.transparency && active(&s).locks.position);
            assert!(s.is_enabled("layer.new.layerFromBackground"));
            assert_eq!(pixel(&s, 5, 4), [1.0, 0.0, 0.0, 1.0]);
            let bounds = active(&s).surface().unwrap().content_bounds();
            let history = s.active().unwrap().history.past_len();
            assert!(s.execute("layer.translate", json!({"dx": 2, "dy": 1})).is_err());
            assert_eq!(active(&s).surface().unwrap().content_bounds(), bounds);
            assert_eq!(s.active().unwrap().history.past_len(), history);

            s.execute("tools.setColors", json!({"background": "#00ff00"})).unwrap();
            s.execute("select.rect", json!({"x": 4, "y": 3, "width": 4, "height": 4})).unwrap();
            s.execute(command, json!({})).unwrap();
            assert_eq!(pixel(&s, 5, 4), [0.0, 1.0, 0.0, 1.0], "{depth}/{command}: fill the Background");
            assert_eq!(pixel(&s, 1, 1), [1.0, 0.0, 0.0, 1.0], "{depth}/{command}: outside selection");
            assert!(s.undo(), "{depth}/{command}: undo");
            assert_eq!(pixel(&s, 5, 4), [1.0, 0.0, 0.0, 1.0]);
            assert!(s.redo(), "{depth}/{command}: redo");
            assert_eq!(pixel(&s, 5, 4), [0.0, 1.0, 0.0, 1.0]);
        }
    }
}

#[test]
fn merge_down_onto_an_ordinary_layer_does_not_create_a_background() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 12, "background": "transparent"})).unwrap();
    let below = active(&s).id;
    s.execute("layer.new.layer", json!({"name": "Top"})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
    s.execute("layer.mergeDown", json!({})).unwrap();
    assert_eq!(active(&s).id, below);
    assert!(!active(&s).locks.transparency && !active(&s).locks.position);
    assert!(!s.is_enabled("layer.new.layerFromBackground"));
    s.execute("select.rect", json!({"x": 4, "y": 3, "width": 4, "height": 4})).unwrap();
    s.execute("edit.clear", json!({})).unwrap();
    assert_eq!(pixel(&s, 5, 4)[3], 0.0);
    assert!(s.execute("layer.translate", json!({"dx": 1, "dy": 1})).is_ok());
}

#[test]
fn multi_layer_duplication_unlocks_only_the_background_copy() {
    let mut s = background(8);
    let bg = active(&s).id;
    let ordinary = s.execute("layer.new.layer", json!({"name": "Other"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.setProps", json!({"layer": ordinary, "locks": {"position": true}})).unwrap();
    select(&mut s, bg, "replace");
    select(&mut s, LayerId(ordinary), "add");
    assert_eq!(s.active().unwrap().selected_layers().len(), 2);
    s.execute("layer.duplicate", json!({})).unwrap();

    assert_eq!(s.active().unwrap().doc.layers.len(), 4);
    let bg_copy = layer(&s, "Background copy");
    assert!(!bg_copy.locks.transparency && !bg_copy.locks.position && !bg_copy.locks.all);
    let bg_copy_id = bg_copy.id;
    assert!(layer(&s, "Background").locks.transparency && layer(&s, "Background").locks.position);
    let other_copy = layer(&s, "Other copy");
    assert!(other_copy.locks.position, "ordinary layer locks must survive duplication");
    let other_copy_id = other_copy.id;
    select(&mut s, other_copy_id, "replace");
    assert!(s.execute("layer.translate", json!({"dx": 1})).is_err());
    select(&mut s, bg_copy_id, "replace");
    assert!(s.execute("layer.translate", json!({"dx": 1})).is_ok());
}

#[test]
fn transparent_canvas_extension_converts_background_to_a_paintable_layer() {
    for depth in [8, 16, 32] {
        let mut s = background(depth);
        let original = active(&s).id;
        s.execute(
            "image.canvasSize",
            json!({
                "width": 24, "height": 12, "anchor": "left", "extensionColor": "transparent"
            }),
        )
        .unwrap();
        assert_eq!(active(&s).id, original);
        assert_eq!(active(&s).name, "Layer 0");
        assert!(!active(&s).locks.transparency && !active(&s).locks.position);
        assert!(!s.is_enabled("layer.new.layerFromBackground"));
        assert_eq!(pixel(&s, 2, 4), [1.0, 1.0, 1.0, 1.0], "{depth}: original area");
        assert_eq!(pixel(&s, 20, 4)[3], 0.0, "{depth}: transparent extension");

        s.execute("select.rect", json!({"x": 18, "y": 2, "width": 4, "height": 4})).unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
        assert_eq!(pixel(&s, 20, 4), [1.0, 0.0, 0.0, 1.0], "{depth}: extension is paintable");
        assert_eq!(pixel(&s, 23, 4)[3], 0.0, "{depth}: outside selection");
        assert_eq!(pixel(&s, 2, 4), [1.0, 1.0, 1.0, 1.0]);
        assert!(s.undo());
        assert_eq!(pixel(&s, 20, 4)[3], 0.0);
    }
}

#[test]
fn opaque_or_no_op_canvas_size_keeps_the_background() {
    let mut s = background(8);
    s.execute(
        "image.canvasSize",
        json!({
            "width": 24, "height": 12, "anchor": "left", "extensionColor": "#00ff00"
        }),
    )
    .unwrap();
    assert_eq!(active(&s).name, "Background");
    assert!(active(&s).locks.transparency && active(&s).locks.position);
    assert_eq!(pixel(&s, 20, 4), [0.0, 1.0, 0.0, 1.0]);
    assert!(s.undo());

    s.execute(
        "image.canvasSize",
        json!({
            "width": 16, "height": 12, "anchor": "left", "extensionColor": "transparent"
        }),
    )
    .unwrap();
    assert_eq!(active(&s).name, "Background");
    assert!(s.is_enabled("layer.new.layerFromBackground"));
}
