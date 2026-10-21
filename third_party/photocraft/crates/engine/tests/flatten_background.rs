//! Flatten Image must leave the same Background behaviour as a newly created document (#1105).

use photocraft_doc::Layer;
use photocraft_engine::Session;
use serde_json::json;

fn layered(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 12, "depth": depth})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
    s
}

fn active(s: &Session) -> &Layer {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap()
}

fn pixel(s: &Session, x: i32, y: i32) -> [f32; 4] {
    active(s).surface().unwrap().rgba(x, y)
}

fn history_len(s: &Session) -> usize {
    s.active().unwrap().history.past_len()
}

#[test]
fn flatten_creates_a_background_in_one_undo_step_at_every_depth() {
    for depth in [8, 16, 32] {
        let mut s = layered(depth);
        let original_ids = s.active().unwrap().doc.layers.iter().map(|l| l.id).collect::<Vec<_>>();
        let original_active = s.active().unwrap().active_layer;
        let format = s.active().unwrap().doc.pixel_format();
        let history = history_len(&s);

        s.execute("layer.flattenImage", json!({})).unwrap();
        let background_id = active(&s).id;
        assert_eq!(s.active().unwrap().doc.layers.len(), 1, "{depth}: one Background");
        assert_eq!(active(&s).name, "Background");
        assert_eq!(active(&s).surface().unwrap().format(), format);
        assert_eq!(pixel(&s, 8, 6), [1.0, 0.0, 0.0, 1.0]);
        assert!(s.is_enabled("layer.new.layerFromBackground"), "{depth}: recognises the Background");
        assert_eq!(history_len(&s), history + 1, "{depth}: one undo step");

        assert!(s.undo());
        assert_eq!(s.active().unwrap().doc.layers.iter().map(|l| l.id).collect::<Vec<_>>(), original_ids);
        assert_eq!(s.active().unwrap().active_layer, original_active);
        assert_eq!(pixel(&s, 8, 6), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(history_len(&s), history);

        assert!(s.redo());
        assert_eq!(s.active().unwrap().doc.layers.len(), 1);
        assert_eq!(active(&s).id, background_id);
        assert!(s.is_enabled("layer.new.layerFromBackground"), "{depth}: redo restores the Background");
        assert_eq!(pixel(&s, 8, 6), [1.0, 0.0, 0.0, 1.0]);
    }
}

#[test]
fn clear_and_cut_fill_the_flattened_background_without_transparency() {
    for depth in [8, 16, 32] {
        for command in ["edit.clear", "edit.cut"] {
            let mut s = layered(depth);
            s.execute("layer.flattenImage", json!({})).unwrap();
            s.execute("tools.setColors", json!({"background": "#00ff00"})).unwrap();
            s.execute("select.rect", json!({"x": 4, "y": 3, "width": 8, "height": 6})).unwrap();
            let history = history_len(&s);

            s.execute(command, json!({})).unwrap();
            assert_eq!(pixel(&s, 8, 6), [0.0, 1.0, 0.0, 1.0], "{depth}/{command}: background colour, still opaque");
            assert_eq!(pixel(&s, 1, 1), [1.0, 0.0, 0.0, 1.0], "{depth}/{command}: outside the selection is untouched");
            assert!(s.is_enabled("layer.new.layerFromBackground"));
            assert_eq!(history_len(&s), history + 1);

            assert!(s.undo());
            assert_eq!(pixel(&s, 8, 6), [1.0, 0.0, 0.0, 1.0]);
            assert!(s.redo());
            assert_eq!(pixel(&s, 8, 6), [0.0, 1.0, 0.0, 1.0]);
        }
    }
}

#[test]
fn flattened_background_refuses_movement_until_converted_to_a_layer() {
    for depth in [8, 16, 32] {
        let mut s = layered(depth);
        s.execute("layer.flattenImage", json!({})).unwrap();
        let id = active(&s).id;
        let bounds = active(&s).surface().unwrap().content_bounds();
        let history = history_len(&s);

        let error = s.execute("layer.translate", json!({"dx": 3, "dy": 2})).unwrap_err();
        assert!(error.to_string().contains("position-locked"), "{depth}: {error}");
        assert_eq!(history_len(&s), history, "{depth}: a rejected move adds no history");
        assert_eq!(active(&s).id, id);
        assert_eq!(active(&s).surface().unwrap().content_bounds(), bounds);
        assert_eq!(pixel(&s, 0, 0), [1.0, 0.0, 0.0, 1.0]);

        s.execute("layer.new.layerFromBackground", json!({})).unwrap();
        assert_eq!(active(&s).id, id);
        assert_eq!(active(&s).name, "Layer 0");
        assert!(!active(&s).locks.position && !active(&s).locks.transparency);
        assert!(!s.is_enabled("layer.new.layerFromBackground"));
        assert_eq!(history_len(&s), history + 1);

        assert!(s.undo());
        assert_eq!(active(&s).name, "Background");
        assert!(s.is_enabled("layer.new.layerFromBackground"));
        assert!(s.redo());
        s.execute("layer.translate", json!({"dx": 3, "dy": 2})).unwrap();
        assert_eq!(pixel(&s, 0, 0)[3], 0.0, "{depth}: a normal layer can move");
        assert_eq!(pixel(&s, 3, 2), [1.0, 0.0, 0.0, 1.0]);
    }
}

#[test]
fn flatten_without_a_document_fails_gracefully() {
    let mut s = Session::new();
    assert!(!s.is_enabled("layer.flattenImage"));
    assert!(s.execute("layer.flattenImage", json!({})).is_err());
    assert!(s.active().is_none());
}
