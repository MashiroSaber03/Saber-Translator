use photocraft_engine::Session;
use serde_json::json;

fn live_line_info(s: &mut Session, layer: u64) -> serde_json::Value {
    s.execute("shape.info", json!({"layer": layer})).unwrap()
}

#[test]
fn live_line_keeps_scaled_thickness_after_endpoint_edit() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 128, "height": 128, "background": "transparent"})).unwrap();
    let layer = s.execute("shape.create", json!({"kind": "line", "from": [10,20], "to": [30,20], "weight": 4})).unwrap()["layer"].as_u64().unwrap();

    s.execute("shape.edit", json!({"transform": [2,0,0,2,0,0]})).unwrap();
    let after_scale = live_line_info(&mut s, layer);
    assert_eq!(after_scale["bounds"], json!([20, 36, 40, 8]));
    assert_eq!(after_scale["live"]["weight"], json!(8.0));

    s.execute("shape.edit", json!({"to": [60,40]})).unwrap();
    let after_edit = live_line_info(&mut s, layer);
    assert_eq!(after_edit["bounds"], after_scale["bounds"]);
    assert_eq!(after_edit["path"], after_scale["path"]);
    assert_eq!(after_edit["live"]["weight"], json!(8.0));
}

#[test]
fn translating_live_lines_leaves_their_weight_unchanged() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 128, "height": 128, "background": "transparent"})).unwrap();
    let layer = s.execute("shape.create", json!({"kind": "line", "from": [10,20], "to": [30,20], "weight": 4})).unwrap()["layer"].as_u64().unwrap();
    s.execute("shape.edit", json!({"move": [10,10]})).unwrap();
    let after_move = live_line_info(&mut s, layer);
    s.execute("shape.edit", json!({"to": [40,30]})).unwrap();
    let after_edit = live_line_info(&mut s, layer);
    assert_eq!(after_edit["path"], after_move["path"]);
    assert_eq!(after_edit["live"]["weight"], json!(4.0));
}

#[test]
fn nonuniform_scaling_uses_the_direction_perpendicular_to_the_line() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 128, "height": 128, "background": "transparent"})).unwrap();
    let horizontal = s.execute("shape.create", json!({"kind": "line", "from": [10,20], "to": [30,20], "weight": 4})).unwrap()["layer"].as_u64().unwrap();
    s.execute("shape.edit", json!({"transform": [2,0,0,1,0,0]})).unwrap();
    assert_eq!(live_line_info(&mut s, horizontal)["live"]["weight"], json!(4.0));
    let vertical = s.execute("shape.create", json!({"kind": "line", "from": [50,10], "to": [50,30], "weight": 4})).unwrap()["layer"].as_u64().unwrap();
    s.execute("shape.edit", json!({"transform": [2,0,0,1,0,0]})).unwrap();
    assert_eq!(live_line_info(&mut s, vertical)["live"]["weight"], json!(8.0));
}
