//! Regression tests for issue #706: sigmas in [0.3, ~0.577) must blur, not silently no-op.
use photocraft_engine::Session;
use serde_json::json;

fn px(s: &mut Session, x: i32, y: i32) -> Vec<f32> {
    serde_json::from_value(s.execute("document.pixel", json!({"x": x, "y": y})).unwrap()).unwrap()
}

fn edge_doc() -> Session {
    let mut s = Session::new();
    // Mid-tone, slightly chromatic halves so sharpening is not clipped away.
    s.execute("file.new", json!({"width": 32, "height": 32, "background": "#602020"})).unwrap();
    s.execute("select.rect", json!({"x": 16, "y": 0, "width": 16, "height": 32})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#206020"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s
}

fn row(s: &mut Session) -> Vec<Vec<f32>> {
    (12..20).map(|x| px(s, x, 16)).collect()
}

#[test]
fn camera_raw_sharpen_radius_half_noop() {
    let mut base_s = edge_doc();
    let base = row(&mut base_s);
    let mut ctl_s = edge_doc();
    ctl_s.execute("filter.cameraRaw", json!({"sharpenAmount":150,"sharpenRadius":3.0,"sharpenDetail":100})).unwrap();
    let ctl = row(&mut ctl_s);
    let mut half_s = edge_doc();
    half_s.execute("filter.cameraRaw", json!({"sharpenAmount":150,"sharpenRadius":0.5,"sharpenDetail":100})).unwrap();
    let half = row(&mut half_s);
    assert_ne!(base, ctl, "control: sharpenRadius 3.0 must change pixels");
    assert_ne!(base, half, "BUG: sharpenRadius 0.5 changed nothing");
}
