//! Regression tests for issue #708: `filter.blurGallery.pathBlur` clamps the
//! scalar `speed`, but each entry of the documented `paths` array is
//! deserialized without a clamp. A huge finite speed saturates the
//! float-to-i32 cast in the halo and the `+ 1` overflows, which the dispatch
//! turns into an internal-error panic. The command must clamp like the scalar
//! path does and blur normally.

use photocraft_engine::Session;
use serde_json::json;

fn path_blur(speed: f64) -> Result<serde_json::Value, photocraft_engine::EngineError> {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "#808080"})).unwrap();
    s.execute("filter.blurGallery.pathBlur", json!({"paths": [{"points": [[0.2, 0.5], [0.8, 0.5]], "speed": speed}]}))
}

#[test]
fn path_blur_huge_entry_speed_does_not_panic() {
    // On main this comes back as a caught panic ("internal error" style)
    // from overflow in the halo radius. After the clamp it must succeed.
    let r = path_blur(1.0e20);
    assert!(r.is_ok(), "huge entry speed must clamp, got error: {r:?}");
}

#[test]
fn path_blur_negative_entry_speed_does_not_panic() {
    let r = path_blur(-1.0e20);
    assert!(r.is_ok(), "huge negative entry speed must clamp, got error: {r:?}");
}

#[test]
fn path_blur_normal_entry_speed_still_blurs() {
    // Control: a documented in-range speed keeps working.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "#808080"})).unwrap();
    s.execute("filter.blurGallery.pathBlur", json!({"paths": [{"points": [[0.2, 0.5], [0.8, 0.5]], "speed": 120.0}]})).expect("in-range speed must succeed");
}
