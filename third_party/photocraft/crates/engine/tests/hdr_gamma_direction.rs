//! Regression tests for issue #710: HdrToning gamma is documented as
//! "1 = neutral; lower = more contrast", so a gamma below 1 must darken
//! midtones (like the crate's other gamma controls, which use 1/gamma as the
//! exponent). On main the value was used directly as the exponent, so the
//! slider ran opposite to its contract.
use photocraft_engine::Session;
use serde_json::json;

fn midtone_luma(gamma: f64) -> f64 {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8, "background": "#808080"})).unwrap();
    s.execute("image.adjustments.hdrToning", json!({"gamma": gamma, "strength": 0.1, "detail": 0, "saturation": 0, "radius": 30})).unwrap();
    let p: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": 4, "y": 4})).unwrap()).unwrap();
    f64::from(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
}

#[test]
fn hdr_toning_gamma_lower_darkens_midtones() {
    let base = midtone_luma(1.0);
    let low = midtone_luma(0.5);
    let high = midtone_luma(2.0);
    // "lower = more contrast" on a mid grey means the midtone moves down.
    assert!(low < base, "gamma 0.5 must darken the midtone: base={base} low={low}");
    assert!(high > base, "gamma 2.0 must lighten the midtone: base={base} high={high}");
}
