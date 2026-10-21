//! #707: `filter.cameraRaw` must clamp the noise-reduction amounts to their
//! documented 0..100 range. Before the fix, `raw_params` deserialized the
//! params verbatim, so `noiseLuminance: 1e9` reached the guided filter as a
//! radius of ~3e9 and the command never finished.

use serde_json::json;

#[test]
fn camera_raw_clamps_noise_amounts() {
    let p = photocraft_engine::lens_cmds::raw_params("filter.cameraRaw", &json!({"noiseLuminance": 1e9, "noiseColor": 1e9})).expect("params parse");
    assert!((p.noise_luminance - 100.0).abs() < f32::EPSILON, "noiseLuminance 1e9 must clamp to 100, got {}", p.noise_luminance);
    assert!((p.noise_color - 100.0).abs() < f32::EPSILON, "noiseColor 1e9 must clamp to 100, got {}", p.noise_color);
}

#[test]
fn camera_raw_clamps_negative_noise_and_details() {
    let p = photocraft_engine::lens_cmds::raw_params(
        "filter.cameraRaw",
        &json!({
            "noiseLuminance": -5.0,
            "noiseColor": -5.0,
            "noiseLuminanceDetail": 1e9,
            "noiseColorDetail": -1.0
        }),
    )
    .expect("params parse");
    assert_eq!(p.noise_luminance, 0.0, "negative noiseLuminance must clamp to 0");
    assert_eq!(p.noise_color, 0.0, "negative noiseColor must clamp to 0");
    assert_eq!(p.noise_luminance_detail, 100.0, "detail must clamp to 100");
    assert_eq!(p.noise_color_detail, 0.0, "negative detail must clamp to 0");
}

#[test]
fn camera_raw_keeps_in_range_noise_untouched() {
    let p = photocraft_engine::lens_cmds::raw_params("filter.cameraRaw", &json!({"noiseLuminance": 25.0, "noiseColor": 40.0})).expect("params parse");
    assert_eq!(p.noise_luminance, 25.0);
    assert_eq!(p.noise_color, 40.0);
}
