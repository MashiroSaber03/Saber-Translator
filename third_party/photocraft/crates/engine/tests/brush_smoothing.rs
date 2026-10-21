//! Smoothing is a tool option: the Brush starts at Photoshop's 10 %, and picking a brush preset
//! keeps the current smoothing (amount and mode) instead of taking the preset's.

use photocraft_engine::Session;
use serde_json::json;

#[test]
fn presets_keep_the_tool_smoothing() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    assert_eq!(s.tools.brush.smoothing.amount, 0.1, "Photoshop's default");
    let preset = s.tools.presets.iter().find(|p| p.brush.smoothing.amount == 0.0).map(|p| p.name.clone()).unwrap();
    let smoothed = s.tools.presets.iter().find(|p| p.brush.smoothing.amount > 0.2).map(|p| p.name.clone()).unwrap();
    s.execute("tools.setBrush", json!({"preset": preset})).unwrap();
    assert_eq!(s.tools.brush.smoothing.amount, 0.1);
    s.execute("tools.setBrush", json!({"brush": {"smoothing": {"amount": 0.6, "pulledString": true}}})).unwrap();
    s.execute("tools.setBrush", json!({"preset": smoothed})).unwrap();
    assert_eq!((s.tools.brush.smoothing.amount, s.tools.brush.smoothing.pulled_string), (0.6, true));
    // An explicit smoothing alongside the preset still applies.
    s.execute("tools.setBrush", json!({"preset": preset, "brush": {"smoothing": {"amount": 0.0}}})).unwrap();
    assert_eq!(s.tools.brush.smoothing.amount, 0.0);
}

#[test]
fn smoothing_switch_is_drivable_and_presets_keep_the_disabled_settings() {
    let mut s = Session::new();
    let params = json!({"brush": {"smoothing": {"enabled": false, "amount": 0.6, "pulledString": true, "catchUpOnEnd": false}}});
    s.execute("tools.setBrush", params.clone()).unwrap();
    let off = s.tools.brush.smoothing.clone();
    assert!(!off.enabled && off.amount == 0.6 && off.pulled_string && !off.catch_up_on_end);
    let preset = s.tools.presets[0].name.clone();
    s.execute("tools.setBrush", json!({"preset": preset})).unwrap();
    assert_eq!(s.tools.brush.smoothing, off);
    s.execute("tools.setBrush", json!({"brush": {"smoothing": {"enabled": true}}})).unwrap();
    assert!(s.tools.brush.smoothing.enabled);
    assert_eq!(s.tools.brush.smoothing.amount, off.amount);
    assert_eq!(s.tools.brush.smoothing.pulled_string, off.pulled_string);
    // The same parameter patch reproduces the off state in another session (CLI/MCP path).
    let mut other = Session::new();
    other.execute("tools.setBrush", params).unwrap();
    assert_eq!(other.tools.brush.smoothing, off);
    let before = other.tools.brush.clone();
    assert!(other.execute("tools.setBrush", json!({"brush": {"smoothing": {"enabled": "no"}}})).is_err());
    assert_eq!(other.tools.brush, before);
}

#[test]
fn an_explicit_stroke_smoothing_amount_overrides_the_tool_switch() {
    let mut s = Session::new();
    s.execute("tools.setBrush", json!({"brush": {"smoothing": {"enabled": false, "amount": 0.6}}})).unwrap();
    let inherited = photocraft_engine::brush_cmds::resolve_brush(&s, &json!({}), "paint.stroke").unwrap();
    assert!(!inherited.smoothing.enabled && inherited.smoothing.amount == 0.6);
    let explicit = photocraft_engine::brush_cmds::resolve_brush(&s, &json!({"smoothing": 0.4}), "paint.stroke").unwrap();
    assert!(explicit.smoothing.enabled && explicit.smoothing.amount == 0.4);
    assert!(!s.tools.brush.smoothing.enabled, "a stroke override does not change the tool");
}
