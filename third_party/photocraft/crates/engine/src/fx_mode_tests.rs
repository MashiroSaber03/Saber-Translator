//! Layer style colours made in a Grayscale or CMYK document are in the document's mode, so a red
//! Color Overlay in a Gray document composites grey, like red painted with Edit › Fill (the
//! effect counterpart of #1905; Image › Mode already converts them).

use crate::Session;
use serde_json::{Value, json};

const KINDS: [&str; 9] = ["dropShadow", "innerShadow", "outerGlow", "innerGlow", "satin", "colorOverlay", "gradientOverlay", "stroke", "bevelEmboss"];

fn session(mode: &str, depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 48, "height": 32, "mode": mode, "depth": depth, "background": "white"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 12, "y": 8, "width": 24, "height": 16})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#808080"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s
}

/// The modes of every colour stored in the active layer's effects (any `{mode, c, alpha}`).
fn effect_colour_modes(s: &Session) -> Vec<String> {
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(o) => {
                if let (Some(m), true, true) = (o.get("mode").and_then(Value::as_str), o.contains_key("c"), o.contains_key("alpha")) {
                    out.push(m.to_string());
                }
                o.values().for_each(|v| walk(v, out));
            }
            Value::Array(a) => a.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let st = s.active().unwrap();
    let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
    let mut out = Vec::new();
    for e in &l.effects.items {
        walk(&serde_json::to_value(e).unwrap(), &mut out);
    }
    out
}

fn pixel(s: &mut Session, x: i32, y: i32) -> [f32; 4] {
    let v = s.execute("document.pixel", json!({"x": x, "y": y})).unwrap();
    let a: Vec<f32> = v.as_array().unwrap().iter().map(|c| c.as_f64().unwrap() as f32).collect();
    [a[0], a[1], a[2], a[3]]
}

fn mode_name(mode: &str) -> &'static str {
    match mode {
        "gray" => "Grayscale",
        "cmyk" => "Cmyk",
        _ => "Rgb",
    }
}

fn assert_in_mode(s: &Session, mode: &str, what: &str) {
    let modes = effect_colour_modes(s);
    assert!(!modes.is_empty(), "{what}: no colours found");
    assert!(modes.iter().all(|m| m == mode_name(mode)), "{what}: colours {modes:?} in a {mode} document");
}

#[test]
fn every_effect_command_stores_colours_in_the_document_mode() {
    for mode in ["gray", "cmyk", "rgb"] {
        for depth in [8, 16, 32] {
            for kind in KINDS {
                let mut s = session(mode, depth);
                s.execute(&format!("layer.layerStyle.{kind}"), json!({"color": "#ff3300"})).unwrap();
                assert_in_mode(&s, mode, &format!("{mode}/{depth} {kind}"));
            }
        }
    }
}

#[test]
fn the_layer_style_dialog_stores_colours_in_the_document_mode() {
    for mode in ["gray", "cmyk"] {
        let mut s = session(mode, 8);
        let effects: Vec<Value> = KINDS.iter().map(|k| json!({"kind": k, "params": {"color": "#ff3300"}})).collect();
        s.execute("layer.layerStyle.replace", json!({ "effects": effects })).unwrap();
        assert_in_mode(&s, mode, &format!("{mode} layer.layerStyle.replace"));
    }
}

#[test]
fn a_red_color_overlay_is_grey_in_a_grayscale_document() {
    for depth in [8, 16, 32] {
        let mut s = session("gray", depth);
        s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff0000"})).unwrap();
        let p = pixel(&mut s, 20, 15);
        // Red painted into a Gray document: 0.299.
        assert!((p[0] - 0.299).abs() < 0.01 && p[0] == p[1] && p[1] == p[2], "{depth}-bit: {p:?}");
    }
}

#[test]
fn a_style_pasted_or_applied_into_a_grayscale_document_converts() {
    // Copied from an RGB document, pasted into a Gray one.
    let mut s = session("rgb", 8);
    s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff0000"})).unwrap();
    s.execute("layer.layerStyle.copyLayerStyle", json!({})).unwrap();
    s.execute("file.new", json!({"width": 48, "height": 32, "mode": "gray", "depth": 8, "background": "white"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 12, "y": 8, "width": 24, "height": 16})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#808080"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("layer.layerStyle.pasteLayerStyle", json!({})).unwrap();
    assert_in_mode(&s, "gray", "Paste Layer Style");
    let p = pixel(&mut s, 20, 15);
    assert!(p[0] == p[1] && p[1] == p[2], "pasted overlay is neutral: {p:?}");
    // A style preset (RGB colours) applied in a Gray document.
    let mut s = session("gray", 16);
    s.execute("style.presets.apply", json!({"preset": "Gold"})).unwrap();
    assert_in_mode(&s, "gray", "style.presets.apply Gold");
}
