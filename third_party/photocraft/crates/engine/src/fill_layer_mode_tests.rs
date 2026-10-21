//! Fill layers made in a Grayscale, CMYK or Lab document keep their colours in the document's
//! mode, so they composite as the same colour painted on a pixel layer does (#1905).

use crate::Session;
use photocraft_doc::{ColorMode, Fill, LayerContent};
use serde_json::{Value, json};

const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

fn session(mode: &str, depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 32, "mode": mode, "depth": depth, "background": "white"})).unwrap();
    s.tools.foreground = BLUE;
    s.tools.background = GREEN;
    s
}

fn pixel(s: &mut Session, x: i32, y: i32) -> [f32; 4] {
    let v = s.execute("document.pixel", json!({"x": x, "y": y})).unwrap();
    let a: Vec<f32> = v.as_array().unwrap().iter().map(|c| c.as_f64().unwrap() as f32).collect();
    [a[0], a[1], a[2], a[3]]
}

/// The composite of `color` painted with Edit › Fill on a new pixel layer: what a fill layer of
/// that colour must look like.
fn painted(mode: &str, depth: u32, color: [f32; 4]) -> [f32; 4] {
    let mut s = session(mode, depth);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": color})).unwrap();
    pixel(&mut s, 10, 10)
}

fn close(a: [f32; 4], b: [f32; 4], tol: f32) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
}

fn active_fill(s: &Session) -> Fill {
    let st = s.active().unwrap();
    match &st.doc.layer(st.active_layer.unwrap()).unwrap().content {
        LayerContent::Fill(f) => f.clone(),
        other => panic!("not a fill layer: {other:?}"),
    }
}

fn fill_colors(f: &Fill) -> Vec<photocraft_doc::Color> {
    match f {
        Fill::Solid(c) => vec![*c],
        Fill::Gradient { stops, .. } => stops.iter().map(|s| s.1).collect(),
        Fill::Pattern { .. } => Vec::new(),
    }
}

fn assert_in_mode(s: &Session, mode: ColorMode, what: &str) {
    let f = active_fill(s);
    for c in fill_colors(&f) {
        assert_eq!(c.mode, mode, "{what}: {f:?}");
    }
}

#[test]
fn solid_color_fill_matches_edit_fill() {
    for (mode, cm) in [("gray", ColorMode::Grayscale), ("cmyk", ColorMode::Cmyk), ("lab", ColorMode::Lab)] {
        for depth in [8, 16, 32] {
            let mut s = session(mode, depth);
            s.execute("layer.newFillLayer.solidColor", json!({"color": "#0000ff"})).unwrap();
            assert_in_mode(&s, cm, mode);
            let got = pixel(&mut s, 10, 10);
            let want = painted(mode, depth, BLUE);
            assert!(close(got, want, 0.01), "{mode}/{depth}: fill layer {got:?}, painted {want:?}");
            if cm == ColorMode::Grayscale {
                assert!((got[0] - 0.114).abs() < 0.01 && got[0] == got[1] && got[1] == got[2], "{mode}/{depth}: {got:?}");
            }
        }
    }
}

#[test]
fn gradient_fill_layers_are_gray_in_a_grayscale_document() {
    let grey = |p: [f32; 4]| (p[0] - p[1]).abs() < 1e-4 && (p[1] - p[2]).abs() < 1e-4;
    let ways: [(&str, Value); 3] = [
        ("layer.newFillLayer.gradient", json!({"from": "#0000ff", "to": "#00ff00", "angle": 0})),
        ("gradient.fill.create", json!({"from": [0, 16], "to": [64, 16]})),
        ("gradient.presets.apply", json!({"preset": "Blue 01", "angle": 0})),
    ];
    for depth in [8, 16, 32] {
        for (cmd, p) in &ways {
            let mut s = session("gray", depth);
            s.execute(cmd, p.clone()).unwrap();
            assert_in_mode(&s, ColorMode::Grayscale, cmd);
            for x in [1, 32, 62] {
                let px = pixel(&mut s, x, 16);
                assert!(grey(px), "{cmd} at 8/16/32 = {depth}, x {x}: {px:?}");
            }
        }
    }
}

#[test]
fn gradient_edits_keep_colours_in_the_document_mode() {
    let mut s = session("cmyk", 8);
    s.execute("gradient.fill.create", json!({"from": [0, 16], "to": [64, 16]})).unwrap();
    s.execute("gradient.fill.set", json!({"stops": [[0, "#ff0000"], [1, "#0000ff"]]})).unwrap();
    assert_in_mode(&s, ColorMode::Cmyk, "gradient.fill.set");
    let start = pixel(&mut s, 0, 16);
    let red = painted("cmyk", 8, [1.0, 0.0, 0.0, 1.0]);
    assert!(close(start, red, 0.02), "{start:?} vs {red:?}");
    s.execute("gradient.presets.select", json!({"preset": "Green 01"})).unwrap();
    assert_in_mode(&s, ColorMode::Cmyk, "gradient.presets.select");
}

#[test]
fn rgb_documents_keep_rgb_fill_colours() {
    let mut s = session("rgb", 8);
    s.execute("layer.newFillLayer.solidColor", json!({"color": "#0000ff"})).unwrap();
    assert_eq!(active_fill(&s), Fill::Solid(photocraft_doc::Color::rgb(0.0, 0.0, 1.0)));
    assert!(close(pixel(&mut s, 10, 10), BLUE, 1e-6));
}

/// A grayscale fill layer converts like any other colour when the mode changes, and a fill made
/// in Gray survives undo/redo unchanged.
#[test]
fn gray_fill_converts_with_the_document() {
    let mut s = session("gray", 16);
    // Tagged sGray, where a gray value reads the same as RGB, so the before/after pixels compare
    // directly (an untagged gray document takes the Dot Gain 20 %-compatible default).
    s.execute("edit.assignProfile", json!({"profile": "sgray"})).unwrap();
    s.execute("layer.newFillLayer.solidColor", json!({"color": "#0000ff"})).unwrap();
    let before = pixel(&mut s, 10, 10);
    s.execute("image.mode.rgb", json!({})).unwrap();
    assert_in_mode(&s, ColorMode::Rgb, "after Image › Mode › RGB");
    let after = pixel(&mut s, 10, 10);
    assert!(close(before, after, 0.01), "{before:?} -> {after:?}");
    s.execute("edit.undo", json!({})).unwrap();
    assert_in_mode(&s, ColorMode::Grayscale, "undo");
}
