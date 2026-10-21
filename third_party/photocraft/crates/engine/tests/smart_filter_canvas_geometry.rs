use photocraft_engine::Session;
use serde_json::{Value, json};

fn fixture() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
    s.execute("filter.convertForSmartFilters", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 8, "height": 16})).unwrap();
    s.execute("filter.stylize.solarize", json!({})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s
}

fn pixel(s: &mut Session, x: i32, y: i32) -> Value {
    s.execute("document.pixel", json!({"x": x, "y": y})).unwrap()
}

fn check_black(s: &mut Session, x: i32, y: i32) {
    let p = pixel(s, x, y);
    assert!(p[0].as_f64().unwrap_or(1.0) < 0.01, "({x},{y}) should be filtered black, got {p}");
}

fn check_red(s: &mut Session, x: i32, y: i32) {
    let p = pixel(s, x, y);
    assert!(p[0].as_f64().unwrap_or(0.0) > 0.99 && p[1].as_f64().unwrap_or(1.0) < 0.01, "({x},{y}) should remain red, got {p}");
}

fn rerender_filter(s: &mut Session) {
    s.execute("layer.smartFilter.setVisible", json!({"index": 0, "visible": false})).unwrap();
    s.execute("layer.smartFilter.setVisible", json!({"index": 0, "visible": true})).unwrap();
}

#[test]
fn image_size_resamples_selection_derived_smart_filter_mask() {
    let mut s = fixture();
    check_black(&mut s, 4, 8);
    check_red(&mut s, 12, 8);
    s.execute("image.imageSize", json!({"width": 32, "height": 32, "resample": "nearest"})).unwrap();
    rerender_filter(&mut s);
    for (x, y) in [(4, 8), (12, 8), (4, 16)] {
        check_black(&mut s, x, y);
    }
    check_red(&mut s, 24, 8);
    assert!(s.undo());
}

#[test]
fn canvas_size_keeps_smart_filter_coverage_after_rerender() {
    let mut s = fixture();
    s.execute("image.canvasSize", json!({"width": 32, "height": 16, "anchor": "right", "extensionColor": "transparent"})).unwrap();
    check_black(&mut s, 20, 8);
    rerender_filter(&mut s);
    check_black(&mut s, 20, 8);
    check_red(&mut s, 28, 8);
}

#[test]
fn crop_keeps_smart_filter_coverage_after_rerender() {
    let mut s = fixture();
    s.execute("image.crop", json!({"x": 4, "y": 0, "width": 12, "height": 16})).unwrap();
    check_red(&mut s, 6, 8);
    rerender_filter(&mut s);
    check_red(&mut s, 6, 8);
    check_red(&mut s, 10, 8);
}
