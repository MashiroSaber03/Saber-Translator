//! An SVG placed as a smart object is a vector smart object: it is rasterised at its placement
//! scale (sharp at 10×, where a 20×10 px raster would blur), re-rendered when the placement
//! changes, and its contents open as shape layers.

use photocraft_doc::LayerContent;
use photocraft_engine::Session;
use photocraft_geom::Rect;
use serde_json::json;

const SVG: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"10\">\
  <rect width=\"20\" height=\"10\" fill=\"#ff0000\"/>\
  <rect x=\"0\" y=\"4\" width=\"20\" height=\"0.5\" fill=\"#000000\"/>\
</svg>";

fn alpha(s: &photocraft_raster::Surface, x: i32, y: i32) -> f32 {
    let px = s.read_region(Rect::new(x, y, x + 1, y + 1));
    *px.last().unwrap()
}

fn red(s: &photocraft_raster::Surface, x: i32, y: i32) -> f32 {
    let px = s.read_region(Rect::new(x, y, x + 1, y + 1));
    px[0]
}

#[test]
fn placed_svg_is_sharp_at_its_placement_scale_and_opens_as_shapes() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 400, "height": 400, "background": "white"})).unwrap();
    let r = photocraft_engine::file_cmds::place_bytes(&mut s, "logo.svg", SVG.as_bytes().to_vec(), None, &json!({"scale": 1000})).unwrap();
    assert_eq!(r["bounds"], json!([100.0, 150.0, 300.0, 250.0]), "{r}");
    let doc = s.active().unwrap().doc.clone();
    let layer = doc.layers.last().unwrap();
    let LayerContent::Smart(sm) = &layer.content else { panic!("{}", layer.content.kind_name()) };
    let surf = layer.surface().expect("rendered");
    // A pixel-exact edge at the top of the 200 × 100 placement, not a bicubic ramp.
    assert!(alpha(surf, 200, 148) < 0.01 && alpha(surf, 200, 151) > 0.99, "{} {}", alpha(surf, 200, 148), alpha(surf, 200, 151));
    // The half-unit black line is a crisp 5 px band (rows 190..195) rather than a faint smear.
    assert!(red(surf, 200, 192) < 0.02, "black inside the line: {}", red(surf, 200, 192));
    assert!(red(surf, 200, 187) > 0.98 && red(surf, 200, 198) > 0.98, "red just outside it");
    assert!(red(surf, 200, 189) > 0.9 || red(surf, 200, 189) < 0.1, "edge rows are not half-covered");

    // Edit Contents on this smart object: the embedded source opens as shape layers.
    let photocraft_doc::SmartSource::Embedded { file_name, bytes } = &sm.source else { panic!("embedded") };
    let inner = photocraft_engine::smart_cmds::decode_source(file_name, bytes).unwrap();
    assert_eq!(inner.layers.len(), 2);
    assert!(inner.layers.iter().all(|l| matches!(l.content, LayerContent::Shape(_))));
}

#[test]
fn re_placing_at_another_scale_re_renders_the_vector_source() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 300, "height": 300, "background": "white"})).unwrap();
    photocraft_engine::file_cmds::place_bytes(&mut s, "logo.svg", SVG.as_bytes().to_vec(), None, &json!({"scale": 100})).unwrap();
    let doc = s.active().unwrap().doc.clone();
    let layer = doc.layers.last().unwrap();
    let LayerContent::Smart(sm) = &layer.content else { panic!() };
    let mut big = sm.clone();
    let [a, b, c, d, _, _] = big.transform.m;
    big.transform.m = [a * 12.0, b * 12.0, c * 12.0, d * 12.0, 30.0, 30.0];
    let surf = photocraft_engine::smart_cmds::render(&doc, &big).unwrap().expect("source available");
    // 20 × 10 units at 12× from (30, 30): a hard edge at x = 270 and the thin line at rows 78..84.
    assert!(alpha(&surf, 268, 60) > 0.99 && alpha(&surf, 271, 60) < 0.01);
    assert!(red(&surf, 150, 81) < 0.02 && red(&surf, 150, 75) > 0.98 && red(&surf, 150, 87) > 0.98);
}
