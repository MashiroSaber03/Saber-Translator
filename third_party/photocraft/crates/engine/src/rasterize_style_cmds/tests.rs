use super::*;
use photocraft_doc::Document;

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn layer(s: &Session, id: u64) -> &Layer {
    doc(s).layer(LayerId(id)).unwrap()
}

fn composite(s: &Session) -> photocraft_compose::Buffer {
    let d = doc(s);
    photocraft_compose::render(d, d.bounds())
}

fn max_diff(a: &photocraft_compose::Buffer, b: &photocraft_compose::Buffer) -> f32 {
    a.px.iter().zip(&b.px).flat_map(|(p, q)| p.iter().zip(q).map(|(x, y)| (x - y).abs())).fold(0.0, f32::max)
}

/// The quantisation step of one stored sample: the baked layer is stored at the document's depth.
fn tolerance(depth: u32) -> f32 {
    match depth {
        8 => 2.5 / 255.0,
        16 => 2.0 / 32768.0 + 1e-4,
        _ => 1e-4,
    }
}

/// A white Background and a red rectangle (partly off the right edge of the canvas) with a
/// Normal-mode drop shadow, outer glow, inner shadow, outside stroke and a half colour overlay.
fn scene(depth: u32) -> (Session, u64) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48, "depth": depth})).unwrap();
    let id = s.execute("layer.new.layer", json!({"name": "a"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("select.rect", json!({"x": 14, "y": 10, "width": 56, "height": 20})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#e02010"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("layer.layerStyle.dropShadow", json!({"blend": "normal", "distance": 6, "size": 4, "color": "#102040"})).unwrap();
    s.execute("layer.layerStyle.outerGlow", json!({"blend": "normal", "size": 5, "color": "#20c040", "add": true})).unwrap();
    s.execute("layer.layerStyle.innerShadow", json!({"blend": "normal", "distance": 3, "size": 3, "add": true})).unwrap();
    s.execute("layer.layerStyle.stroke", json!({"size": 2, "position": "outside", "color": "#0000ff", "add": true})).unwrap();
    s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ffff00", "opacity": 50, "add": true})).unwrap();
    (s, id)
}

fn bake_and_check(s: &mut Session, id: u64, depth: u32) {
    let before = composite(s);
    let steps = s.active().unwrap().history.past_len();
    let r = s.execute(ID, json!({})).unwrap();
    assert_eq!(r["layers"], json!([id]));
    assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one history step");
    let l = layer(s, id);
    assert!(l.effects.items.is_empty(), "effects baked");
    assert!(matches!(l.content, LayerContent::Raster(_)));
    assert!(l.mask.is_none() && l.vector_mask.is_none(), "masks applied");
    assert_eq!(l.fill_opacity, 1.0);
    let after = composite(s);
    let d = max_diff(&before, &after);
    assert!(d <= tolerance(depth), "depth {depth}: composite changed by {d}");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(layer(s, id).effects.items.len(), 5, "undo restores the effects");
    assert_eq!(max_diff(&before, &composite(s)), 0.0, "undo restores the picture");
}

#[test]
fn bakes_effects_without_changing_the_composite() {
    for depth in [8, 16, 32] {
        let (mut s, id) = scene(depth);
        bake_and_check(&mut s, id, depth);
    }
}

#[test]
fn fill_opacity_masks_and_layer_opacity_survive_the_bake() {
    for depth in [8, 16, 32] {
        let (mut s, id) = scene(depth);
        s.execute("layer.setProps", json!({"layer": id, "fill": 0.4, "opacity": 0.7})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 34, "height": 48})).unwrap();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        let square = json!({"subpaths": [{"closed": true, "knots": [[4, 4], [60, 4], [60, 26], [4, 26]]}]});
        s.execute("layer.vectorMask.add", json!({"layer": id, "path": square})).unwrap();
        bake_and_check(&mut s, id, depth);
        s.execute("edit.redo", json!({})).unwrap();
        assert_eq!(layer(&s, id).opacity, 0.7, "the layer's own opacity stays a property");
    }
}

#[test]
fn keeps_the_pixels_past_the_canvas() {
    let (mut s, id) = scene(8);
    s.execute(ID, json!({"layer": id})).unwrap();
    let LayerContent::Raster(px) = &layer(&s, id).content else { panic!() };
    // The rectangle runs to x = 70 and its stroke and glow further, past the 64 px canvas.
    assert!(px.content_bounds().x1 > 70, "{:?}", px.content_bounds());
}

#[test]
fn rasterizes_fill_content_first() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    s.execute("layer.newFillLayer.solidColor", json!({"color": "#3366cc"})).unwrap();
    let id = s.active().unwrap().active_layer.unwrap().0;
    let square = json!({"subpaths": [{"closed": true, "knots": [[8, 8], [28, 8], [28, 20], [8, 20]]}]});
    s.execute("layer.vectorMask.add", json!({"layer": id, "path": square})).unwrap();
    s.execute("layer.layerStyle.stroke", json!({"size": 3, "position": "outside", "color": "#ff0000"})).unwrap();
    let before = composite(&s);
    let steps = s.active().unwrap().history.past_len();
    s.execute(ID, json!({})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "rasterize and bake are one step");
    assert!(matches!(layer(&s, id).content, LayerContent::Raster(_)));
    assert!(layer(&s, id).effects.items.is_empty());
    assert!(max_diff(&before, &composite(&s)) <= tolerance(8));
}

#[test]
fn rasterize_layer_turns_a_vector_mask_into_a_pixel_mask_and_keeps_effects() {
    let (mut s, id) = scene(8);
    // Effects alone: Rasterize › Layer has nothing to do (Rasterize › Layer Style bakes them).
    assert!(s.execute("layer.rasterize.layer", json!({})).is_err());
    let square = json!({"subpaths": [{"closed": true, "knots": [[4, 4], [40, 4], [40, 40], [4, 40]]}]});
    s.execute("layer.vectorMask.add", json!({"layer": id, "path": square})).unwrap();
    let before = composite(&s);
    s.execute("layer.rasterize.layer", json!({})).unwrap();
    let l = layer(&s, id);
    assert!(l.vector_mask.is_none() && l.mask.is_some(), "vector mask became a pixel mask");
    assert_eq!(l.effects.items.len(), 5, "effects stay live");
    assert!(max_diff(&before, &composite(&s)) <= tolerance(8));
}

#[test]
fn several_selected_layers_bake_in_one_step() {
    let (mut s, a) = scene(8);
    let b = s.execute("layer.new.layer", json!({"name": "b"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("select.rect", json!({"x": 2, "y": 34, "width": 10, "height": 8})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#00ff00"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("layer.layerStyle.stroke", json!({"size": 2})).unwrap();
    let plain = s.execute("layer.new.layer", json!({"name": "plain"})).unwrap()["layer"].as_u64().unwrap();
    s.active_mut().unwrap().selected_layers = vec![LayerId(a), LayerId(b), LayerId(plain)];
    let steps = s.active().unwrap().history.past_len();
    let r = s.execute(ID, json!({})).unwrap();
    assert_eq!(r["layers"], json!([a, b]));
    assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
    assert!(layer(&s, a).effects.items.is_empty() && layer(&s, b).effects.items.is_empty());
}

#[test]
fn disabled_and_bad_params_fail_gracefully() {
    let mut empty = Session::new();
    assert!(empty.execute(ID, json!({})).is_err(), "no document");
    let (mut s, id) = scene(8);
    let bg = doc(&s).layers[0].id.0;
    assert!(s.execute(ID, json!({"layer": bg})).is_err(), "the Background has no effects");
    assert!(s.execute(ID, json!({"layer": u64::MAX})).is_err(), "no such layer");
    assert!(s.execute(ID, json!({"layer": u64::MAX})).is_err(), "no such layer");
    assert_eq!(layer(&s, id).effects.items.len(), 5, "a bad call changes nothing");
    s.execute("layer.layerStyle.hideAllEffects", json!({})).unwrap();
    assert!(s.execute(ID, json!({"layer": id})).is_err(), "hidden effects: nothing to bake");
    s.execute("layer.layerStyle.showAllEffects", json!({})).unwrap();
    let g = s.execute("layer.new.group", json!({})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.layerStyle.stroke", json!({"layer": g})).unwrap();
    assert!(s.execute(ID, json!({"layer": g})).is_err(), "group styles are not baked");
}
