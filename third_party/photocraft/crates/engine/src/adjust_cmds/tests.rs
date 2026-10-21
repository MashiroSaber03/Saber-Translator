use super::*;

const DEPTHS: [u64; 3] = [8, 16, 32];
const MODES: [&str; 4] = ["rgb", "gray", "cmyk", "lab"];

fn session(depth: u64, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 48, "height": 32, "depth": depth, "mode": mode})).unwrap();
    paint(&mut s, |x, y| [0.1 + x as f32 / 60.0, 0.2 + y as f32 / 50.0, 0.6 - x as f32 / 120.0, 1.0]);
    s
}

fn paint(s: &mut Session, f: impl Fn(i32, i32) -> [f32; 4]) {
    s.edit("setup", |doc, active| {
        let b = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let mut data = Vec::new();
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                data.extend(photocraft_raster::from_rgba(&fmt, f(x, y)));
            }
        }
        surf.write_region(b, &data);
        Ok(())
    })
    .unwrap();
}

fn rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
}

fn changed(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

#[test]
fn destructive_adjustments_all_depths_and_modes_one_undo_step() {
    let cases: [(&str, Value); 4] = [
        ("image.adjustments.shadowsHighlights", json!({"shadowAmount": 80, "blackClip": 0, "whiteClip": 0})),
        ("image.adjustments.replaceColor", json!({"color": "#4d66cc", "fuzziness": 200, "hue": 90, "saturation": 30})),
        ("image.adjustments.matchColor", json!({"neutralize": true, "intensity": 50})),
        ("image.adjustments.hdrToning", json!({"strength": 2, "exposure": 0.5})),
    ];
    for (id, params) in &cases {
        for depth in DEPTHS {
            for mode in MODES {
                let mut s = session(depth, mode);
                let before = rgba(&s, 3, 3);
                let steps = s.active().unwrap().history.past_len();
                s.execute(id, params.clone()).unwrap_or_else(|e| panic!("{id} {depth} {mode}: {e}"));
                let after = rgba(&s, 3, 3);
                // Gray documents can't show hue changes from Replace Color at a gray pixel.
                if !(mode == "gray" && (id.ends_with("replaceColor") || id.ends_with("matchColor"))) {
                    assert!(changed(before, after) > 1e-3, "{id} {depth} {mode}: {before:?} → {after:?}");
                }
                assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "{id}: one history step");
                s.execute("edit.undo", json!({})).unwrap();
                assert!(changed(before, rgba(&s, 3, 3)) < 1e-6, "{id} undo");
            }
        }
    }
}

#[test]
fn destructive_adjustments_respect_selection() {
    let mut s = session(8, "rgb");
    s.execute("select.all", json!({})).ok();
    s.edit("sel", |doc, _| {
        let mut sel = Surface::new(photocraft_color::PixelFormat::GRAY8);
        sel.write_region(Rect::new(0, 0, 10, 32), &vec![1.0; 320]);
        doc.selection = Some(sel);
        Ok(())
    })
    .unwrap();
    let outside = rgba(&s, 30, 5);
    let inside = rgba(&s, 5, 5);
    s.execute("image.adjustments.shadowsHighlights", json!({"shadowAmount": 100, "blackClip": 0, "whiteClip": 0})).unwrap();
    assert!(changed(inside, rgba(&s, 5, 5)) > 1e-3);
    assert_eq!(outside, rgba(&s, 30, 5));
}

#[test]
fn match_color_from_another_document() {
    let mut s = session(8, "rgb");
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "#e05010"})).unwrap();
    // The target is document 0.
    s.execute("document.activate", json!({"document": 0})).unwrap();
    let before = rgba(&s, 10, 10);
    s.execute("image.adjustments.matchColor", json!({"source": 1})).unwrap();
    let after = rgba(&s, 10, 10);
    assert!(after[0] > before[0] && after[2] < before[2], "{before:?} → {after:?}");
    // Fade 100 is a no-op; a bad source index is an error.
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("image.adjustments.matchColor", json!({"source": 1, "fade": 100})).unwrap();
    assert!(changed(before, rgba(&s, 10, 10)) < 2.0 / 255.0);
    assert!(s.execute("image.adjustments.matchColor", json!({"source": 7})).is_err());
    // Source "None" with default options is a no-op.
    let now = rgba(&s, 10, 10);
    s.execute("image.adjustments.matchColor", json!({})).unwrap();
    assert!(changed(now, rgba(&s, 10, 10)) < 2.0 / 255.0);
}

#[test]
fn hdr_toning_flattens() {
    let mut s = session(16, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    assert_eq!(doc(&s).layers.len(), 2);
    s.execute("image.adjustments.hdrToning", json!({})).unwrap();
    assert_eq!(doc(&s).layers.len(), 1);
    assert_eq!(doc(&s).layers[0].name, "Background");
}

#[test]
fn disabled_without_pixels() {
    let mut s = session(8, "rgb");
    s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    for id in ["image.adjustments.shadowsHighlights", "image.adjustments.replaceColor", "image.adjustments.matchColor"] {
        assert!(!s.is_enabled(id), "{id}");
    }
    assert!(s.is_enabled("image.adjustments.hdrToning"));
    assert!(!Session::new().is_enabled("image.adjustments.hdrToning"));
}

#[test]
fn selective_color_layer_and_destructive() {
    for depth in DEPTHS {
        for mode in ["rgb", "cmyk", "lab"] {
            let mut s = session(depth, mode);
            paint(&mut s, |_, _| [0.9, 0.1, 0.1, 1.0]);
            let before = rgba(&s, 1, 1);
            // Reds: −100 % cyan (absolute) makes red redder; +100 % black darkens.
            s.execute("image.adjustments.selectiveColor", json!({"method": "absolute", "reds": [0, 0, 0, 60]})).unwrap();
            let after = rgba(&s, 1, 1);
            assert!(after[0] < before[0] - 0.05, "{depth} {mode}: {before:?} → {after:?}");
        }
    }
    let mut s = session(8, "rgb");
    let r = s.execute("layer.newAdjustmentLayer.selectiveColor", json!({"colors": "blues", "yellow": 40})).unwrap();
    let id = photocraft_doc::LayerId(r["layer"].as_u64().unwrap());
    let get = |s: &Session| match &doc(s).layer(id).unwrap().content {
        LayerContent::Adjustment(Adjustment::SelectiveColor { relative, adjustments }) => (*relative, *adjustments),
        other => panic!("{other:?}"),
    };
    assert_eq!(get(&s).1[4], [0.0, 0.0, 40.0, 0.0]);
    // Properties edits merge one range at a time.
    s.execute("layer.setAdjustment", json!({"layer": id.0, "colors": "neutrals", "black": -20, "method": "absolute"})).unwrap();
    let (rel, a) = get(&s);
    assert!(!rel);
    assert_eq!(a[4][2], 40.0);
    assert_eq!(a[7][3], -20.0);
    // No parameters: reset.
    s.execute("layer.setAdjustment", json!({"layer": id.0})).unwrap();
    assert_eq!(get(&s), (true, [[0.0; 4]; 9]));
}

#[test]
fn color_lookup_builtin_data_and_errors() {
    let mut s = session(8, "rgb");
    let r = s.execute("layer.newAdjustmentLayer.colorLookup", json!({"lut": "dayForNight"})).unwrap();
    let id = photocraft_doc::LayerId(r["layer"].as_u64().unwrap());
    let px = photocraft_compose::render(doc(&s), Rect::from_xywh(40, 30, 1, 1)).px[0];
    assert!(px[2] > px[0], "night is blue: {px:?}");
    // Switching interpolation keeps the table.
    s.execute("layer.setAdjustment", json!({"layer": id.0, "interpolation": "tetrahedral", "dither": true})).unwrap();
    match &doc(&s).layer(id).unwrap().content {
        LayerContent::Adjustment(Adjustment::ColorLookup { lut: Some(_), size: 33, tetrahedral: true, dither: true, name }) => {
            assert_eq!(name, "Day for Night")
        }
        other => panic!("{other:?}"),
    }
    // Embedded .cube text (an inverting LUT) applied destructively, at every depth.
    let mut cube = String::from("LUT_3D_SIZE 2\n");
    for b in 0..2 {
        for g in 0..2 {
            for r in 0..2 {
                cube.push_str(&format!("{} {} {}\n", 1 - r, 1 - g, 1 - b));
            }
        }
    }
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        let before = rgba(&s, 5, 5);
        s.execute("image.adjustments.colorLookup", json!({"data": cube, "fileName": "invert.cube"})).unwrap();
        let after = rgba(&s, 5, 5);
        for c in 0..3 {
            assert!((after[c] - (1.0 - before[c])).abs() < 2.0 / 255.0, "{depth}: {before:?} {after:?}");
        }
    }
    assert!(s.execute("image.adjustments.colorLookup", json!({"lut": "nope"})).is_err());
    // A custom input domain is rejected with the reason instead of silently shifting the look.
    // (A fresh pixel layer: this `s` still has the adjustment layer active, and the command is
    // disabled there - the domain rejection itself needs a pixel layer to reach.)
    s.execute("layer.new.layer", json!({})).unwrap();
    let domained = "TITLE \"d\"
DOMAIN_MIN 0.1 0.1 0.1
DOMAIN_MAX 0.9 0.9 0.9
LUT_3D_SIZE 2
0 0 0
0 0 1
0 1 0
0 1 1
1 0 0
1 0 1
1 1 0
1 1 1
";
    let e = s.execute("image.adjustments.colorLookup", json!({"data": domained, "fileName": "log.cube"})).unwrap_err();
    assert!(e.to_string().contains("DOMAIN_MIN"), "{e}");
    assert!(s.execute("image.adjustments.colorLookup", json!({"data": "LUT_3D_SIZE 3\n0 0 0\n"})).is_err());
    assert!(s.execute("layer.newAdjustmentLayer.colorLookup", json!({"file": "/nonexistent/x.cube"})).is_err());
    let looks = s.execute("image.adjustments.colorLookup.list", json!({})).unwrap();
    assert_eq!(looks.as_array().unwrap().len(), photocraft_cms::lutfile::BUILTIN.len());
}

/// The kind's parameters, read back from the model (Color Lookup with its table name and size).
fn adjustment_of(s: &Session, id: photocraft_doc::LayerId) -> Adjustment {
    match &doc(s).layer(id).unwrap().content {
        LayerContent::Adjustment(a) => a.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn selective_color_and_color_lookup_reject_wrong_typed_params() {
    let cube = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
    // (kind, params, the key the error must name)
    let cases: Vec<(&str, Value, &str)> = vec![
        ("selectiveColor", json!({"method": 5}), "method"),
        ("selectiveColor", json!({"method": "sideways"}), "method"),
        ("selectiveColor", json!({"relative": "yes"}), "relative"),
        ("selectiveColor", json!({"reds": "not-an-array"}), "reds"),
        ("selectiveColor", json!({"blues": [10, 20]}), "blues"),
        ("selectiveColor", json!({"neutrals": [10, "20", 0, 0]}), "neutrals"),
        ("selectiveColor", json!({"colors": 3, "cyan": 50}), "colors"),
        ("selectiveColor", json!({"colors": "purples", "cyan": 50}), "purples"),
        ("selectiveColor", json!({"colors": "reds", "cyan": "50"}), "cyan"),
        ("selectiveColor", json!({"colors": "blacks", "black": [1]}), "black"),
        ("colorLookup", json!({"lut": 42}), "lut"),
        ("colorLookup", json!({"interpolation": 5}), "interpolation"),
        ("colorLookup", json!({"interpolation": "bicubic"}), "interpolation"),
        ("colorLookup", json!({"tetrahedral": 1}), "tetrahedral"),
        ("colorLookup", json!({"dither": "yes"}), "dither"),
        ("colorLookup", json!({"data": 42}), "data"),
        ("colorLookup", json!({"data": cube, "fileName": 7}), "fileName"),
        ("colorLookup", json!({"file": 42}), "file"),
    ];
    let named = |r: Result<Value>, key: &str, what: &str| match r {
        Err(e @ EngineError::BadParams { .. }) => assert!(e.to_string().contains(key), "{what}: {e}"),
        other => panic!("{what}: expected BadParams naming `{key}`, got {other:?}"),
    };
    for depth in DEPTHS {
        for mode in ["rgb", "cmyk", "lab"] {
            for (kind, p, key) in &cases {
                let what = format!("{kind} {p} {depth} {mode}");
                let mut s = session(depth, mode);
                let (layers, steps, px) = (doc(&s).layers.len(), s.active().unwrap().history.past_len(), rgba(&s, 3, 3));
                named(s.execute(&format!("image.adjustments.{kind}"), p.clone()), key, &what);
                named(s.execute(&format!("layer.newAdjustmentLayer.{kind}"), p.clone()), key, &what);
                assert_eq!(doc(&s).layers.len(), layers, "{what}: no layer added");
                assert_eq!(s.active().unwrap().history.past_len(), steps, "{what}: no history step");
                assert_eq!(rgba(&s, 3, 3), px, "{what}: pixels unchanged");
            }
        }
    }
    // An update with a wrong-typed key fails and keeps every existing value.
    let mut s = session(8, "rgb");
    let made = [
        ("selectiveColor", json!({"method": "absolute", "reds": [10, -20, 30, -40], "colors": "blues", "yellow": 40})),
        ("colorLookup", json!({"lut": "warm", "interpolation": "tetrahedral", "dither": true})),
    ];
    for (kind, p) in made {
        let id = photocraft_doc::LayerId(s.execute(&format!("layer.newAdjustmentLayer.{kind}"), p).unwrap()["layer"].as_u64().unwrap());
        let before = adjustment_of(&s, id);
        for (k, p, key) in cases.iter().filter(|c| c.0 == kind) {
            let mut p = p.clone();
            p["layer"] = json!(id.0);
            named(s.execute("layer.setAdjustment", p.clone()), key, &format!("setAdjustment {k} {p}"));
            assert_eq!(adjustment_of(&s, id), before, "{p}");
        }
        // Null still means "not given": the update keeps everything.
        let nulls = json!({"layer": id.0, "method": null, "reds": null, "colors": null, "lut": null, "interpolation": null, "dither": null});
        s.execute("layer.setAdjustment", nulls).unwrap();
        assert_eq!(adjustment_of(&s, id), before, "{kind}: nulls");
    }
    // Well-typed values reach the model.
    match adjustment_of(&s, doc(&s).layers[1].id) {
        Adjustment::SelectiveColor { relative: false, adjustments } => {
            assert_eq!(adjustments[0], [10.0, -20.0, 30.0, -40.0]);
            assert_eq!(adjustments[4], [0.0, 0.0, 40.0, 0.0]);
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(adjustment_of(&s, doc(&s).layers[2].id), Adjustment::ColorLookup { lut: Some(_), tetrahedral: true, dither: true, .. }));
}

#[test]
fn new_kinds_round_trip_through_psd_and_pcraft() {
    let mut s = session(8, "rgb");
    s.execute("layer.newAdjustmentLayer.selectiveColor", json!({"reds": [10, -20, 30, -40], "method": "absolute"})).unwrap();
    s.execute("layer.newAdjustmentLayer.colorLookup", json!({"lut": "warm", "dither": true})).unwrap();
    let d = doc(&s).clone();
    let bytes = photocraft_io::export(&d, "x.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("x.psd", &bytes).unwrap().document;
    let kinds: Vec<&str> = back
        .layers
        .iter()
        .filter_map(|l| match &l.content {
            LayerContent::Adjustment(a) => Some(crate::commands::adjustment_kind(a)),
            _ => None,
        })
        .collect();
    assert_eq!(kinds, ["selectiveColor", "colorLookup"]);
    let pc = photocraft_format::save_to_bytes(&d, &Default::default()).unwrap();
    let again = photocraft_format::load_from_bytes(&pc).unwrap();
    for (a, b) in d.layers.iter().zip(&again.layers) {
        assert_eq!(a.content, b.content);
    }
}

fn mask_at(s: &Session, x: i32, y: i32) -> f32 {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0)
}

/// #2456: Edit > Fill (and the fill keys) fill the targeted mask of an adjustment layer.
#[test]
fn fill_works_on_an_adjustment_layers_mask() {
    let mut s = session(8, "rgb");
    s.execute("layer.newAdjustmentLayer.curves", json!({})).unwrap();
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    let mask = json!({"target": "mask"});
    assert!(!s.is_enabled("edit.fill"), "no pixels to fill without the mask target");
    assert!(s.is_enabled_with("edit.fill", &mask));
    assert!(s.is_enabled_with("edit.fillForeground", &mask));
    s.execute("edit.fill", json!({"contents": "black", "target": "mask"})).unwrap();
    assert_eq!(mask_at(&s, 3, 3), 0.0, "the mask is filled black");
    s.execute("edit.fill", json!({"contents": "white", "target": "mask"})).unwrap();
    assert_eq!(mask_at(&s, 3, 3), 1.0);
}

/// #780: with the layer mask targeted, Invert (⌘I) inverts the mask, past the canvas too, and
/// leaves the layer's pixels alone; one history step.
#[test]
fn invert_with_the_mask_targeted_inverts_the_mask() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        s.execute("select.rect", json!({"x": 24, "y": 0, "width": 24, "height": 32})).unwrap();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        let (pixels, off_canvas) = (rgba(&s, 3, 3), mask_at(&s, -500, 3));
        let steps = s.active().unwrap().history.past_len();
        s.execute("image.adjustments.invert", json!({"target": "mask"})).unwrap();
        assert_eq!((mask_at(&s, 3, 3), mask_at(&s, 30, 3)), (1.0, 0.0), "depth {depth}");
        assert_eq!(mask_at(&s, -500, 3), 1.0 - off_canvas, "the mask's untouched area inverts too");
        assert_eq!(rgba(&s, 3, 3), pixels, "the layer is untouched");
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
        // Through a selection, only the selected part.
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 10, "height": 32})).unwrap();
        s.execute("image.adjustments.invert", json!({"target": "mask"})).unwrap();
        assert_eq!((mask_at(&s, 3, 3), mask_at(&s, 12, 3), mask_at(&s, 30, 3)), (0.0, 1.0, 0.0));
        for _ in 0..3 {
            s.undo(); // invert, select, invert
        }
        assert_eq!((mask_at(&s, 3, 3), mask_at(&s, 30, 3), mask_at(&s, -500, 3)), (0.0, 1.0, off_canvas));
    }
}

/// Photoshop: a mask made from the selection (Reveal Selection, the Layers panel's mask button)
/// takes the selection with it, as Hide Selection does. So ⌘I on the new mask right away inverts
/// all of it; with the selection kept it inverted only the inside, leaving the mask all black.
/// Undo brings the selection back with the mask gone, in one step.
#[test]
fn reveal_selection_drops_the_selection_so_invert_flips_the_whole_mask() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        s.execute("select.rect", json!({"x": 24, "y": 0, "width": 24, "height": 32})).unwrap();
        let selection = s.active().unwrap().doc.selection.clone();
        let steps = s.active().unwrap().history.past_len();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        assert!(s.active().unwrap().doc.selection.is_none(), "depth {depth}: the selection became the mask");
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one step");
        assert_eq!((mask_at(&s, 3, 3), mask_at(&s, 30, 3)), (0.0, 1.0));
        s.execute("image.adjustments.invert", json!({"target": "mask"})).unwrap();
        assert_eq!((mask_at(&s, 3, 3), mask_at(&s, 30, 3)), (1.0, 0.0), "depth {depth}: inside and outside both flip");
        s.undo();
        s.undo();
        let st = s.active().unwrap();
        assert_eq!(st.doc.selection, selection, "undo restores the selection");
        assert!(st.doc.layer(st.active_layer.unwrap()).unwrap().mask.is_none(), "and removes the mask");
    }
}

// ---------- Image › Adjustments as smart filters (FILE-215-4) ----------

/// A painted pixel layer converted to a smart object, over the painted Background.
fn smart_session() -> Session {
    let mut s = session(8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s, |x, y| [0.1 + x as f32 / 60.0, 0.2 + y as f32 / 50.0, 0.6 - x as f32 / 120.0, 1.0]);
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    s
}

fn flatten(s: &Session) -> Vec<[f32; 4]> {
    photocraft_compose::flatten(doc(s)).px
}

fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter().zip(b).flat_map(|(p, q)| (0..4).map(move |c| (p[c] - q[c]).abs())).fold(0.0, f32::max)
}

fn smart_filters_of(s: &Session) -> Vec<photocraft_doc::SmartFilter> {
    let d = s.active().unwrap();
    match &d.doc.layer(d.active_layer.unwrap()).unwrap().content {
        LayerContent::Smart(sm) => sm.smart_filters.clone(),
        other => panic!("not a smart object: {}", other.kind_name()),
    }
}

/// FILE-215-4: Image › Adjustments on a smart object record a smart filter (Photoshop) instead
/// of baking into a copy of the pixels: it renders exactly what the destructive command would
/// bake in, and can be hidden, re-edited and undone, each in one history step.
#[test]
fn adjustments_on_a_smart_object_record_a_smart_filter() {
    let mut s = smart_session();
    let original = flatten(&s);
    // The destructive result on the same pixels: the smart filter must match it exactly.
    let mut plain = session(8, "rgb");
    plain.execute("image.adjustments.invert", json!({})).unwrap();
    let baked = flatten(&plain);
    assert!(max_diff(&original, &baked) > 0.5, "the control changes the pixels");
    assert!(s.is_enabled("image.adjustments.invert"), "an adjustment runs on a smart object");
    let steps = s.active().unwrap().history.past_len();
    s.execute("image.adjustments.invert", json!({})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one history step");
    let sf = smart_filters_of(&s);
    assert_eq!(sf.len(), 1);
    assert_eq!(sf[0].command, "image.adjustments.invert");
    assert!(sf[0].visible && sf[0].opacity == 1.0 && sf[0].blend == photocraft_color::BlendMode::Normal);
    assert!(sf[0].params.get("__kind").is_none(), "the engine's private key is not recorded");
    assert_eq!(max_diff(&flatten(&s), &baked), 0.0, "the smart filter renders the destructive result");
    // Hiding the filter shows the original exactly.
    s.execute("layer.smartFilter.setVisible", json!({"index": 0})).unwrap();
    assert_eq!(flatten(&s), original);
    s.undo();
    assert_eq!(max_diff(&flatten(&s), &baked), 0.0, "undo restores the filtered pixels");
    // Re-editing the recorded params re-renders; the same params give the same pixels.
    s.execute("layer.smartFilter.setParams", json!({"index": 0, "params": {}})).unwrap();
    assert_eq!(max_diff(&flatten(&s), &baked), 0.0, "the same params render the same pixels");
    // The setParams step, then the filter itself, undo away.
    s.undo();
    s.undo();
    assert!(smart_filters_of(&s).is_empty(), "the filter is gone");
    assert_eq!(flatten(&s), original);
}

/// A selection becomes the smart filter's mask, and the adjustment with its own command
/// (Shadows/Highlights) records a filter too.
#[test]
fn adjustment_smart_filters_take_a_selection_as_their_mask() {
    let mut s = smart_session();
    let original = flatten(&s);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 24, "height": 32})).unwrap();
    s.execute("image.adjustments.shadowsHighlights", json!({"shadowAmount": 100, "blackClip": 0, "whiteClip": 0})).unwrap();
    let d = s.active().unwrap();
    let LayerContent::Smart(sm) = &d.doc.layer(d.active_layer.unwrap()).unwrap().content else { panic!("not smart") };
    assert_eq!(sm.smart_filters.len(), 1);
    assert_eq!(sm.smart_filters[0].command, "image.adjustments.shadowsHighlights");
    assert!(sm.filter_mask.is_some(), "the selection became the filter mask");
    let now = flatten(&s);
    let at = |px: &[[f32; 4]], x: i32, y: i32| px[(y * 48 + x) as usize];
    assert!(changed(at(&original, 5, 16), at(&now, 5, 16)) > 1e-3, "inside the selection the shadows lifted");
    assert_eq!(at(&now, 40, 16), at(&original, 40, 16), "outside the mask the composite is unchanged");
}

/// A wrong-typed value errors before anything is recorded (never a panic), and the adjustments
/// that can't re-render from a source (Replace Color, Match Color) stay off a smart object.
#[test]
fn a_bad_adjustment_param_on_a_smart_object_is_an_error() {
    let mut s = smart_session();
    let (filters, steps, flat) = (smart_filters_of(&s).len(), s.active().unwrap().history.past_len(), flatten(&s));
    let e = s.execute("image.adjustments.colorLookup", json!({"lut": 42})).unwrap_err();
    assert!(e.to_string().contains("lut"), "{e}");
    assert_eq!(smart_filters_of(&s).len(), filters, "nothing recorded");
    assert_eq!(s.active().unwrap().history.past_len(), steps, "no history step");
    assert_eq!(flatten(&s), flat, "pixels untouched");
    assert!(!s.is_enabled("image.adjustments.replaceColor"), "only adjustments that re-render become smart filters");
    assert!(!s.is_enabled("image.adjustments.matchColor"), "only adjustments that re-render become smart filters");
}

/// #780: a targeted mask enables Invert (and filters) on layers without pixels, such as an
/// adjustment layer; viewing the mask (⌥-click) targets it too.
#[test]
fn a_targeted_mask_enables_editing_it_on_any_layer() {
    let mut s = session(8, "rgb");
    s.execute("layer.newAdjustmentLayer.levels", json!({})).unwrap();
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    let mask = json!({"target": "mask"});
    assert!(!s.is_enabled("image.adjustments.invert"), "no pixels to invert");
    assert!(s.is_enabled_with("image.adjustments.invert", &mask));
    assert!(s.is_enabled_with("filter.blur.gaussianBlur", &mask));
    assert!(!s.is_enabled_with("image.adjustments.desaturate", &mask), "only what edits the mask");
    s.execute("image.adjustments.invert", mask.clone()).unwrap();
    assert_eq!((mask_at(&s, 3, 3), mask_at(&s, -500, -500)), (0.0, 0.0), "reveal all → hide all");
    s.execute("view.layerMask", json!({"mode": "gray"})).unwrap();
    assert!(s.is_enabled("image.adjustments.invert"), "the mask view targets the mask");
    s.execute("image.adjustments.invert", json!({})).unwrap();
    assert_eq!(mask_at(&s, 3, 3), 1.0);
    // A filter on a pixel layer's targeted mask leaves the pixels alone.
    s.execute("layer.delete", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 24, "y": 0, "width": 24, "height": 32})).unwrap();
    s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
    let pixels = rgba(&s, 23, 3);
    s.execute("filter.blur.gaussianBlur", json!({"radius": 3, "target": "mask"})).unwrap();
    assert!(mask_at(&s, 23, 3) > 0.0 && mask_at(&s, 23, 3) < 1.0, "the mask's edge is blurred");
    assert_eq!(rgba(&s, 23, 3), pixels);
    // No mask: an error, not the layer's pixels.
    s.execute("layer.layerMask.delete", json!({})).unwrap();
    assert!(!s.is_enabled_with("image.adjustments.invert", &mask));
    assert!(s.execute("image.adjustments.invert", mask).is_err());
    assert_eq!(rgba(&s, 23, 3), pixels);
}

#[test]
fn replace_color_localized_clusters_only_change_the_colour_near_the_sample() {
    // The same cream on the left (x < 8) and the right (x ≥ 40), grey between.
    let cream = [0.9, 0.88, 0.79, 1.0];
    let mut s = session(8, "rgb");
    paint(&mut s, |x, _| if !(8..40).contains(&x) { cream } else { [0.5, 0.5, 0.5, 1.0] });
    let base = |s: &Session| (rgba(s, 3, 16), rgba(s, 44, 16), rgba(s, 20, 16));
    let (left, right, grey) = base(&s);
    let shift = json!({"color": "#e6e0c9", "fuzziness": 20, "hue": 180, "saturation": 60});
    // Without clusters the colour is replaced everywhere (unchanged behaviour).
    s.execute("image.adjustments.replaceColor", shift.clone()).unwrap();
    assert!(changed(rgba(&s, 3, 16), left) > 0.05 && changed(rgba(&s, 44, 16), right) > 0.05);
    s.execute("edit.undo", json!({})).unwrap();
    // Picked on the left with Range 50 % (24 px): only the left cream changes.
    let mut local = shift.clone();
    local["points"] = json!([[2, 16]]);
    local["localized"] = json!(true);
    local["range"] = json!(50);
    s.execute("image.adjustments.replaceColor", local.clone()).unwrap();
    assert!(changed(rgba(&s, 3, 16), left) > 0.05, "near the sample");
    assert_eq!(rgba(&s, 44, 16), right, "the same colour far from the sample");
    assert_eq!(rgba(&s, 20, 16), grey);
    s.execute("edit.undo", json!({})).unwrap();
    // Inside a selection that doesn't start at the origin the sample stays where it was picked.
    s.execute("select.rect", json!({"x": 1, "y": 4, "width": 46, "height": 24})).unwrap();
    s.execute("image.adjustments.replaceColor", local).unwrap();
    assert!(changed(rgba(&s, 3, 16), left) > 0.05);
    assert_eq!(rgba(&s, 44, 16), right);
    // Clusters need a place to be near.
    let err = s.execute("image.adjustments.replaceColor", json!({"localized": true})).unwrap_err();
    assert!(err.to_string().contains("points"), "{err}");
    let err = s.execute("image.adjustments.replaceColor", json!({"points": [[1]]})).unwrap_err();
    assert!(err.to_string().contains("bad point"), "{err}");
}
