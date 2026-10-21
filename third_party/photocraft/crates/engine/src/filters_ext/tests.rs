use super::*;
use photocraft_geom::Rect;

fn session() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 48, "height": 32})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("pattern", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        for y in 0..32 {
            for x in 0..48 {
                let v = ((x * 5 + y * 3) % 17) as f32 / 16.0;
                surf.write_pixel(x, y, &[v, 1.0 - v, (x % 3) as f32 / 2.0, 1.0]);
            }
        }
        Ok(())
    })
    .unwrap();
    s
}

fn pixels(s: &Session) -> Vec<f32> {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 48, 32))
}

const IDS: [&str; 34] = [
    "filter.pixelate.colorHalftone",
    "filter.pixelate.crystallize",
    "filter.pixelate.facet",
    "filter.pixelate.fragment",
    "filter.pixelate.mezzotint",
    "filter.pixelate.pointillize",
    "filter.stylize.diffuse",
    "filter.stylize.extrude",
    "filter.stylize.oilPaint",
    "filter.stylize.tiles",
    "filter.stylize.traceContour",
    "filter.stylize.wind",
    "filter.distort.shear",
    "filter.distort.zigZag",
    "filter.render.fibers",
    "filter.render.lensFlare",
    "filter.render.lightingEffects",
    "filter.render.relight",
    "filter.noise.reduceNoise",
    "filter.blur.smartBlur",
    "filter.blur.lensBlur",
    "filter.blur.shapeBlur",
    "filter.blurGallery.tiltShift",
    "filter.blurGallery.irisBlur",
    "filter.blurGallery.fieldBlur",
    "filter.blurGallery.spinBlur",
    "filter.blurGallery.pathBlur",
    "filter.other.custom",
    "filter.other.hsbHsl",
    "filter.video.deInterlace",
    "filter.video.ntscColors",
    "filter.stylize.wind",
    "filter.distort.zigZag",
    "filter.blur.smartBlur",
];

fn test_params(id: &str) -> Value {
    match id {
        "filter.distort.shear" => json!({"amount": 40}),
        "filter.extrude" => json!({}),
        "filter.stylize.extrude" => json!({"size": 6, "depth": 40}),
        "filter.stylize.tiles" => json!({"count": 4, "maxOffset": 30}),
        "filter.blur.shapeBlur" => json!({"radius": 5, "shape": "star"}),
        "filter.blur.lensBlur" => json!({"radius": 4}),
        "filter.blurGallery.tiltShift" => json!({"blur": 6}),
        "filter.blurGallery.irisBlur" => json!({"blur": 6}),
        "filter.blurGallery.fieldBlur" => json!({"blur": 4}),
        "filter.blurGallery.pathBlur" => json!({"speed": 8}),
        "filter.other.custom" => json!({"kernel": [0,0,0,0,0, 0,0,-1,0,0, 0,-1,5,-1,0, 0,0,-1,0,0, 0,0,0,0,0]}),
        "filter.blur.smartBlur" => json!({"threshold": 60}),
        _ => json!({}),
    }
}

#[test]
fn ids_match_the_menu_catalog_and_are_unique() {
    let all = crate::command_specs();
    for id in IDS {
        assert!(crate::commands::find(id).is_some(), "{id} not registered");
        assert_eq!(all.iter().filter(|c| c.id == id).count(), 1, "{id} registered twice");
        assert!(params_for(id, &Value::Null).is_some(), "{id} has no params mapping");
    }
    assert!(crate::commands::find("filter.distort.displace").is_some());
}

#[test]
fn every_new_filter_runs_changes_pixels_and_undoes_in_one_step() {
    for id in IDS {
        let mut s = session();
        let before = pixels(&s);
        let depth = s.active().unwrap().history.past_len();
        let r = s.execute(id, test_params(id)).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(r.get("filter").is_some(), "{id}");
        assert_ne!(pixels(&s), before, "{id} changed nothing");
        assert_eq!(s.active().unwrap().history.past_len(), depth + 1, "{id} is one history step");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(pixels(&s), before, "{id} undo");
    }
}

#[test]
fn new_filters_work_at_16_and_32_bit() {
    for (depth, mode) in [(16, "rgb"), (32, "rgb"), (8, "cmyk"), (16, "lab"), (8, "gray")] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 24, "height": 16, "depth": depth, "mode": mode})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#3366cc"})).ok();
        for id in IDS {
            s.execute(id, test_params(id)).unwrap_or_else(|e| panic!("{id} at {depth}-bit {mode}: {e}"));
        }
    }
}

#[test]
fn disabled_without_pixel_layer() {
    let mut s = Session::new();
    for id in IDS {
        assert!(!s.is_enabled(id), "{id}");
    }
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    assert!(!s.is_enabled("filter.stylize.oilPaint"));
}

#[test]
fn selection_limits_new_filters() {
    let mut s = session();
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 12, "height": 12})).unwrap();
    let before = pixels(&s);
    s.execute("filter.pixelate.crystallize", json!({"cellSize": 4})).unwrap();
    let after = pixels(&s);
    let far = (28 * 48 + 40) * 4;
    assert_eq!(after[far..far + 4], before[far..far + 4]);
    assert_ne!(after[..12 * 4], before[..12 * 4]);
}

#[test]
fn last_filter_repeats_new_filters() {
    let mut s = session();
    s.execute("filter.stylize.diffuse", json!({"seed": 3})).unwrap();
    let once = pixels(&s);
    assert!(s.is_enabled("filter.lastFilter"));
    s.execute("filter.lastFilter", json!({})).unwrap();
    assert_ne!(pixels(&s), once);
}

#[test]
fn colour_filters_use_and_record_tool_colours() {
    let mut s = session();
    s.tools.foreground = [1.0, 0.0, 0.0, 1.0];
    s.tools.background = [0.0, 0.0, 1.0, 1.0];
    let r = s.execute("filter.render.fibers", json!({"seed": 1})).unwrap();
    assert_eq!(r["filter"]["foreground"], json!([1.0, 0.0, 0.0, 1.0]));
    let px = pixels(&s);
    for p in px.as_chunks::<4>().0 {
        assert!(p[1] < 1e-2 && (p[0] + p[2] - 1.0).abs() < 2e-2, "{p:?}");
    }
}

#[test]
fn fibers_fill_an_empty_layer() {
    // #5241: Fibers renders from the colours, not the layer's pixels, so an empty layer fills
    // like in Photoshop (and can be transformed afterwards).
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 48, "height": 32})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("filter.render.fibers", json!({"seed": 3})).unwrap();
    let px = pixels(&s);
    assert!(px.as_chunks::<4>().0.iter().all(|p| p[3] == 1.0), "every pixel is rendered opaque");
    s.execute("edit.transform.flipHorizontal", json!({})).unwrap();

    // With a selection only the selection fills.
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 6})).unwrap();
    s.execute("filter.render.fibers", json!({"seed": 3})).unwrap();
    let d = s.active().unwrap();
    let surf = d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap();
    assert_eq!(surf.content_bounds(), Rect::new(4, 4, 14, 10));
}

#[test]
fn displace_reads_another_document() {
    let mut s = session();
    // Map document: a flat colour that shifts 3 px right at 25 % (see the algo test).
    s.execute("file.new", json!({"width": 8, "height": 8, "depth": 32})).unwrap();
    let v = 0.5 + 3.0 / 64.0;
    s.edit("map", |doc, _| {
        let bg = doc.layers[0].id;
        doc.layer_mut(bg).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 8, 8), &[v, 0.5, 0.5, 1.0]);
        Ok(())
    })
    .unwrap();
    s.set_active(0);
    let before = pixels(&s);
    assert!(s.execute("filter.distort.displace", json!({})).is_err(), "needs a map");
    s.execute("filter.distort.displace", json!({"mapDocument": 1, "horizontal": 25, "vertical": 0, "fit": "tile", "undefinedAreas": "wrap"})).unwrap();
    let after = pixels(&s);
    let at = |v: &[f32], x: usize, y: usize| v[(y * 48 + x) * 4];
    for x in 5..40 {
        assert!((at(&after, x, 10) - at(&before, x + 3, 10)).abs() < 1e-2, "x={x}");
    }
}

#[test]
fn lens_blur_uses_the_layer_mask_as_depth() {
    let mut s = session();
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    let before = pixels(&s);
    // Reveal-all mask = depth 1 everywhere; focal distance 255 puts it all in focus.
    s.execute("filter.blur.lensBlur", json!({"radius": 6, "depthMap": "layerMask", "focalDistance": 255})).unwrap();
    assert_eq!(pixels(&s), before);
    s.execute("filter.blur.lensBlur", json!({"radius": 6, "depthMap": "layerMask", "focalDistance": 0})).unwrap();
    assert_ne!(pixels(&s), before);
}

#[test]
fn params_map_to_algorithm_units() {
    assert_eq!(params_for("filter.pixelate.crystallize", &json!({"cellSize": 1})), Some(FilterParams::Crystallize { cell_size: 3.0, seed: 0 }));
    assert_eq!(
        params_for("filter.stylize.traceContour", &json!({"level": 300, "edge": "upper"})),
        Some(FilterParams::TraceContour { level: 255.0, upper: true })
    );
    let Some(FilterParams::IrisBlur { pins }) = params_for("filter.blurGallery.irisBlur", &json!({"pins": [{"x": 0.2, "blur": 30}, {"x": 0.8}]})) else {
        panic!()
    };
    assert_eq!(pins.len(), 2);
    assert_eq!(pins[0].blur, 30.0);
    assert_eq!(pins[1].blur, IrisPin::default().blur);
    let Some(FilterParams::Shear { points, .. }) = params_for("filter.distort.shear", &json!({"points": [[0, 0.1], [1, -0.1]]})) else { panic!() };
    assert_eq!(points, vec![[0.0, 0.1], [1.0, -0.1]]);
    assert!(params_for("filter.nope", &json!({})).is_none());
    assert_eq!(
        params_for("filter.render.relight", &json!({})),
        Some(FilterParams::Relight { angle: 45.0, elevation: 40.0, intensity: 40.0, ambient: 55.0, warmth: 0.0, softness: 25.0 })
    );
}

fn paint_shaded_disc(s: &mut Session, size: i32) {
    s.edit("disc", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let cx = size as f32 / 2.0;
        let rad = size as f32 * 0.42;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cx;
                let d2 = dx * dx + dy * dy;
                let px = if d2 <= rad * rad {
                    let nz = (rad * rad - d2).max(0.0).sqrt() / rad;
                    let sh = 0.22 + 0.78 * nz;
                    [0.55 * sh, 0.55 * sh, 0.55 * sh, 1.0]
                } else {
                    [0.18, 0.18, 0.18, 1.0]
                };
                surf.write_pixel(x, y, &px);
            }
        }
        Ok(())
    })
    .unwrap();
}

fn relight_session(w: i32, h: i32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    paint_shaded_disc(&mut s, w.min(h));
    s
}

#[test]
fn relight_command_exists_and_toggles_with_document() {
    let spec = crate::commands::find("filter.render.relight").expect("registered");
    assert_eq!(spec.menu, &["Filter", "Render"]);
    assert_eq!(spec.label, "Relight…");
    assert!(spec.journal);
    let mut s = Session::new();
    assert!(!s.is_enabled("filter.render.relight"));
    s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    assert!(s.is_enabled("filter.render.relight"));
    s.execute("type.create", json!({"x": 8, "y": 16, "text": "Hi", "size": 12})).unwrap();
    assert!(!s.is_enabled("filter.render.relight"));
}

#[test]
fn relight_is_disabled_on_a_pixel_locked_layer() {
    let mut s = relight_session(32, 32);
    assert!(s.is_enabled("filter.render.relight"));
    s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
    assert!(!s.is_enabled("filter.render.relight"));
    assert!(s.execute("filter.render.relight", json!({})).is_err());
}

#[test]
fn relight_defaults_change_a_shaded_disc_and_undo() {
    let mut s = relight_session(48, 48);
    let before = {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 48, 48))
    };
    let depth = s.active().unwrap().history.past_len();
    s.execute("filter.render.relight", json!({})).unwrap();
    let after = {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 48, 48))
    };
    assert_ne!(after, before);
    assert_eq!(s.active().unwrap().history.past_len(), depth + 1);
    s.execute("edit.undo", json!({})).unwrap();
    let undone = {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 48, 48))
    };
    assert_eq!(undone, before);
}

#[test]
fn relight_records_a_smart_filter_and_can_be_hidden() {
    let mut s = relight_session(32, 32);
    let before = {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 32, 32))
    };
    s.execute("filter.convertForSmartFilters", json!({})).unwrap();
    s.execute("filter.render.relight", json!({"angle": 180, "intensity": 70})).unwrap();
    let (is_relight, lit) = {
        let d = s.active().unwrap();
        let layer = d.doc.layer(d.active_layer.unwrap()).unwrap();
        let photocraft_doc::LayerContent::Smart(sm) = &layer.content else { panic!("expected a smart object") };
        let ok = sm.smart_filters.iter().any(|f| f.command == "filter.render.relight");
        (ok, layer.surface().unwrap().read_region(Rect::new(0, 0, 32, 32)))
    };
    assert!(is_relight);
    assert_ne!(lit, before);
    s.execute("layer.smartFilter.setVisible", json!({"index": 0, "visible": false})).unwrap();
    let restored = {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 32, 32))
    };
    assert_eq!(restored, before);
}

#[test]
fn relight_rejects_bad_params() {
    let mut s = relight_session(16, 16);
    assert!(s.execute("filter.render.relight", json!({"intensity": 999})).is_err());
    assert!(s.execute("filter.render.relight", json!({"elevation": -3})).is_err());
    assert!(s.execute("filter.render.relight", json!({"angle": "east"})).is_err());
    assert!(s.execute("filter.render.relight", json!({"softness": 0})).is_err());
    assert!(s.execute("filter.render.relight", json!(null)).is_err());
}

#[test]
fn relight_honours_the_selection() {
    let mut s = relight_session(48, 32);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 12, "height": 12})).unwrap();
    let before = pixels(&s);
    s.execute("filter.render.relight", json!({"angle": 180, "intensity": 70})).unwrap();
    let after = pixels(&s);
    let far = (28 * 48 + 40) * 4;
    assert_eq!(after[far..far + 4], before[far..far + 4]);
    assert_ne!(after[..12 * 4], before[..12 * 4]);
}
