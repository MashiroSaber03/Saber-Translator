use super::*;

mod selection;
mod stack_cache;

const DEPTHS: [u64; 3] = [8, 16, 32];
const W: i32 = 64;
const H: i32 = 48;

fn session(depth: u64) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": W, "height": H, "depth": depth})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s
}

/// Paints a textured block on the active layer, partly outside the canvas (top-left), so
/// conversion has to keep off-canvas pixels and crop/translate correctly.
fn paint(s: &mut Session) {
    s.edit("setup", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let r = Rect::new(-6, 5, 40, 38);
        let mut data = Vec::new();
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let v = ((x * 7 + y * 13) % 23) as f32 / 22.0;
                let a = if (x + y) % 5 == 0 { 0.5 } else { 1.0 };
                data.extend(photocraft_raster::from_rgba(&fmt, [v, 1.0 - v, ((x * y) % 7) as f32 / 6.0, a]));
            }
        }
        surf.write_region(r, &data);
        Ok(())
    })
    .unwrap();
}

fn flat(s: &Session) -> Vec<[f32; 4]> {
    photocraft_compose::flatten(&s.active().unwrap().doc).px
}

fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter().zip(b).flat_map(|(p, q)| (0..4).map(move |c| (p[c] - q[c]).abs())).fold(0.0, f32::max)
}

fn mean_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    let sum: f32 = a.iter().zip(b).map(|(p, q)| (0..4).map(|c| (p[c] - q[c]).abs()).sum::<f32>()).sum();
    sum / (a.len() * 4) as f32
}

fn active_smart(s: &Session) -> SmartObject {
    let d = s.active().unwrap();
    match &d.doc.layer(d.active_layer.unwrap()).unwrap().content {
        LayerContent::Smart(sm) => sm.clone(),
        other => panic!("not smart: {}", other.kind_name()),
    }
}

fn convert(s: &mut Session) -> u64 {
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].as_u64().unwrap()
}

#[test]
fn conversion_is_pixel_identical_at_every_depth() {
    for depth in DEPTHS {
        let mut s = session(depth);
        paint(&mut s);
        let id = s.active().unwrap().active_layer.unwrap();
        s.edit("opacity", |doc, _| {
            doc.layer_mut(id).unwrap().opacity = 0.6;
            Ok(())
        })
        .unwrap();
        let before = flat(&s);
        convert(&mut s);
        assert_eq!(max_diff(&flat(&s), &before), 0.0, "depth {depth}");
        let d = s.active().unwrap();
        let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
        assert!((l.opacity - 0.6).abs() < 1e-6, "opacity stays on the smart layer");
        let sm = active_smart(&s);
        let SmartSource::Embedded { file_name, bytes } = &sm.source else { panic!() };
        assert!(file_name.ends_with(".pcraft"));
        // The nested document is cropped to the content, including the off-canvas part.
        let inner = decode_source(file_name, bytes).unwrap();
        assert_eq!((inner.size.width, inner.size.height), (46, 33));
        assert_eq!(inner.depth, d.doc.depth);
        assert_eq!(sm.transform, Affine::translate(-6.0, 5.0));
        // Re-rendering from the source reproduces the conversion cache exactly.
        let rendered = render(&d.doc, &sm).unwrap().unwrap();
        let b = Rect::new(-6, 5, 40, 38);
        assert_eq!(rendered.read_region(b), sm.cache.as_ref().unwrap().read_region(b), "depth {depth}");
        s.undo();
        assert_eq!(flat(&s), before);
    }
}

#[test]
fn masked_layer_and_group_convert_within_rounding() {
    let mut s = session(8);
    paint(&mut s);
    s.edit("mask", |doc, active| {
        let l = doc.layer_mut(active.unwrap()).unwrap();
        let mut m = LayerMask::reveal_all();
        m.surface.fill_rect(Rect::new(0, 0, 20, 20), &[0.3]);
        l.mask = Some(m);
        Ok(())
    })
    .unwrap();
    let before = flat(&s);
    convert(&mut s);
    assert!(max_diff(&flat(&s), &before) <= 1.0 / 255.0 + 1e-6);
    // A group converts too (its children move into the nested document).
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s);
    s.execute("layer.groupLayers", json!({})).unwrap();
    let before = flat(&s);
    convert(&mut s);
    assert!(max_diff(&flat(&s), &before) <= 1.0 / 255.0 + 1e-6);
}

#[test]
fn pattern_fill_converts_to_editable_smart_object_without_losing_its_pattern() {
    let mut s = session(8);
    s.execute("layer.newFillLayer.pattern", json!({"pattern": "Bricks"})).unwrap();
    let before = flat(&s);
    let pattern_id = s.active().unwrap().doc.patterns[0].id.clone();
    assert!(before.iter().any(|p| p[3] > 0.0), "the pattern fill is visible");

    convert(&mut s);
    assert!(max_diff(&flat(&s), &before) < 1e-6, "conversion keeps every pattern pixel");
    let sm = active_smart(&s);
    let SmartSource::Embedded { file_name, bytes } = &sm.source else { panic!("not embedded") };
    let inner = decode_source(file_name, bytes).unwrap();
    assert!(inner.patterns.iter().any(|p| p.id == pattern_id), "the embedded source keeps the pattern resource");

    // The smart object's source remains live after opening and editing its contents.
    let child = s.execute("layer.smartObjects.editContents", json!({})).unwrap()["document"].as_u64().unwrap() as usize;
    assert_eq!(s.active_index(), Some(child));
    assert!(s.active().unwrap().doc.patterns.iter().any(|p| p.id == pattern_id));
    s.set_active(0);
    assert!(max_diff(&flat(&s), &before) < 1e-6);
    s.undo();
    assert!(max_diff(&flat(&s), &before) < 1e-6);
}

#[test]
fn empty_layer_cannot_be_converted() {
    let mut s = session(8);
    assert!(s.execute("layer.smartObjects.convertToSmartObject", json!({})).is_err());
}

#[test]
fn smart_filters_are_re_editable() {
    for depth in [8, 16] {
        let mut s = session(depth);
        paint(&mut s);
        let original = flat(&s);
        s.execute("filter.convertForSmartFilters", json!({})).unwrap();
        assert!(!s.is_enabled("filter.convertForSmartFilters"), "already smart");
        s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
        let r2 = flat(&s);
        assert!(max_diff(&r2, &original) > 0.05);
        let sm = active_smart(&s);
        assert_eq!(sm.smart_filters.len(), 1);
        let SmartSource::Embedded { bytes, .. } = &sm.source else { panic!() };
        // Change the radius: the result comes from the source, not from the blurred pixels.
        s.execute("layer.smartFilter.setParams", json!({"index": 0, "params": {"radius": 6}})).unwrap();
        let r6 = flat(&s);
        assert!(max_diff(&r6, &r2) > 0.01);
        s.execute("layer.smartFilter.setParams", json!({"params": {"radius": 2}})).unwrap();
        assert!(max_diff(&flat(&s), &r2) < 1e-6, "same params, same pixels");
        // Hiding the filter shows the original exactly.
        s.execute("layer.smartFilter.setVisible", json!({"index": 0})).unwrap();
        assert_eq!(flat(&s), original);
        assert!(!active_smart(&s).smart_filters[0].visible);
        s.undo();
        assert!(max_diff(&flat(&s), &r2) < 1e-6);
        s.redo();
        assert_eq!(flat(&s), original);
        // All of that decoded the contents at most once (the conversion seeded the cache).
        assert!(source_decode_count(bytes, s.active().unwrap().doc.pixel_format()) <= 1);
    }
}

#[test]
fn filter_stack_order_blend_disable_and_clear() {
    let mut s = session(8);
    paint(&mut s);
    let original = flat(&s);
    convert(&mut s);
    s.execute("filter.other.offset", json!({"horizontal": 5, "undefinedAreas": "transparent"})).unwrap();
    s.execute("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
    let ab = flat(&s);
    s.execute("layer.smartFilter.move", json!({"index": 1, "to": 0})).unwrap();
    assert_eq!(active_smart(&s).smart_filters[0].command, "filter.blur.gaussianBlur");
    assert!(max_diff(&flat(&s), &ab) > 1e-3, "order matters");
    s.undo();
    assert_eq!(flat(&s), ab);
    // 0 % opacity on both filters = original.
    s.execute("layer.smartFilter.blendingOptions", json!({"index": 0, "opacity": 0})).unwrap();
    s.execute("layer.smartFilter.blendingOptions", json!({"index": 1, "opacity": 0, "blend": "multiply"})).unwrap();
    assert!(max_diff(&flat(&s), &original) < 1e-6);
    assert_eq!(active_smart(&s).smart_filters[1].blend, BlendMode::Multiply);
    assert!(s.execute("layer.smartFilter.blendingOptions", json!({"blend": "nope"})).is_err());
    s.undo();
    s.undo();
    s.execute("layer.smartFilter.disableSmartFilters", json!({})).unwrap();
    assert_eq!(flat(&s), original);
    assert!(!active_smart(&s).filters_enabled);
    s.execute("layer.smartFilter.disableSmartFilters", json!({})).unwrap();
    assert_eq!(flat(&s), ab);
    s.execute("layer.smartFilter.delete", json!({"index": 0})).unwrap();
    assert_eq!(active_smart(&s).smart_filters.len(), 1);
    assert!(s.execute("layer.smartFilter.delete", json!({"index": 5})).is_err());
    s.execute("layer.smartFilter.clearSmartFilters", json!({})).unwrap();
    assert_eq!(flat(&s), original);
    assert!(!s.is_enabled("layer.smartFilter.clearSmartFilters"));
}

/// Issue #466: an explicit `layer` target is checked instead of the active layer, which stays
/// active; without one the commands still follow the active layer (menus).
#[test]
fn smart_filter_commands_check_an_explicit_layer_target() {
    let mut s = session(8);
    paint(&mut s);
    let so = convert(&mut s);
    s.execute("filter.sharpen.smartSharpen", json!({"amount": 65, "radius": 1})).unwrap();
    let curves = s.execute("layer.newAdjustmentLayer.curves", json!({"points": [[0, 0], [255, 255]]})).unwrap()["layer"].as_u64().unwrap();
    let active = |s: &Session| s.active().unwrap().active_layer.unwrap().0;
    assert_eq!(active(&s), curves);
    let filter = |s: &Session| match &s.active().unwrap().doc.layer(LayerId(so)).unwrap().content {
        LayerContent::Smart(sm) => sm.smart_filters[0].clone(),
        other => panic!("not smart: {}", other.kind_name()),
    };
    // No target: the active Curves layer decides, as the menu shows.
    assert!(!s.is_enabled("layer.smartFilter.setParams"));
    assert!(matches!(s.execute("layer.smartFilter.setParams", json!({"params": {"amount": 80}})), Err(EngineError::Disabled(..))));
    let r = s.execute("layer.smartFilter.setParams", json!({"layer": so, "index": 0, "params": {"amount": 80}})).unwrap();
    assert_eq!(r, json!({"layer": so}));
    assert_eq!(filter(&s).params["amount"], json!(80));
    s.execute("layer.smartFilter.setVisible", json!({"layer": so, "visible": false})).unwrap();
    assert!(!filter(&s).visible);
    assert_eq!(active(&s), curves, "the active layer stays");
    // A target that isn't a smart object with filters is still refused, and the active layer stays.
    assert!(matches!(s.execute("layer.smartFilter.setParams", json!({"layer": curves, "params": {}})), Err(EngineError::Disabled(..))));
    assert!(s.execute("layer.smartFilter.setParams", json!({"layer": u64::MAX, "params": {}})).is_err());
    assert_eq!(active(&s), curves);
}

/// #4271: dragging smart filters onto another smart object moves them, ⌥-dragging copies them;
/// one filter or all of them (with the filter mask), each in one undo step.
#[test]
fn smart_filters_move_or_copy_to_another_smart_object() {
    let mut s = session(8);
    paint(&mut s);
    let a = convert(&mut s);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 16, "height": 48})).unwrap();
    s.execute("filter.other.offset", json!({"horizontal": 5, "undefinedAreas": "transparent"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s);
    let b = convert(&mut s);
    let pixel = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
    let so = |s: &Session, id: u64| match &s.active().unwrap().doc.layer(LayerId(id)).unwrap().content {
        LayerContent::Smart(sm) => sm.clone(),
        other => panic!("not smart: {}", other.kind_name()),
    };
    let commands = |sm: &SmartObject| sm.smart_filters.iter().map(|f| f.command.clone()).collect::<Vec<_>>();
    let (offset, blur) = ("filter.other.offset", "filter.blur.gaussianBlur");
    assert_eq!(commands(&so(&s, a)), [offset, blur]);
    assert!(so(&s, a).filter_mask.is_some());
    let before = flat(&s);

    // ⌥-drag the blur: B gets a copy (re-rendered), A keeps it.
    s.execute("layer.smartFilter.transfer", json!({"from": a, "to": b, "filter": 1, "copy": true})).unwrap();
    assert_eq!(commands(&so(&s, b)), [blur]);
    assert_eq!(commands(&so(&s, a)), [offset, blur]);
    assert!(so(&s, b).filter_mask.is_none(), "one filter comes without the mask");
    assert!(max_diff(&flat(&s), &before) > 1e-3, "B is blurred");
    s.undo();
    assert!(so(&s, b).smart_filters.is_empty());
    assert_eq!(flat(&s), before);

    // Drag the offset: it moves to the top of B's filters.
    s.redo();
    s.execute("layer.smartFilter.transfer", json!({"from": a, "to": b, "filter": 0})).unwrap();
    assert_eq!(commands(&so(&s, a)), [blur]);
    assert_eq!(commands(&so(&s, b)), [blur, offset]);

    // Drag the Smart Filters row: all of A's filters and its filter mask move to B.
    let mask = so(&s, a).filter_mask;
    s.execute("layer.smartFilter.transfer", json!({"from": a, "to": b})).unwrap();
    assert!(so(&s, a).smart_filters.is_empty() && so(&s, a).filter_mask.is_none());
    assert_eq!(commands(&so(&s, b)), [blur, offset, blur]);
    assert_eq!(so(&s, b).filter_mask, mask);
    s.undo();
    assert_eq!(so(&s, a).smart_filters.len(), 1);
    assert_eq!(so(&s, b).smart_filters.len(), 2);

    // Refused: onto itself, onto a layer that isn't a smart object, a bad index, no filters.
    let transfer = |s: &mut Session, p: Value| s.execute("layer.smartFilter.transfer", p);
    assert!(transfer(&mut s, json!({"from": a, "to": a})).is_err());
    assert!(transfer(&mut s, json!({"from": a, "to": pixel})).is_err());
    assert!(transfer(&mut s, json!({"from": a, "to": b, "filter": 5})).is_err());
    assert!(transfer(&mut s, json!({"from": a, "to": b, "index": 0})).is_err(), "`index` isn't taken for `filter`");
    assert!(transfer(&mut s, json!({"from": pixel, "to": b})).is_err());
    s.execute("layer.smartFilter.clearSmartFilters", json!({"layer": a})).unwrap();
    assert!(transfer(&mut s, json!({"from": a, "to": b})).is_err());
    assert_eq!(so(&s, b).smart_filters.len(), 2, "nothing changed");
}

#[test]
fn selection_becomes_the_filter_mask() {
    let mut s = session(8);
    paint(&mut s);
    let original = flat(&s);
    convert(&mut s);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 16, "height": 48})).unwrap();
    s.execute("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
    assert!(active_smart(&s).filter_mask.is_some());
    let px = flat(&s);
    let at = |v: &Vec<[f32; 4]>, x: i32, y: i32| v[(y * W + x) as usize];
    assert_eq!(at(&px, 30, 20), at(&original, 30, 20), "outside the mask: unfiltered");
    assert_ne!(at(&px, 6, 20), at(&original, 6, 20), "inside the mask: filtered");
    s.execute("layer.smartFilter.disableFilterMask", json!({})).unwrap();
    assert_ne!(at(&flat(&s), 30, 20), at(&original, 30, 20));
    s.execute("layer.smartFilter.deleteFilterMask", json!({})).unwrap();
    assert!(active_smart(&s).filter_mask.is_none());
    assert!(!s.is_enabled("layer.smartFilter.deleteFilterMask"));
}

/// Scales `id` to 10 % around the origin and back, returning the composite.
fn shrink_and_grow(s: &mut Session, id: u64) -> Vec<[f32; 4]> {
    s.execute("edit.transform", json!({"layer": id, "matrix": [0.1, 0, 0, 0.1, 0, 0]})).unwrap();
    s.execute("edit.transform", json!({"layer": id, "matrix": [10, 0, 0, 10, 0, 0]})).unwrap();
    flat(s)
}

#[test]
fn transforms_re_render_from_source_losslessly() {
    let mut raster = session(8);
    paint(&mut raster);
    let original = flat(&raster);
    let rid = raster.active().unwrap().active_layer.unwrap().0;
    let raster_result = shrink_and_grow(&mut raster, rid);

    let mut smart = session(8);
    paint(&mut smart);
    let sid = convert(&mut smart);
    // Scaled down, the smart object really is small...
    smart.execute("edit.transform", json!({"layer": sid, "matrix": [0.1, 0, 0, 0.1, 0, 0]})).unwrap();
    let small = active_smart(&smart).cache.unwrap().content_bounds();
    assert!(small.width() <= 7, "{small:?}");
    // ...and scaled back up it is the original again.
    smart.execute("edit.transform", json!({"layer": sid, "matrix": [10, 0, 0, 10, 0, 0]})).unwrap();
    let smart_result = flat(&smart);
    let (es, er) = (mean_diff(&smart_result, &original), mean_diff(&raster_result, &original));
    assert!(es < 1e-6, "smart error {es}");
    assert!(er > 0.02, "raster error {er}");
    // A non-integer move re-renders with bicubic; moving back is exact again.
    smart.execute("edit.transform", json!({"layer": sid, "matrix": [1, 0, 0, 1, 0.5, 0.25]})).unwrap();
    assert!(max_diff(&flat(&smart), &original) > 0.01);
    smart.execute("edit.transform", json!({"layer": sid, "matrix": [1, 0, 0, 1, -0.5, -0.25]})).unwrap();
    assert!(max_diff(&flat(&smart), &original) < 1e-6);
    // Whole-pixel moves (Move tool) shift without re-rendering.
    smart.execute("layer.translate", json!({"dx": 3, "dy": -2})).unwrap();
    assert_eq!(active_smart(&smart).transform, Affine::translate(-3.0, 3.0));
    // Perspective keeps the full projective placement (and stays a smart object).
    smart.execute("edit.transform", json!({"layer": sid, "quad": [[0, 0], [10, 0], [12, 10], [-2, 10]]})).unwrap();
    assert!(active_smart(&smart).perspective.is_some());
}

#[test]
fn filters_follow_transforms() {
    let mut s = session(16);
    paint(&mut s);
    let id = convert(&mut s);
    s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
    let blurred = flat(&s);
    s.execute("edit.transform", json!({"layer": id, "matrix": [2, 0, 0, 2, 0, 0]})).unwrap();
    assert_eq!(active_smart(&s).smart_filters.len(), 1);
    s.execute("edit.transform", json!({"layer": id, "matrix": [0.5, 0, 0, 0.5, 0, 0]})).unwrap();
    assert!(max_diff(&flat(&s), &blurred) < 1e-6, "filter re-applied to the re-rendered source");
}

#[test]
fn edit_contents_updates_the_parent() {
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    let before = flat(&s);
    let r = s.execute("layer.smartObjects.editContents", json!({})).unwrap();
    let child = r["document"].as_u64().unwrap() as usize;
    assert_eq!(s.active_index(), Some(child));
    assert!(s.is_enabled("layer.smartObjects.saveContents"));
    // Paint the nested document's layer solid red.
    s.edit("paint", |doc, _| {
        let l = &mut doc.layers[0];
        let b = Rect::new(0, 0, 46, 33);
        l.surface_mut().unwrap().fill_rect(b, &[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("layer.smartObjects.saveContents", json!({})).unwrap();
    assert!(!s.active().unwrap().is_dirty());
    s.set_active(0);
    let after = flat(&s);
    assert_eq!(after[(20 * W + 10) as usize], [1.0, 0.0, 0.0, 1.0]);
    assert_ne!(after, before);
    s.undo();
    assert_eq!(flat(&s), before, "the update is one undoable step in the parent");
    s.redo();
    // Editing the same smart object again switches to the open document, no second copy.
    let open = s.documents().len();
    let again = s.execute("layer.smartObjects.editContents", json!({})).unwrap();
    assert_eq!((again["document"].as_u64(), s.active_index(), s.documents().len()), (Some(child as u64), Some(child), open));
    // Closing an edited contents document also commits it.
    s.set_active(child);
    s.edit("paint2", |doc, _| {
        doc.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 46, 33), &[0.0, 0.0, 1.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("file.close", json!({})).unwrap();
    assert!(s.smart_links.is_empty());
    s.set_active(0);
    assert_eq!(flat(&s)[(20 * W + 10) as usize], [0.0, 0.0, 1.0, 1.0]);
    let d = s.active().unwrap();
    assert!(matches!(d.doc.layer(LayerId(id)).unwrap().content, LayerContent::Smart(_)));
}

#[test]
fn rasterize_and_via_copy() {
    let mut s = session(8);
    paint(&mut s);
    convert(&mut s);
    s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
    let blurred = flat(&s);
    // Via copy: an independent smart object above.
    let copy = s.execute("layer.smartObjects.newSmartObjectViaCopy", json!({})).unwrap()["layer"].as_u64().unwrap();
    assert_eq!(s.active().unwrap().active_layer, Some(LayerId(copy)));
    s.execute("layer.smartFilter.setParams", json!({"params": {"radius": 8}})).unwrap();
    let d = s.active().unwrap();
    let n_smart = d.doc.walk().iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Smart(_))).count();
    assert_eq!(n_smart, 2);
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("layer.smartObjects.rasterize", json!({})).unwrap();
    let d = s.active().unwrap();
    assert!(matches!(d.doc.layer(d.active_layer.unwrap()).unwrap().content, LayerContent::Raster(_)));
    assert!(max_diff(&flat(&s), &blurred) <= 1.0 / 255.0 + 1e-6);
    assert!(!s.is_enabled("layer.smartObjects.rasterize"));
}

#[test]
fn pcraft_round_trip_keeps_live_smart_objects() {
    for depth in [8, 16] {
        let mut s = session(depth);
        paint(&mut s);
        convert(&mut s);
        s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 30, "height": 30})).unwrap();
        s.execute("filter.noise.addNoise", json!({"amount": 20, "seed": 3})).unwrap();
        s.execute("layer.smartFilter.blendingOptions", json!({"opacity": 0.5, "blend": "screen"})).unwrap();
        let doc = s.active().unwrap().doc.clone();
        let bytes = photocraft_format::save_to_bytes(&doc, &Default::default()).unwrap();
        let back = photocraft_format::load_from_bytes(&bytes).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        let (a, b) = (smart(&doc, id).unwrap(), smart(&back, id).unwrap());
        assert_eq!(a, b, "depth {depth}");
        assert_eq!(b.smart_filters.len(), 2);
        assert!(b.filter_mask.is_some());
        // Still live after loading: re-editing works and matches a fresh render.
        let mut s2 = Session::new();
        s2.add_document(back, None);
        s2.select_layer(id).unwrap();
        s2.execute("layer.smartFilter.setVisible", json!({"index": 1, "visible": false})).unwrap();
        s.execute("layer.smartFilter.setVisible", json!({"index": 1, "visible": false})).unwrap();
        assert_eq!(flat(&s2), flat(&s));
    }
}

#[test]
fn psd_placed_layer_renders_from_lnk2_data() {
    // A PNG embedded the way Photoshop stores placed files: a global lnk2 block keyed by uuid.
    let mut inner = Document::with_background(
        "in",
        Size::new(10, 8),
        photocraft_color::ColorMode::Rgb,
        photocraft_color::SampleType::U8,
        photocraft_color::Color::rgba(0.0, 1.0, 0.0, 1.0),
    );
    inner.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 5, 8), &[1.0, 0.0, 0.0, 1.0]);
    let png = photocraft_io::export(&inner, "png", &Default::default()).unwrap().bytes;
    let lnk2 = photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile { uuid: "uuid-1".into(), file_name: "in.png".into(), bytes: png });
    let mut s = session(8);
    s.edit("place", |doc, active| {
        doc.metadata.psd_global_blocks.push((*b"8BIM", *b"lnk2", Arc::new(lnk2)));
        // Photoshop's rendering (deliberately different, to see the re-render happen).
        let mut cache = Surface::new(doc.pixel_format());
        cache.fill_rect(Rect::new(20, 10, 40, 26), &[0.0, 0.0, 1.0, 1.0]);
        let mut sm = SmartObject::new(SmartSource::Linked { path: "uuid-1".into() }, Affine { m: [2.0, 0.0, 0.0, 2.0, 20.0, 10.0] }, Some(cache));
        sm.psd_raw = Some(Arc::new(vec![1, 2, 3]));
        let mut l = Layer::new("placed", LayerContent::Smart(sm));
        l.psd_blocks.push((*b"SoLd", Arc::new(vec![1, 2, 3])));
        *active = Some(doc.insert_above(*active, l));
        Ok(())
    })
    .unwrap();
    // Nothing re-renders until something changes (PSD round trips stay byte-stable).
    assert_eq!(active_smart(&s).cache.unwrap().rgba(25, 12), [0.0, 0.0, 1.0, 1.0]);
    s.execute("filter.other.offset", json!({"horizontal": 0})).unwrap();
    let c = active_smart(&s).cache.unwrap();
    assert_eq!(c.rgba(23, 15), [1.0, 0.0, 0.0, 1.0], "left half red, from the source");
    assert_eq!(c.rgba(37, 15), [0.0, 1.0, 0.0, 1.0]);
    assert!(active_smart(&s).psd_raw.is_some(), "filters alone keep the placed-layer block");
    // Embedding resolves the lnk2 data.
    assert!(s.is_enabled("layer.smartObjects.convertToEmbedded"));
    s.execute("layer.smartObjects.convertToEmbedded", json!({})).unwrap();
    assert!(matches!(active_smart(&s).source, SmartSource::Embedded { ref file_name, .. } if file_name == "in.png"));
}

#[test]
fn export_replace_and_linked_contents() {
    let dir = std::env::temp_dir().join(format!("pcraft-smart-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut s = session(8);
    paint(&mut s);
    convert(&mut s);
    let out = dir.join("contents.pcraft");
    let r = s.execute("layer.smartObjects.exportContents", json!({"path": out.to_str().unwrap()})).unwrap();
    assert!(photocraft_format::is_pcraft(&std::fs::read(&out).unwrap()));
    assert_eq!(r["converted"], false);
    // Another extension exports the contents in that format.
    for (ext, magic) in [("png", &b"\x89PNG"[..]), ("jpeg", &b"\xFF\xD8\xFF"[..]), ("psd", &b"8BPS"[..])] {
        let other = dir.join(format!("contents.{ext}"));
        let r = s.execute("layer.smartObjects.exportContents", json!({"path": other.to_str().unwrap()})).unwrap();
        assert_eq!(r["converted"], true, "{ext}");
        assert!(std::fs::read(&other).unwrap().starts_with(magic), "{ext}");
    }
    let bad = dir.join("contents.nope");
    assert!(s.execute("layer.smartObjects.exportContents", json!({"path": bad.to_str().unwrap()})).is_err());
    // Replace with a flat PNG.
    let png_doc = Document::with_background(
        "p",
        Size::new(4, 4),
        photocraft_color::ColorMode::Rgb,
        photocraft_color::SampleType::U8,
        photocraft_color::Color::rgba(1.0, 1.0, 0.0, 1.0),
    );
    let png = dir.join("yellow.png");
    std::fs::write(&png, photocraft_io::export(&png_doc, "png", &Default::default()).unwrap().bytes).unwrap();
    s.execute("layer.smartObjects.replaceContents", json!({"path": png.to_str().unwrap()})).unwrap();
    let sm = active_smart(&s);
    assert_eq!(sm.cache.as_ref().unwrap().content_bounds(), Rect::new(-6, 5, -2, 9), "transform kept");
    assert!(s.execute("layer.smartObjects.replaceContents", json!({})).is_err());
    // Linked: write out, edit the file, update.
    let linked = dir.join("linked.png");
    s.execute("layer.smartObjects.convertToLinked", json!({"path": linked.to_str().unwrap()})).unwrap();
    assert!(matches!(active_smart(&s).source, SmartSource::Linked { .. }));
    let mut red = png_doc.clone();
    let rb = red.bounds();
    red.layers[0].surface_mut().unwrap().fill_rect(rb, &[1.0, 0.0, 0.0, 1.0]);
    std::fs::write(&linked, photocraft_io::export(&red, "png", &Default::default()).unwrap().bytes).unwrap();
    s.execute("layer.smartObjects.updateAllModifiedContent", json!({})).unwrap();
    assert_eq!(active_smart(&s).cache.unwrap().rgba(-5, 6), [1.0, 0.0, 0.0, 1.0]);
    s.execute("layer.smartObjects.convertToEmbedded", json!({})).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    // Missing linked file: re-rendering fails cleanly and changes nothing.
    s.execute("layer.smartObjects.relinkToFile", json!({"path": dir.join("gone.png").to_str().unwrap()})).unwrap_err();
    assert!(matches!(active_smart(&s).source, SmartSource::Embedded { .. }));
}

#[test]
fn commands_are_registered_with_catalog_ids() {
    for id in [
        "layer.smartObjects.convertToSmartObject",
        "filter.convertForSmartFilters",
        "layer.smartObjects.newSmartObjectViaCopy",
        "layer.smartObjects.rasterize",
        "layer.smartObjects.editContents",
        "layer.smartObjects.replaceContents",
        "layer.smartObjects.exportContents",
        "layer.smartObjects.updateModifiedContent",
        "layer.smartObjects.updateAllModifiedContent",
        "layer.smartObjects.convertToEmbedded",
        "layer.smartObjects.convertToLinked",
        "layer.smartObjects.relinkToFile",
        "layer.smartFilter.disableSmartFilters",
        "layer.smartFilter.clearSmartFilters",
        "layer.smartFilter.blendingOptions",
        "layer.smartFilter.deleteFilterMask",
        "layer.smartFilter.disableFilterMask",
    ] {
        let spec = crate::commands::find(id).unwrap_or_else(|| panic!("{id}"));
        assert!(!spec.menu.is_empty(), "{id}");
    }
    let s = Session::new();
    assert!(!s.is_enabled("layer.smartObjects.convertToSmartObject"));
    assert!(!s.is_enabled("layer.smartObjects.editContents"));
}

#[test]
fn smart_filter_blur_repeats_the_canvas_edge_like_a_layer_filter() {
    for depth in [8, 32] {
        let fill_canvas = |s: &mut Session| {
            s.edit("setup", |doc, active| {
                let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
                let fmt = surf.format();
                let r = Rect::new(0, 0, W, H);
                let mut data = Vec::new();
                for y in r.y0..r.y1 {
                    for x in r.x0..r.x1 {
                        let v = if (x / 8 + y / 8) % 2 == 0 { 0.2 } else { 0.9 };
                        data.extend(photocraft_raster::from_rgba(&fmt, [v, 0.5, 1.0 - v, 1.0]));
                    }
                }
                surf.write_region(r, &data);
                Ok(())
            })
            .unwrap();
        };
        let mut layer = session(depth);
        fill_canvas(&mut layer);
        layer.execute("filter.blur.gaussianBlur", json!({"radius": 4})).unwrap();
        let mut smart = session(depth);
        fill_canvas(&mut smart);
        convert(&mut smart);
        smart.execute("filter.blur.gaussianBlur", json!({"radius": 4})).unwrap();
        let (a, b) = (flat(&layer), flat(&smart));
        // The corner stays opaque (no transparency fades in from outside the canvas).
        assert!((b[0][3] - 1.0).abs() < 1e-3, "corner alpha {}", b[0][3]);
        assert!(max_diff(&a, &b) < 2.0 / 255.0, "smart re-render matches the layer filter ({depth}-bit)");
    }
}

/// A distorted smart object keeps its fourth corner: through Edit Contents → Save (which
/// re-renders from the source) the placement and the rendered corners don't move.
#[test]
fn distort_survives_edit_contents() {
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    let r = active_smart(&s).cache.unwrap().content_bounds();
    let (x0, y0, x1, y1) = (r.x0, r.y0, r.x1, r.y1);
    // Distort: only the bottom-right corner moves (in by 12, up by 8): not a parallelogram.
    let quad = json!([[x0, y0], [x1, y0], [x1 - 12, y1 - 8], [x0, y1]]);
    s.execute("edit.transform", json!({"layer": id, "rect": [x0, y0, x1, y1], "quad": quad})).unwrap();
    let placed = active_smart(&s);
    assert!(placed.perspective.is_some(), "Distort keeps a projective placement");
    let before = flat(&s);
    s.execute("layer.smartObjects.editContents", json!({})).unwrap();
    s.execute("layer.smartObjects.saveContents", json!({})).unwrap();
    s.set_active(0);
    assert_eq!(active_smart(&s).perspective, placed.perspective, "the placement survives the save");
    assert!(mean_diff(&before, &flat(&s)) < 0.01, "the re-render matches the distorted look");
    // The bottom-right of the original frame is now empty: the corner really moved in.
    let sm = active_smart(&s);
    let c = sm.cache.as_ref().unwrap();
    assert_eq!(c.rgba(x1 - 2, y1 - 2)[3], 0.0, "no pixels where the corner used to be");
}

/// Warp handles on a distorted smart object sit on its four distorted corners (the warp is mapped
/// through the full projective placement, not its affine approximation), and warping it keeps the
/// Distort placement.
#[test]
fn warp_on_a_distorted_smart_object_follows_the_corners() {
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    let r = active_smart(&s).cache.unwrap().content_bounds();
    let (x0, y0, x1, y1) = (f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1));
    let quad = [[x0, y0], [x1, y0], [x1 - 12.0, y1 - 8.0], [x0, y1]];
    s.execute("edit.transform", json!({"layer": id, "rect": [x0, y0, x1, y1], "quad": quad})).unwrap();
    let mut sm = active_smart(&s);
    let h = placement(&sm);
    let inv = h.inverse().unwrap();
    let src: Vec<(f64, f64)> = quad.iter().map(|p| inv.apply(p[0], p[1])).collect();
    let b = src.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| [b[0].min(p.0), b[1].min(p.1), b[2].max(p.0), b[3].max(p.1)]);
    sm.warp = Some(photocraft_geom::warp::Warp::custom(photocraft_geom::warp::BezierMesh::identity(b, 1, 1), b));
    let layer = Layer::new("w", LayerContent::Smart(sm));
    let doc_warp = crate::warp_cmds::smart_warp_doc_space(&layer).unwrap();
    let pts = doc_warp.mesh.unwrap().points;
    for (q, i) in quad.iter().zip([0usize, 3, 15, 12]) {
        let p = pts[i];
        assert!((p[0] - q[0]).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6, "corner {i}: {p:?} vs {q:?}");
    }
    // Warping the distorted smart object works and keeps its projective placement.
    let before = active_smart(&s).perspective;
    assert!(before.is_some());
    s.execute("edit.transform.warp", json!({"style": "arc", "bend": 30})).unwrap();
    let after = active_smart(&s);
    assert!(after.warp.is_some());
    assert_eq!(after.perspective, before);
}

/// A `mesh` whose points are exactly the placement leaves a sheared smart object unchanged: its
/// knots span the contents, not the bounding frame of the warp box seen from the contents (#4113).
#[test]
fn identity_mesh_on_a_skewed_smart_object_changes_nothing() {
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    let r = active_smart(&s).cache.unwrap().content_bounds();
    let (x0, y0, x1, y1) = (f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1));
    let quad = [[x0, y0], [x1, y0 + 10.0], [x1, y1 + 10.0], [x0, y1]];
    s.execute("edit.transform", json!({"layer": id, "rect": [x0, y0, x1, y1], "quad": quad})).unwrap();
    let before = flat(&s);
    let sm = active_smart(&s);
    let (h, b) = (placement(&sm), crate::smart_cmds::source_box(&s.active().unwrap().doc, &sm).unwrap());
    let place = |u: f64, v: f64| {
        let (x, y) = h.apply(b[0] + u * (b[2] - b[0]), b[1] + v * (b[3] - b[1]));
        [x, y]
    };
    let mesh = photocraft_geom::warp::BezierMesh::fit(&place, vec![0.0, 1.0], vec![0.0, 0.25, 1.0]);
    let rect = [x0, y0, x1, y1 + 10.0];
    s.execute("layer.smartObjects.warp", json!({"layer": id, "style": "custom", "rect": rect, "mesh": mesh})).unwrap();
    assert!(mean_diff(&before, &flat(&s)) < 0.01, "mean diff {}", mean_diff(&before, &flat(&s)));
}

fn convert_to_layers(s: &mut Session) -> Value {
    s.execute("layer.smartObjects.convertToLayers", json!({})).unwrap()
}

fn active_layer(s: &Session) -> Layer {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().clone()
}

#[test]
fn convert_to_layers_round_trips_a_layer_at_every_depth() {
    for depth in DEPTHS {
        let mut s = session(depth);
        paint(&mut s);
        let id = s.active().unwrap().active_layer.unwrap();
        s.edit("props", |doc, _| {
            let l = doc.layer_mut(id).unwrap();
            l.name = "Sky".into();
            l.opacity = 0.6;
            l.blend = BlendMode::Multiply;
            Ok(())
        })
        .unwrap();
        let before = flat(&s);
        convert(&mut s);
        let smart = flat(&s);
        let r = convert_to_layers(&mut s);
        assert_eq!(r["group"], false, "one layer replaces the smart object");
        let l = active_layer(&s);
        assert_eq!(l.id.0, r["layer"].as_u64().unwrap());
        assert!(matches!(&l.content, LayerContent::Raster(px) if px.format() == s.active().unwrap().doc.pixel_format()), "depth {depth}");
        assert_eq!((l.name.as_str(), l.blend), ("Sky", BlendMode::Multiply));
        assert!((l.opacity - 0.6).abs() < 1e-6);
        assert_eq!(max_diff(&flat(&s), &before), 0.0, "depth {depth}: whole-pixel placement is exact");
        s.undo();
        assert!(matches!(active_layer(&s).content, LayerContent::Smart(_)), "one undoable step");
        assert_eq!(flat(&s), smart);
    }
}

#[test]
fn convert_to_layers_unpacks_several_layers_into_a_group_with_the_smart_layers_properties() {
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    s.execute("layer.setProps", json!({"layer": id, "name": "Placed", "opacity": 0.5})).unwrap();
    // A second layer inside the contents (Edit Contents, then save).
    let child = s.execute("layer.smartObjects.editContents", json!({})).unwrap()["document"].as_u64().unwrap() as usize;
    s.execute("layer.new.layer", json!({"name": "Top"})).unwrap();
    s.edit("paint", |doc, active| {
        doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(2, 2, 8, 8), &[0.0, 0.0, 1.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("layer.smartObjects.saveContents", json!({})).unwrap();
    s.execute("file.close", json!({})).unwrap();
    assert_ne!(s.active_index(), Some(child));
    s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
    s.execute("filter.blur.gaussianBlur", json!({"radius": 1})).unwrap();
    // Without the filters, the smart object looks exactly like the unpacked group.
    s.execute("layer.smartFilter.disableSmartFilters", json!({})).unwrap();
    let before = flat(&s);
    let r = convert_to_layers(&mut s);
    assert_eq!((r["group"].as_bool(), r["discardedSmartFilters"].as_u64()), (Some(true), Some(2)));
    let g = active_layer(&s);
    assert_eq!((g.name.as_str(), g.blend), ("Placed", BlendMode::Normal), "named after the smart object, isolated");
    assert!((g.opacity - 0.5).abs() < 1e-6);
    let names: Vec<_> = g.children().unwrap().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names.last(), Some(&"Top"));
    assert_eq!(names.len(), 2);
    assert!(max_diff(&flat(&s), &before) <= 1.0 / 255.0 + 1e-6);
    assert!(!s.is_enabled("layer.smartObjects.convertToLayers"), "a group isn't a smart object");
}

#[test]
fn convert_to_layers_keeps_a_smart_layer_mask_on_a_group() {
    let mut s = session(16);
    paint(&mut s);
    convert(&mut s);
    s.execute("layer.layerMask.hideAll", json!({})).unwrap();
    let before = flat(&s);
    assert!(convert_to_layers(&mut s)["group"].as_bool().unwrap(), "the mask can't fold into the layer");
    let g = active_layer(&s);
    assert!(g.mask.is_some() && g.children().unwrap()[0].mask.is_none());
    assert_eq!(max_diff(&flat(&s), &before), 0.0);
}

#[test]
fn convert_to_layers_applies_the_placement() {
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    s.execute("edit.transform", json!({"layer": id, "matrix": [0.5, 0, 0, 0.5, 10.25, 3]})).unwrap();
    let before = flat(&s);
    convert_to_layers(&mut s);
    let l = active_layer(&s);
    assert!(matches!(l.content, LayerContent::Raster(_)));
    assert!(mean_diff(&flat(&s), &before) < 2e-3, "{}", mean_diff(&flat(&s), &before));
    // Distort keeps the projective placement for pixels.
    let mut s = session(8);
    paint(&mut s);
    let id = convert(&mut s);
    s.execute("edit.transform", json!({"layer": id, "quad": [[0, 0], [30, 0], [36, 30], [-4, 30]]})).unwrap();
    let before = flat(&s);
    convert_to_layers(&mut s);
    assert!(mean_diff(&flat(&s), &before) < 2e-3);
}

#[test]
fn convert_to_layers_rejects_what_it_cannot_unpack() {
    let mut s = session(8);
    assert!(!s.is_enabled("layer.smartObjects.convertToLayers"));
    assert!(s.execute("layer.smartObjects.convertToLayers", json!({})).is_err());
    paint(&mut s);
    let raster = s.active().unwrap().active_layer.unwrap().0;
    let id = convert(&mut s);
    // Bad targets fail without panicking: a pixel layer, a missing layer.
    for p in [json!({"layer": raster}), json!({"layer": u64::MAX})] {
        assert!(s.execute("layer.smartObjects.convertToLayers", p).is_err());
    }
    // A warp can't be applied to the unpacked layers: an error, and the document is unchanged.
    s.execute("edit.transform.warp", json!({"layer": id, "style": "arc", "bend": 30})).unwrap();
    let steps = s.active().unwrap().history.entries().len();
    assert!(s.execute("layer.smartObjects.convertToLayers", json!({})).is_err());
    assert_eq!(s.active().unwrap().history.entries().len(), steps);
}

#[test]
fn convert_to_layers_conforms_the_contents_to_the_document_mode_and_depth() {
    let mut s = session(8);
    paint(&mut s);
    convert(&mut s);
    // The contents stay 8-bit RGB while the document becomes 16-bit grayscale.
    s.execute("image.mode.bits16", json!({})).unwrap();
    s.execute("image.mode.grayscale", json!({})).unwrap();
    convert_to_layers(&mut s);
    let fmt = s.active().unwrap().doc.pixel_format();
    assert!(matches!(&active_layer(&s).content, LayerContent::Raster(px) if px.format().mode == fmt.mode && px.format().sample == fmt.sample));
}

fn deny_contents_paths(cmd: &str, p: &Value) -> Result<()> {
    if matches!(cmd, "layer.smartObjects.editContents" | "layer.smartObjects.convertToLayers" | "layer.smartObjects.saveContents") && p.get("path").is_some() {
        return Err(other(format!("source path refused: {cmd}: {}", p["path"])));
    }
    Ok(())
}

#[test]
fn automation_smart_contents_accepts_memory_sources_and_explicit_targets() {
    for (depth, psd_uuid) in [(8, false), (16, true), (32, false)] {
        let mut s = session(depth);
        paint(&mut s);
        let id = convert(&mut s);
        if psd_uuid {
            s.edit("PSD source", |doc, _| {
                let SmartSource::Embedded { file_name, bytes } = smart(doc, LayerId(id))?.source.clone() else { panic!() };
                let linked = photocraft_io::linked::LinkedFile { uuid: "embedded-id".into(), file_name, bytes: bytes.to_vec() };
                doc.metadata.psd_global_blocks.push((*b"8BIM", *b"lnk2", Arc::new(photocraft_io::linked::encode_linked_file(&linked))));
                smart_mut(doc, LayerId(id))?.source = SmartSource::Linked { path: linked.uuid };
                Ok(())
            })
            .unwrap();
        }
        // An explicit smart-object target must work while a pixel layer is selected.
        let raster = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
        s.authorize = Some(deny_contents_paths);
        let child = s.execute("layer.smartObjects.editContents", json!({"layer": id})).unwrap()["document"].as_u64().unwrap() as usize;
        assert_eq!(s.active().unwrap().doc.depth.bits(), depth as u32);
        assert!(s.active().unwrap().path.is_none());
        assert_eq!(s.documents()[0].active_layer, Some(LayerId(raster)));
        s.close(child).unwrap();
        let r = s.execute("layer.smartObjects.convertToLayers", json!({"layer": id})).unwrap();
        assert!(matches!(s.active().unwrap().doc.layer(LayerId(r["layer"].as_u64().unwrap())).unwrap().content, LayerContent::Raster(_)));
    }
}

fn contents_file() -> std::path::PathBuf {
    let doc = Document::with_background(
        "source",
        Size::new(4, 4),
        photocraft_doc::ColorMode::Rgb,
        photocraft_doc::SampleType::U8,
        photocraft_color::Color::rgb(1.0, 0.0, 0.0),
    );
    let path = std::env::temp_dir().join(format!("pcraft-contents-{}-{}.pcraft", std::process::id(), doc.id.0));
    std::fs::write(&path, encode_source(&doc).unwrap()).unwrap();
    path
}

#[test]
fn automation_smart_contents_refuses_disk_sources_before_changes() {
    let path = contents_file();
    let source_bytes = std::fs::read(&path).unwrap();
    for cmd in ["layer.smartObjects.editContents", "layer.smartObjects.convertToLayers"] {
        let mut s = session(8);
        paint(&mut s);
        let id = convert(&mut s);
        let sibling = s
            .edit("linked sources", |doc, _| {
                let source = SmartSource::Linked { path: path.to_string_lossy().into_owned() };
                smart_mut(doc, LayerId(id))?.source = source.clone();
                let mut cache = Surface::new(doc.pixel_format());
                cache.fill_rect(Rect::new(0, 0, 4, 4), &[0.0, 0.0, 1.0, 1.0]);
                let sibling = Layer::new("linked sibling", LayerContent::Smart(SmartObject::new(source, Affine::IDENTITY, Some(cache))));
                let sid = sibling.id;
                doc.layers.insert(0, sibling);
                Ok(sid)
            })
            .unwrap();
        s.authorize = Some(deny_contents_paths);
        let before = s.active().unwrap().doc.clone();
        let revision = s.active().unwrap().revision;
        let history = s.active().unwrap().history.entries().len();
        let error = s.execute(cmd, json!({"layer": id})).unwrap_err().to_string();
        // The error Debug-quotes the path (doubled backslashes on Windows), so match its file name.
        assert!(error.contains("source path refused") && error.contains(path.file_name().unwrap().to_str().unwrap()), "{error}");
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert_eq!((s.documents().len(), s.active().unwrap().revision, s.active().unwrap().history.entries().len()), (1, revision, history));

        // Human commands retain their linked-file behavior. Returning to an already-open child
        // reads no source, and saving it updates only its parent layer, never a linked sibling.
        s.authorize = None;
        let r = s.execute(cmd, json!({"layer": id})).unwrap();
        if cmd.ends_with("editContents") {
            let child = r["document"].as_u64().unwrap() as usize;
            s.set_active(0);
            s.authorize = Some(deny_contents_paths);
            let again = s.execute(cmd, json!({"layer": id})).unwrap();
            assert_eq!(again["document"], r["document"]);
            assert_eq!(s.documents().len(), 2);
            s.execute("image.adjustments.invert", json!({})).unwrap();
            s.execute("layer.smartObjects.saveContents", json!({})).unwrap();
            assert!(!s.active().unwrap().is_dirty());
            let parent = &s.documents()[0].doc;
            assert!(matches!(smart(parent, LayerId(id)).unwrap().source, SmartSource::Embedded { .. }));
            assert_eq!(smart(parent, sibling).unwrap().cache.as_ref().unwrap().rgba(1, 1), [0.0, 0.0, 1.0, 1.0]);
            s.close(child).unwrap();
        } else {
            assert_eq!(s.active().unwrap().doc.layer(LayerId(r["layer"].as_u64().unwrap())).unwrap().surface().unwrap().rgba(-5, 6), [1.0, 0.0, 0.0, 1.0]);
        }
    }
    assert_eq!(std::fs::read(&path).unwrap(), source_bytes, "committing contents does not overwrite the linked file");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn automation_smart_contents_checks_nested_reads_at_nontrivial_placements() {
    let path = contents_file();
    for (scale, psd_uuid, allowed) in [(1.0, false, true), (1.5, false, false), (1.5, true, true)] {
        let mut inner = Document::new("inner", Size::new(4, 4), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        let source_path = if psd_uuid {
            let linked =
                photocraft_io::linked::LinkedFile { uuid: "nested-id".into(), file_name: "source.pcraft".into(), bytes: std::fs::read(&path).unwrap() };
            inner.metadata.psd_global_blocks.push((*b"8BIM", *b"lnk2", Arc::new(photocraft_io::linked::encode_linked_file(&linked))));
            linked.uuid
        } else {
            path.to_string_lossy().into_owned()
        };
        let mut cache = Surface::new(inner.pixel_format());
        cache.fill_rect(Rect::new(0, 0, 4, 4), &[0.0, 0.0, 1.0, 1.0]);
        let nested =
            Layer::new("nested", LayerContent::Smart(SmartObject::new(SmartSource::Linked { path: source_path }, Affine::IDENTITY, Some(cache.clone()))));
        inner.layers.push(Layer::group("group", vec![nested]));
        let mut parent = Document::new("parent", Size::new(8, 8), inner.mode, inner.depth);
        parent.layers.push(Layer::new(
            "outer",
            LayerContent::Smart(SmartObject::new(
                SmartSource::Embedded { file_name: "inner.pcraft".into(), bytes: Arc::new(encode_source(&inner).unwrap()) },
                Affine { m: [scale, 0.0, 0.0, scale, 0.0, 0.0] },
                Some(cache),
            )),
        ));
        let mut s = Session::new();
        s.add_document(parent, None);
        s.authorize = Some(deny_contents_paths);
        let before = s.active().unwrap().doc.clone();
        let r = s.execute("layer.smartObjects.convertToLayers", json!({}));
        if !allowed {
            assert!(r.unwrap_err().to_string().contains("source path refused"));
            assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
            assert_eq!(s.active().unwrap().revision, 1);
        } else {
            r.unwrap();
            let sm = s
                .active()
                .unwrap()
                .doc
                .walk()
                .into_iter()
                .find_map(|(_, _, l)| match &l.content {
                    LayerContent::Smart(sm) => Some(sm),
                    _ => None,
                })
                .unwrap();
            assert_eq!(sm.cache.as_ref().unwrap().rgba(1, 1), if psd_uuid { [1.0, 0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0, 1.0] });
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn automation_smart_contents_malformed_embedded_sources_leave_documents_unchanged() {
    for cmd in ["layer.smartObjects.editContents", "layer.smartObjects.convertToLayers"] {
        let mut s = session(8);
        paint(&mut s);
        let id = convert(&mut s);
        s.edit("bad source", |doc, _| {
            smart_mut(doc, LayerId(id))?.source = SmartSource::Embedded { file_name: "broken.psd".into(), bytes: Arc::new(b"8BPS".to_vec()) };
            Ok(())
        })
        .unwrap();
        s.authorize = Some(deny_contents_paths);
        let before = s.active().unwrap().doc.clone();
        let revision = s.active().unwrap().revision;
        assert!(s.execute(cmd, json!({})).is_err());
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert_eq!((s.documents().len(), s.active().unwrap().revision), (1, revision));
    }
}

#[test]
fn inspect_reports_the_smart_source() {
    // A PSD placed layer: its file in the global `lnk2` block, the layer linked to its uuid.
    let inner = Document::with_background(
        "in",
        Size::new(4, 4),
        photocraft_color::ColorMode::Rgb,
        photocraft_color::SampleType::U8,
        photocraft_color::Color::rgba(0.0, 1.0, 0.0, 1.0),
    );
    let png = photocraft_io::export(&inner, "png", &Default::default()).unwrap().bytes;
    let lnk2 = photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile { uuid: "uuid-1".into(), file_name: "in.png".into(), bytes: png });
    let mut s = session(8);
    s.edit("place", |doc, active| {
        doc.metadata.psd_global_blocks.push((*b"8BIM", *b"lnk2", Arc::new(lnk2)));
        let sm = SmartObject::new(SmartSource::Linked { path: "uuid-1".into() }, Affine::IDENTITY, None);
        *active = Some(doc.insert_above(*active, Layer::new("front", LayerContent::Smart(sm))));
        Ok(())
    })
    .unwrap();
    let source = |s: &Session| crate::inspect::layer(s.active().unwrap().doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap())["smartSource"].clone();
    let front = source(&s);
    assert_eq!(front, json!({"kind": "linked", "path": "uuid-1"}));
    // The document's info knows the file is embedded in the PSD (a `liFD` item) and says so,
    // with its file name; the uuid stays its key.
    let info = crate::inspect::document(s.active().unwrap());
    assert_eq!(info["layers"][0]["smartSource"], json!({"kind": "embedded", "fileName": "in.png", "uuid": "uuid-1"}));
    // A duplicate shares the file; New Smart Object via Copy gets its own.
    s.execute("layer.duplicate", json!({})).unwrap();
    assert_eq!(source(&s), front, "a duplicate shares its source");
    s.execute("layer.smartObjects.newSmartObjectViaCopy", json!({})).unwrap();
    assert_eq!(source(&s), json!({"kind": "embedded", "fileName": "in.png"}), "via copy embeds its own");

    let mut s = session(8);
    paint(&mut s);
    convert(&mut s);
    assert_eq!(source(&s)["kind"], "embedded");
    assert!(source(&s)["fileName"].is_string());
}

#[test]
fn displace_smart_filter_reads_its_map_path_on_re_render() {
    let dir = std::env::temp_dir().join(format!("pcraft-displace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A white map displaces by the full scale; mid-grey would not move anything.
    let white = Document::with_background(
        "map",
        Size::new(W as u32, H as u32),
        photocraft_color::ColorMode::Rgb,
        photocraft_color::SampleType::U8,
        photocraft_color::Color::rgba(1.0, 1.0, 1.0, 1.0),
    );
    let map = dir.join("white.png");
    std::fs::write(&map, photocraft_io::export(&white, "png", &Default::default()).unwrap().bytes).unwrap();
    for depth in DEPTHS {
        let mut s = session(depth);
        paint(&mut s);
        convert(&mut s);
        let before = flat(&s);
        s.execute("filter.distort.displace", json!({"horizontal": 8, "vertical": 8, "mapPath": map.to_str().unwrap()})).unwrap();
        assert!(max_diff(&before, &flat(&s)) > 0.1, "the smart filter displaced the pixels at {depth}-bit");
        // A later re-render (here: hide and show the filter) reads the map again.
        s.execute("layer.smartFilter.setVisible", json!({"index": 0, "visible": false})).unwrap();
        assert_eq!(max_diff(&before, &flat(&s)), 0.0, "hidden at {depth}-bit");
        s.execute("layer.smartFilter.setVisible", json!({"index": 0, "visible": true})).unwrap();
        assert!(max_diff(&before, &flat(&s)) > 0.1, "shown again at {depth}-bit");
    }
    std::fs::remove_dir_all(&dir).ok();
}
