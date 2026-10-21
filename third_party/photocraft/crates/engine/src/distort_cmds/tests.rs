use super::*;

/// 80×60 document, a new layer with a patterned 50×30 block at (10, 10).
fn session(depth: u64) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 80, "height": 60, "depth": depth})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("paint", |doc, active| {
        let l = doc.layer_mut(active.unwrap()).unwrap();
        let surf = l.surface_mut().unwrap();
        for y in 10..40 {
            for x in 10..60 {
                let v = ((x / 3 + y / 3) % 2) as f32;
                surf.write_pixel(x, y, &[v, 1.0 - v, (x as f32) / 80.0, 1.0]);
            }
        }
        Ok(())
    })
    .unwrap();
    s
}

fn active_surface(s: &Session) -> Surface {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().clone()
}

fn worst(a: &Surface, b: &Surface, r: Rect) -> f32 {
    a.read_region(r).iter().zip(b.read_region(r)).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn smart(s: &Session) -> photocraft_doc::SmartObject {
    let st = s.active().unwrap();
    match &st.doc.layer(st.active_layer.unwrap()).unwrap().content {
        LayerContent::Smart(sm) => sm.clone(),
        _ => panic!("not smart"),
    }
}

#[test]
fn liquify_identity_stroke_displacement_reconstruct_and_undo() {
    for depth in [8u64, 16, 32] {
        let mut s = session(depth);
        let before = active_surface(&s);
        let r = s.execute(LIQUIFY, json!({})).unwrap();
        assert_eq!(r["changed"], false);
        let r = s.execute(LIQUIFY, json!({"strokes": [{"tool": "forwardWarp", "size": 30, "pressure": 0, "points": [[30, 25], [45, 25]]}]})).unwrap();
        assert_eq!(r["changed"], false, "zero pressure is a no-op");
        let hist = s.active().unwrap().history.past_len();
        let stroke = json!({"tool": "forwardWarp", "size": 30, "points": [[30, 25], [42, 25]]});
        let r = s.execute(LIQUIFY, json!({"strokes": [stroke]})).unwrap();
        assert_eq!(r["changed"], true);
        assert!(r["maxDisplacement"].as_f64().unwrap() > 4.0);
        assert_eq!(s.active().unwrap().history.past_len(), hist + 1, "one history step");
        let after = active_surface(&s);
        // The pattern moved right: the output at (42, 25) shows what was a few px to the left.
        let d = (0..12).map(|k| worst_px(&after.pixel(42, 25), &before.pixel(42 - k, 25))).enumerate().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        assert!(d >= 3, "{depth}: displaced by {d}");
        assert_ne!(after, before);
        // Reconstruct All in the same stroke list restores the original.
        let mut s2 = session(depth);
        s2.execute(LIQUIFY, json!({"strokes": [stroke, {"tool": "reconstructAll", "amount": 100}]})).unwrap();
        assert!(worst(&active_surface(&s2), &before, Rect::new(0, 0, 80, 60)) <= 1.0 / 255.0);
        s.undo();
        assert_eq!(active_surface(&s), before);
        assert!(s.execute(LIQUIFY, json!({"strokes": [{"tool": "melt", "points": [[1, 1]]}]})).is_err());
        assert!(s.execute(LIQUIFY, json!({"strokes": [{"points": [[1]]}]})).is_err());
        assert!(s.execute(LIQUIFY, json!({"strokes": [], "meshSize": 0})).is_err());
    }
}

fn worst_px(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn liquify_respects_the_selection() {
    let mut s = session(8);
    let before = active_surface(&s);
    s.execute("select.all", json!({})).ok();
    s.edit("sel", |doc, _| {
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 35, 60), &[1.0]);
        doc.selection = Some(sel);
        Ok(())
    })
    .unwrap();
    s.execute(LIQUIFY, json!({"strokes": [{"tool": "bloat", "size": 60, "points": [[35, 25], [35, 25], [35, 25], [35, 25]]}]})).unwrap();
    let after = active_surface(&s);
    assert_eq!(after.read_region(Rect::new(36, 0, 80, 60)), before.read_region(Rect::new(36, 0, 80, 60)), "unselected pixels stay");
    assert_ne!(after.read_region(Rect::new(10, 10, 35, 40)), before.read_region(Rect::new(10, 10, 35, 40)));
}

#[test]
fn liquify_elapsed_time_is_wasm_safe() {
    let mut s = session(8);
    let r = s.execute(LIQUIFY, json!({"strokes": [{"tool": "forwardWarp", "size": 30, "points": [[30, 25], [42, 25]]}]})).unwrap();
    assert_eq!(r["changed"], true);
    let ms = r["ms"].as_f64().unwrap();
    assert!(ms.is_finite() && ms >= 0.0);
    #[cfg(target_arch = "wasm32")]
    assert_eq!(ms, 0.0, "the web build uses the wasm-safe stopwatch fallback");
}

#[test]
fn puppet_identity_translation_and_undo() {
    for depth in [8u64, 16, 32] {
        let mut s = session(depth);
        let before = active_surface(&s);
        assert_eq!(s.execute(PUPPET, json!({})).unwrap()["changed"], false);
        assert_eq!(s.execute(PUPPET, json!({"pins": [{"src": [20, 20], "dst": [20, 20]}]})).unwrap()["changed"], false);
        s.execute(PUPPET, json!({"pins": [{"src": [30, 25], "dst": [36, 29]}], "mode": "rigid", "interpolation": "bilinear"})).unwrap();
        let after = active_surface(&s);
        assert_eq!(after.content_bounds(), Rect::new(16, 14, 66, 44), "{depth}");
        assert!(
            worst(&after.convert(before.format()), &photocraft_algo::resample::translate_surface(&before, 6, 4), Rect::new(16, 14, 66, 44)) <= 1.0 / 255.0,
            "{depth}"
        );
        s.undo();
        assert_eq!(active_surface(&s), before);
        assert!(s.execute(PUPPET, json!({"pins": [{"src": [1, 1]}]})).is_err());
        assert!(s.execute(PUPPET, json!({"pins": [], "mode": "wobbly"})).is_err());
        assert!(s.execute(PUPPET, json!({"pins": [], "density": "lots"})).is_err());
    }
}

fn mask_at(s: &Session, x: i32, y: i32) -> f32 {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0)
}

/// [`session`] with a reveal-all mask that hides the stripes `stripes` (x ranges, y 10..40).
fn masked_session(depth: u64, stripes: &[(i32, i32)]) -> Session {
    let mut s = session(depth);
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    s.edit("mask", |doc, active| {
        let m = doc.layer_mut(active.unwrap()).unwrap().mask.as_mut().unwrap();
        for &(x0, x1) in stripes {
            m.surface.fill_rect(Rect::new(x0, 10, x1, 40), &[0.0]);
        }
        Ok(())
    })
    .unwrap();
    s
}

#[test]
fn puppet_moves_a_linked_mask_with_the_pixels() {
    // #4910: Puppet Warp warped only the pixels and left a linked mask behind. A one-pin rigid
    // warp moves the block by (6, 4); the stripe the mask hides (x 20..24) must move with it.
    for depth in [8u64, 16, 32] {
        let mut s = masked_session(depth, &[(20, 24)]);
        let before = s.active().unwrap().doc.clone();
        s.execute(PUPPET, json!({"pins": [{"src": [30, 25], "dst": [36, 29]}], "mode": "rigid", "interpolation": "bilinear"})).unwrap();
        for y in [16, 30, 42] {
            assert!(mask_at(&s, 27, y) < 0.01, "{depth}: the stripe moved to x 26..30 (y {y})");
            assert!(mask_at(&s, 21, y) > 0.99, "{depth}: and left x 20..24 (y {y})");
        }
        assert!(mask_at(&s, 27, 12) > 0.99, "{depth}: and down by 4");
        // Pixels and mask still line up: the same layer pixel is hidden as before the warp.
        let was = before.layer(s.active().unwrap().active_layer.unwrap()).unwrap().surface().unwrap().pixel(21, 26);
        assert!(worst_px(&active_surface(&s).pixel(27, 30), &was) <= 1.0 / 255.0, "{depth}");
        s.undo();
        assert!(mask_at(&s, 21, 30) < 0.01 && mask_at(&s, 27, 30) > 0.99, "{depth}: undo puts the mask back");
    }
}

#[test]
fn puppet_leaves_an_unlinked_mask_in_place() {
    let mut s = masked_session(8, &[(20, 24)]);
    s.execute("layer.layerMask.linked", json!({"linked": false})).unwrap();
    s.execute(PUPPET, json!({"pins": [{"src": [30, 25], "dst": [36, 29]}], "mode": "rigid"})).unwrap();
    assert!(mask_at(&s, 21, 30) < 0.01 && mask_at(&s, 27, 30) > 0.99);
    assert_eq!(active_surface(&s).content_bounds(), Rect::new(16, 14, 66, 44), "the pixels still moved");
}

#[test]
fn puppet_with_a_selection_moves_only_the_selected_part_of_a_linked_mask() {
    // The selection lifts the block's left part (x 10..30): the stripe hidden there moves with
    // it, the one outside (x 40..44) stays over its pixels.
    let mut s = masked_session(8, &[(20, 24), (40, 44)]);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 30, "height": 60})).unwrap();
    s.execute(PUPPET, json!({"pins": [{"src": [20, 25], "dst": [26, 29]}], "mode": "rigid"})).unwrap();
    assert!(mask_at(&s, 27, 30) < 0.01 && mask_at(&s, 21, 30) > 0.99, "the selected stripe moved");
    assert!(mask_at(&s, 41, 30) < 0.01 && mask_at(&s, 45, 30) > 0.99, "the other stayed");
    assert_eq!(active_surface(&s).pixel(41, 30)[3], 1.0, "over its pixels");
}

#[test]
fn puppet_with_an_expansion_that_empties_the_mesh_keeps_the_layer() {
    // #951: a pinned warp over an empty mesh panicked in the solver; it must not erase the layer
    // either (the rasterizer keeps only pixels inside the mesh).
    let mut s = session(8);
    let before = active_surface(&s);
    s.execute(PUPPET, json!({"pins": [{"src": [30, 25], "dst": [36, 29]}], "expansion": -200})).unwrap();
    assert_eq!(active_surface(&s).read_region(Rect::new(0, 0, 80, 60)), before.read_region(Rect::new(0, 0, 80, 60)));
}

#[test]
fn puppet_refuses_pins_that_stretch_the_content_too_far() {
    // #1019: a pin dragged to (2^20, 2^20) stretched the mesh over ~17.6 million tiles, and the
    // renderer allocated bookkeeping for every one before drawing anything.
    let far = json!({"pins": [{"src": [20, 20], "dst": [20, 20]}, {"src": [50, 30], "dst": [1048576, 1048576]}]});
    let full = Rect::new(0, 0, 80, 60);
    for depth in [8u64, 16, 32] {
        for selected in [false, true] {
            let mut s = session(depth);
            if selected {
                s.execute("select.rect", json!({"x": 10, "y": 10, "width": 30, "height": 30})).unwrap();
            }
            let (before, past) = (active_surface(&s), s.active().unwrap().history.past_len());
            let err = s.execute(PUPPET, far.clone()).unwrap_err();
            assert!(err.to_string().contains("too far"), "{depth} {selected}: {err}");
            assert_eq!(active_surface(&s).read_region(full), before.read_region(full), "{depth} {selected}: layer unchanged");
            assert_eq!(s.active().unwrap().history.past_len(), past, "{depth} {selected}: no history step");
        }
        // Controls: a rigid move still moves the content, and a pin 4000 px out (well inside the
        // budget) still warps.
        let mut s = session(depth);
        s.execute(PUPPET, json!({"pins": [{"src": [20, 20], "dst": [30, 25]}, {"src": [50, 30], "dst": [60, 35]}], "mode": "rigid"})).unwrap();
        assert_eq!(active_surface(&s).content_bounds(), Rect::new(20, 15, 70, 45), "{depth}");
        let mut s = session(depth);
        s.execute(PUPPET, json!({"pins": [{"src": [20, 20], "dst": [20, 20]}, {"src": [50, 30], "dst": [4050, 4030]}]})).unwrap();
        assert!(active_surface(&s).content_bounds().x1 > 1000, "{depth}");
    }
    // A smart object stores the filter; a warp too far to render leaves its pixels as they were.
    let mut s = session(8);
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    let before = active_surface(&s);
    s.execute(PUPPET_SMART, far).unwrap();
    assert_eq!(smart(&s).smart_filters.len(), 1);
    assert_eq!(active_surface(&s), before);
}

#[test]
fn puppet_two_pins_bend_the_block() {
    let mut s = session(8);
    // Hold the left end, lift the right end: the right side rises, the left stays.
    s.execute(PUPPET, json!({"pins": [{"src": [15, 25], "dst": [15, 25]}, {"src": [55, 25], "dst": [55, 10]}], "mode": "normal", "density": "more"})).unwrap();
    let after = active_surface(&s);
    assert!(after.pixel(12, 25)[3] > 0.5, "the held end stays");
    assert!(after.content_bounds().y0 < 6, "{:?}", after.content_bounds());
}

#[test]
fn puppet_mesh_follows_the_outline_so_separate_shapes_move_separately() {
    // #3105: the mesh was every grid cell touching opaque pixels, so two bars 24 px apart (more
    // than a mesh cell at Fewer Points) shared cells and a pin on one dragged the other along.
    // The mesh now follows the outline grown by the expansion: two meshes, and the unpinned one
    // stays put.
    for depth in [8u64, 16, 32] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 260, "height": 140, "depth": depth})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, active| {
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            for (y0, y1) in [(20, 60), (84, 120)] {
                for y in y0..y1 {
                    for x in 10..250 {
                        let v = ((x / 4 + y / 4) % 2) as f32;
                        surf.write_pixel(x, y, &[v, 1.0 - v, (y as f32) / 140.0, 1.0]);
                    }
                }
            }
            Ok(())
        })
        .unwrap();
        let before = active_surface(&s);
        s.execute(PUPPET, json!({"pins": [{"src": [130, 40], "dst": [130, 30]}], "density": "fewer", "expansion": 2, "interpolation": "bilinear"})).unwrap();
        let after = active_surface(&s).convert(before.format());
        let lower = Rect::new(0, 70, 260, 140);
        let still = worst(&after, &before, lower);
        assert!(still <= 1.0 / 255.0, "{depth}: the unpinned bar stays ({still})");
        assert!(
            worst(&after, &photocraft_algo::resample::translate_surface(&before, 0, -10), Rect::new(0, 0, 260, 60)) <= 1.0 / 255.0,
            "{depth}: the pinned bar moves"
        );
    }
}

#[test]
fn perspective_identity_known_homography_and_mask() {
    for depth in [8u64, 16, 32] {
        let mut s = session(depth);
        let before = active_surface(&s);
        let q = [[10.0, 10.0], [60.0, 10.0], [60.0, 40.0], [10.0, 40.0]];
        let r = s.execute(PERSPECTIVE, json!({"planes": [{"src": q, "dst": q}]})).unwrap();
        assert_eq!(r["changed"], false);
        // Shift the whole quad by (5, 3): a pure translation of the layer.
        let d = q.map(|p| [p[0] + 5.0, p[1] + 3.0]);
        s.execute(PERSPECTIVE, json!({"planes": [{"src": q, "dst": d}], "interpolation": "bilinear"})).unwrap();
        let after = active_surface(&s);
        assert_eq!(after.content_bounds(), Rect::new(15, 13, 65, 43));
        assert!(
            worst(&after.convert(before.format()), &photocraft_algo::resample::translate_surface(&before, 5, 3), Rect::new(15, 13, 65, 43)) <= 1.0 / 255.0,
            "{depth}"
        );
        s.undo();
        assert_eq!(active_surface(&s), before);
    }
    // A keystone: corners land where asked.
    let mut s = session(8);
    let q = [[10.0, 10.0], [60.0, 10.0], [60.0, 40.0], [10.0, 40.0]];
    let d = [[20.0, 10.0], [50.0, 10.0], [70.0, 50.0], [0.0, 50.0]];
    s.execute("layer.layerMask.revealAll", json!({})).ok();
    s.execute(PERSPECTIVE, json!({"planes": [{"src": q, "dst": d}]})).unwrap();
    let after = active_surface(&s);
    let b = after.content_bounds();
    assert!(b.x0 <= 1 && b.x1 >= 69 && b.y1 >= 49 && b.y0 >= 9, "{b:?}");
    assert!(after.pixel(15, 12)[3] < 0.1, "the top narrowed");
    assert!(s.execute(PERSPECTIVE, json!({})).is_err());
    assert!(s.execute(PERSPECTIVE, json!({"planes": [{"src": [[0, 0], [0, 0], [0, 0], [0, 0]], "dst": [[0, 0], [0, 0], [0, 0], [0, 0]]}]})).is_err());
    assert!(s.execute(PERSPECTIVE, json!({"planes": [{"src": q, "dst": d}], "straighten": "sideways"})).is_err());
}

#[test]
fn perspective_with_a_selection_moves_only_the_selected_part_of_a_linked_mask() {
    // #3083: the mask hides a stripe inside the selection (x 12..16) and one outside it (x 40..44).
    let mask_at = |s: &Session, x, y| {
        let st = s.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0)
    };
    for selected in [true, false] {
        let mut s = session(8);
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        s.edit("mask", |doc, active| {
            let m = doc.layer_mut(active.unwrap()).unwrap().mask.as_mut().unwrap();
            m.surface.fill_rect(Rect::new(12, 10, 16, 40), &[0.0]);
            m.surface.fill_rect(Rect::new(40, 10, 44, 40), &[0.0]);
            Ok(())
        })
        .unwrap();
        if selected {
            s.execute("select.rect", json!({"x": 0, "y": 0, "width": 30, "height": 60})).unwrap();
        }
        let q = [[0.0, 0.0], [30.0, 0.0], [30.0, 60.0], [0.0, 60.0]];
        let d = q.map(|p| [p[0] + 4.0, p[1]]);
        s.execute(PERSPECTIVE, json!({"planes": [{"src": q, "dst": d}], "interpolation": "bilinear"})).unwrap();
        // The selected stripe moves with the pixels either way.
        assert!(mask_at(&s, 17, 25) < 0.01 && mask_at(&s, 13, 25) > 0.99, "selected={selected}");
        if selected {
            // The unselected stripe stays over the unselected pixels.
            assert!(mask_at(&s, 43, 25) < 0.01 && mask_at(&s, 47, 25) > 0.99);
            assert_eq!(active_surface(&s).pixel(43, 25)[3], 1.0);
        } else {
            // Without a selection the whole mask moves with the layer.
            assert!(mask_at(&s, 47, 25) < 0.01 && mask_at(&s, 43, 25) > 0.99);
        }
    }
}

#[test]
fn smart_objects_store_distortions_as_smart_filters_losslessly() {
    for depth in [8u64, 16, 32] {
        let mut s = session(depth);
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        let original = active_surface(&s);
        assert!(s.is_enabled(PUPPET_SMART) && s.is_enabled(PERSPECTIVE_SMART));
        s.execute(LIQUIFY, json!({"strokes": [{"tool": "twirlCw", "size": 40, "points": [[35, 25], [35, 25], [35, 25]]}]})).unwrap();
        let liq = active_surface(&s);
        assert_ne!(liq, original);
        s.execute(PUPPET_SMART, json!({"pins": [{"src": [30, 25], "dst": [34, 25]}], "mode": "rigid"})).unwrap();
        let q = [[10.0, 10.0], [60.0, 10.0], [60.0, 40.0], [10.0, 40.0]];
        s.execute(PERSPECTIVE_SMART, json!({"planes": [{"src": q, "dst": [[12.0, 10.0], [58.0, 10.0], [60.0, 40.0], [10.0, 40.0]]}]})).unwrap();
        let sm = smart(&s);
        let ids: Vec<&str> = sm.smart_filters.iter().map(|f| f.command.as_str()).collect();
        assert_eq!(ids, [LIQUIFY, PUPPET_SMART, PERSPECTIVE_SMART]);
        // Deleting the top two filters re-renders exactly the liquified state; deleting all
        // gives back the original pixels (nothing accumulated).
        s.execute("layer.smartFilter.delete", json!({})).unwrap();
        s.execute("layer.smartFilter.delete", json!({})).unwrap();
        assert_eq!(active_surface(&s), liq, "{depth}");
        s.execute("layer.smartFilter.delete", json!({})).unwrap();
        assert_eq!(active_surface(&s), original, "{depth}");
        s.undo();
        assert_eq!(active_surface(&s), liq);
    }
}

#[test]
fn enabled_states() {
    let mut s = session(8);
    assert!(s.is_enabled(LIQUIFY) && s.is_enabled(PUPPET) && s.is_enabled(PERSPECTIVE));
    assert!(!s.is_enabled(PUPPET_SMART) && !s.is_enabled(PERSPECTIVE_SMART));
    s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    assert!(!s.is_enabled(PUPPET) && !s.is_enabled(PERSPECTIVE));
    let mut empty = Session::new();
    assert!(!empty.is_enabled(LIQUIFY));
    assert!(empty.execute(PUPPET, json!({"pins": []})).is_err());
}

#[test]
fn background_layer_is_floated_by_geometric_warps() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "depth": 8})).unwrap();
    s.execute(PUPPET, json!({"pins": [{"src": [20, 15], "dst": [25, 15]}], "mode": "rigid"})).unwrap();
    let st = s.active().unwrap();
    let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
    assert_eq!(l.name, "Layer 0");
    assert!(l.surface().unwrap().format().alpha);
    assert!(l.surface().unwrap().pixel(2, 15)[3] < 0.01, "exposed area is transparent");
}

#[test]
fn liquify_with_a_far_off_stroke_point_returns() {
    // #937: the issue's request walked ~3.3e14 dab steps along a segment to (1e15, 0).
    for depth in [8u64, 16, 32] {
        let mut s = session(depth);
        let r = s.execute(LIQUIFY, json!({"strokes": [{"tool": "forwardWarp", "size": 30, "points": [[0, 0], [1e15, 0]]}]})).unwrap();
        assert_eq!(r["changed"], true, "{depth}: the start of the stroke is on the canvas");
        let mut s = session(depth);
        let before = active_surface(&s);
        s.execute(LIQUIFY, json!({"strokes": [{"tool": "bloat", "size": 30, "points": [[5000, 5000], [1e15, 1e15]]}]})).unwrap();
        assert_eq!(active_surface(&s).read_region(Rect::new(0, 0, 80, 60)), before.read_region(Rect::new(0, 0, 80, 60)), "{depth}: off-canvas only");
    }
}

/// The three distortions on the active smart object, each with a visible change.
fn smart_distortions() -> [(&'static str, Value); 3] {
    let q = [[10.0, 10.0], [60.0, 10.0], [60.0, 40.0], [10.0, 40.0]];
    [
        (LIQUIFY, json!({"strokes": [{"tool": "twirlCw", "size": 40, "points": [[35, 25], [35, 25], [35, 25]]}]})),
        (PUPPET_SMART, json!({"pins": [{"src": [30, 25], "dst": [34, 25]}], "mode": "rigid"})),
        (PERSPECTIVE_SMART, json!({"planes": [{"src": q, "dst": [[12.0, 10.0], [58.0, 10.0], [60.0, 40.0], [10.0, 40.0]]}]})),
    ]
}

#[test]
fn locked_smart_objects_refuse_distortions() {
    // As filters do: a pixel or full lock (the layer's own or its group's) refuses the smart
    // filter, leaving the layer and the history as they were.
    for lock in [json!({"all": true}), json!({"pixels": true})] {
        for on_group in [false, true] {
            for (cmd, p) in smart_distortions() {
                let mut s = session(8);
                s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
                let layer = s.active().unwrap().active_layer.unwrap();
                if on_group {
                    s.execute("layer.new.groupFromLayers", json!({})).unwrap();
                }
                s.execute("layer.lockLayers", lock.clone()).unwrap();
                s.execute("layer.select", json!({"layer": layer.0})).unwrap();
                let locks = s.active().unwrap().doc.effective_locks(layer);
                assert!(locks.all || locks.pixels, "the lock reaches the smart object ({lock}, group: {on_group})");
                let steps = s.active().unwrap().history.past_len();
                let before = s.active().unwrap().doc.layer(layer).unwrap().clone();
                let r = s.execute(cmd, p);
                assert!(r.is_err(), "{cmd} edited a smart object locked with {lock} (group: {on_group}): {r:?}");
                let st = s.active().unwrap();
                let LayerContent::Smart(sm) = &st.doc.layer(layer).unwrap().content else { panic!("not smart") };
                assert!(sm.smart_filters.is_empty(), "{cmd} {lock} group: {on_group}");
                assert_eq!(st.doc.layer(layer).unwrap().surface(), before.surface(), "{cmd} {lock} group: {on_group}");
                assert_eq!(st.history.past_len(), steps, "{cmd} {lock} group: {on_group}: no history step");
            }
        }
    }
}

#[test]
fn unlocked_or_position_locked_smart_objects_take_distortions() {
    // Filters don't check the position lock, so neither do the smart-filter distortions.
    for lock in [None, Some(json!({"position": true})), Some(json!({"transparency": true}))] {
        for (cmd, p) in smart_distortions() {
            let mut s = session(8);
            s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
            if let Some(lock) = &lock {
                s.execute("layer.lockLayers", lock.clone()).unwrap();
            }
            let steps = s.active().unwrap().history.past_len();
            s.execute(cmd, p).unwrap();
            assert_eq!(smart(&s).smart_filters.len(), 1, "{cmd} {lock:?}");
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "{cmd} {lock:?}");
        }
    }
}
