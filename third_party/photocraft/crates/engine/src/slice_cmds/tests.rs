use super::*;
use photocraft_doc::LayerId;
use photocraft_doc::effects::Effect;

fn session(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 120, "height": 90, "depth": depth, "name": "site"})).unwrap();
    s
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn add_square(s: &mut Session, r: Rect) -> LayerId {
    s.edit("sq", |doc, active| {
        let fmt = doc.pixel_format();
        let mut l = Layer::raster(doc.next_layer_name("Layer"), fmt);
        l.surface_mut().unwrap().fill_rect(r, &photocraft_raster::from_rgba(&fmt, [1.0, 0.0, 0.0, 1.0]));
        let id = doc.insert_above(None, l);
        *active = Some(id);
        Ok(id)
    })
    .unwrap()
}

#[test]
fn user_slices_and_options() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let r = s.execute("slice.new", json!({"rect": [20, 10, 40, 20], "name": "logo", "url": "https://example.org", "alt": "Logo"})).unwrap();
        let id = r["slice"].as_u64().unwrap() as u32;
        assert_eq!(r["number"], json!(3), "top band, left cell, then the slice");
        let l = s.execute("slice.list", json!({})).unwrap();
        let all = l["slices"].as_array().unwrap();
        assert_eq!(all.len(), 5);
        let me = all.iter().find(|v| v["id"] == json!(id)).unwrap();
        assert_eq!(me["name"], "logo");
        assert_eq!(me["rect"], json!([20, 10, 40, 20]));
        assert_eq!(me["origin"], "user");
        assert!(all.iter().filter(|v| v["origin"] == "auto").count() == 4);
        // Slice Options by number; then undo.
        s.execute("slice.set", json!({"number": 3, "kind": "noImage", "cellText": "hi", "background": "#ff0000"})).unwrap();
        let sl = doc(&s).slices.get(id).unwrap();
        assert_eq!(sl.kind, SliceKind::NoImage);
        assert_eq!(sl.background, Some([255, 255, 0, 0]));
        assert!(s.undo());
        assert_eq!(doc(&s).slices.get(id).unwrap().kind, SliceKind::Image);
        assert!(s.execute("slice.set", json!({"slice": id, "kind": "bogus"})).is_err());
        assert!(s.execute("slice.new", json!({"rect": [500, 500, 10, 10]})).is_err(), "outside the canvas");
    }
}

#[test]
fn rect_param_rejects_corner_overflow() {
    assert_eq!(rect_param(&json!({"rect": [1, 2, 3, 4]})), Some(Rect::new(1, 2, 4, 6)));
    assert_eq!(rect_param(&json!({"x": 1, "y": 2, "width": 3, "height": 4})), Some(Rect::new(1, 2, 4, 6)));
    assert_eq!(rect_param(&json!({"rect": [i32::MAX, 0, 1, 1]})), None);
    assert_eq!(rect_param(&json!({"x": 0, "y": i32::MAX, "width": 1, "height": 1})), None);

    let mut s = session(8);
    assert!(s.execute("slice.new", json!({"rect": [i32::MAX, 0, 1, 1]})).is_err());
    let id = s.execute("slice.new", json!({"rect": [1, 1, 8, 8]})).unwrap()["slice"].as_u64().unwrap();
    assert!(s.execute("slice.set", json!({"slice": id, "rect": [0, i32::MAX, 1, 1]})).is_err());
}

#[test]
fn promote_auto_slice_and_divide() {
    let mut s = session(8);
    s.execute("slice.new", json!({"rect": [0, 0, 60, 45]})).unwrap();
    // An auto slice promoted through Slice Options keeps one history step.
    let before = s.active().unwrap().history.past_len();
    let r = s.execute("slice.set", json!({"number": 2, "name": "right"})).unwrap();
    assert_eq!(r["origin"], "user");
    assert_eq!(r["name"], "right");
    assert_eq!(s.active().unwrap().history.past_len(), before + 1);
    assert_eq!(doc(&s).slices.list.len(), 2);
    let id = doc(&s).slices.list[0].id;
    let r = s.execute("slice.divide", json!({"slice": id, "horizontal": 3, "vertical": 2})).unwrap();
    assert_eq!(r["count"], 6);
    assert_eq!(doc(&s).slices.list.len(), 7);
    let total: u64 = doc(&s).slices.list.iter().take(6).map(|s| u64::from(s.rect.width()) * u64::from(s.rect.height())).sum();
    assert_eq!(total, 60 * 45);
    assert!(s.execute("slice.divide", json!({"slice": id})).is_err());
}

#[test]
fn divide_wide_slice_preserves_exact_edges_and_history() {
    for depth in [8, 16, 32] {
        for rect in
            [Rect::new(-2_000_000_000, -1_000_000_000, 2_000_000_000, 1_000_000_000), Rect::new(-1_000_000_000, -1_000_000_000, 1_000_000_000, 1_000_000_000)]
        {
            let mut s = session(depth);
            s.edit("loaded wide slice", |doc, _| {
                doc.slices.list.push(Slice { id: 1, rect, name: "wide".into(), ..Default::default() });
                Ok(())
            })
            .unwrap();
            let before = doc(&s).slices.clone();
            let past = s.active().unwrap().history.past_len();
            let result = s.execute("slice.divide", json!({"slice": 1, "horizontal": 3, "vertical": 3})).unwrap();
            assert_eq!(result["count"], 9);
            assert_eq!(s.active().unwrap().history.past_len(), past + 1);
            let parts = &doc(&s).slices.list;
            assert_eq!(parts.len(), 9);
            assert_eq!(parts[0].name, "wide");
            for row in parts.as_chunks::<3>().0 {
                assert_eq!(row[0].rect.x0, rect.x0);
                assert_eq!(row[2].rect.x1, rect.x1);
                assert_eq!(row.iter().map(|s| u64::from(s.rect.width())).sum::<u64>(), u64::from(rect.width()));
                for pair in row.windows(2) {
                    assert_eq!(pair[0].rect.x1, pair[1].rect.x0);
                    assert_eq!(pair[0].rect.y0, pair[1].rect.y0);
                    assert_eq!(pair[0].rect.y1, pair[1].rect.y1);
                }
            }
            assert_eq!(parts[0].rect.y0, rect.y0);
            assert_eq!(parts[8].rect.y1, rect.y1);
            for i in 0..6 {
                assert_eq!(parts[i].rect.y1, parts[i + 3].rect.y0);
            }
            let divided = doc(&s).slices.clone();
            assert!(s.undo());
            assert_eq!(doc(&s).slices, before);
            assert!(s.redo());
            assert_eq!(doc(&s).slices, divided);
        }
    }
}

#[test]
fn slices_from_guides_and_clear() {
    let mut s = session(8);
    s.edit("guides", |d, _| {
        d.guides.vertical = vec![40.0, 80.0];
        d.guides.horizontal = vec![30.0];
        Ok(())
    })
    .unwrap();
    s.slice_from_guides_check();
    assert!(s.is_enabled("view.clearSlices"));
    let r = s.execute("view.clearSlices", json!({})).unwrap();
    assert_eq!(r["cleared"], 6);
    assert!(doc(&s).slices.is_empty());
    assert!(!s.is_enabled("view.clearSlices"));
}

impl Session {
    fn slice_from_guides_check(&mut self) {
        let r = self.execute("slice.fromGuides", json!({})).unwrap();
        assert_eq!(r["slices"], 6);
        let rects: Vec<Rect> = self.active().unwrap().doc.slices.list.iter().map(|s| s.rect).collect();
        assert!(rects.contains(&Rect::new(40, 30, 80, 90)));
    }
}

#[test]
fn lock_slices_blocks_editing() {
    let mut s = session(8);
    s.execute("slice.new", json!({"rect": [0, 0, 10, 10]})).unwrap();
    assert_eq!(s.execute("view.lockSlices", json!({})).unwrap()["locked"], true);
    assert!(!s.is_enabled("slice.new"));
    assert!(!s.is_enabled("view.clearSlices"));
    assert!(s.execute("slice.new", json!({"rect": [0, 0, 10, 10]})).is_err());
    s.execute("view.lockSlices", json!({"on": false})).unwrap();
    assert!(s.is_enabled("slice.new"));
}

#[test]
fn layer_based_slice_follows_the_layer_and_its_effects() {
    for depth in [8, 16] {
        let mut s = session(depth);
        assert!(!s.is_enabled("layer.newLayerBasedSlice"), "Background layer");
        let id = add_square(&mut s, Rect::new(10, 10, 30, 20));
        let r = s.execute("layer.newLayerBasedSlice", json!({})).unwrap();
        assert_eq!(r["rect"], json!([10, 10, 20, 10]));
        assert!(!s.is_enabled("layer.newLayerBasedSlice"), "one per layer");
        let sid = r["slice"].as_u64().unwrap() as u32;
        // Moving the layer moves the slice.
        s.execute("layer.translate", json!({"layer": id.0, "dx": 5, "dy": 7})).unwrap();
        assert_eq!(doc(&s).slices.get(sid).unwrap().rect, Rect::new(15, 17, 35, 27));
        // A drop shadow grows it.
        s.edit("fx", |d, _| {
            let l = d.layer_mut(id).unwrap();
            l.effects.enabled = true;
            l.effects.items.push(Effect::default_drop_shadow());
            Ok(())
        })
        .unwrap();
        s.execute("slice.list", json!({})).unwrap();
        let grown = doc(&s).slices.get(sid).unwrap().rect;
        assert!(grown.contains_rect(&Rect::new(15, 17, 35, 27)) && grown != Rect::new(15, 17, 35, 27), "{grown:?}");
        // Deleting the layer deletes its slice.
        s.execute("layer.delete", json!({"layer": id.0})).unwrap();
        assert!(doc(&s).slices.get(sid).is_none());
    }
}

#[test]
fn layer_slice_outsets_saturate_when_a_layer_moves_near_coordinate_limits() {
    for offset in [i32::MIN + 100, i32::MAX - 100] {
        let mut s = session(8);
        let layer = add_square(&mut s, Rect::new(10, 10, 30, 20));
        let slice = s.execute("layer.newLayerBasedSlice", json!({})).unwrap()["slice"].as_u64().unwrap() as u32;
        s.execute("slice.set", json!({"slice": slice, "outsets": [10_000, 10_000, 10_000, 10_000]})).unwrap();
        let before = doc(&s).slices.clone();
        s.execute("layer.translate", json!({"layer": layer.0, "dx": offset, "dy": offset})).unwrap();
        let moved = doc(&s).slices.get(slice).unwrap().rect;
        let content = layer_bounds(doc(&s).layer(layer).unwrap());
        assert_eq!(moved, content.inflate(10_000));
        assert!(moved.contains_rect(&content));
        s.execute("slice.list", json!({})).unwrap();
        assert_eq!(doc(&s).slices.get(slice).unwrap().rect, moved);
        assert!(s.undo());
        assert_eq!(doc(&s).slices, before);
        assert!(s.redo());
        s.execute("slice.list", json!({})).unwrap();
        assert_eq!(doc(&s).slices.get(slice).unwrap().rect, moved);
    }
}

#[test]
fn slices_round_trip_through_psd_and_pcraft() {
    let mut s = session(8);
    add_square(&mut s, Rect::new(50, 40, 70, 60));
    s.execute("layer.newLayerBasedSlice", json!({})).unwrap();
    s.execute("slice.new", json!({"rect": [5, 5, 30, 20], "name": "hero", "url": "https://example.org", "alt": "Hero", "target": "_blank"})).unwrap();
    let d = doc(&s).clone();
    for ext in ["psd", "pcraft"] {
        let bytes = photocraft_io::export(&d, &format!("x.{ext}"), &Default::default()).unwrap().bytes;
        let back = photocraft_io::import(&format!("x.{ext}"), &bytes).unwrap().document;
        assert_eq!(back.slices.list.len(), 2, "{ext}");
        let hero = back.slices.list.iter().find(|s| s.name == "hero").unwrap();
        assert_eq!((hero.url.as_str(), hero.alt.as_str(), hero.target.as_str()), ("https://example.org", "Hero", "_blank"));
        assert_eq!(hero.rect, Rect::new(5, 5, 35, 25));
        let ls = back.slices.list.iter().find(|s| s.origin == SliceOrigin::Layer).unwrap();
        let layer = back.layer(ls.layer.unwrap()).unwrap();
        assert_eq!(layer.name, "Layer 1", "{ext}: the slice still names its layer");
        assert_eq!(ls.rect, Rect::new(50, 40, 70, 60));
    }
}

#[test]
fn exhausted_slice_ids_fail_without_mutating_document_or_history() {
    let mut s = session(8);
    s.edit("loaded max slice", |doc, _| {
        doc.slices.list.push(Slice { id: 1, rect: Rect::new(10, 10, 20, 20), ..Default::default() });
        doc.slices.list.push(Slice { id: u32::MAX, rect: Rect::new(10, 10, 20, 20), ..Default::default() });
        Ok(())
    })
    .unwrap();
    let past = s.active().unwrap().history.past_len();
    let revision = s.active().unwrap().revision;

    for (command, params) in [
        ("slice.new", json!({"rect": [30, 30, 10, 10]})),
        ("slice.divide", json!({"slice": 1, "horizontal": 1, "vertical": 2})),
        ("slice.promote", json!({"number": 1})),
    ] {
        assert!(s.execute(command, params).unwrap_err().to_string().contains("slice id space exhausted"), "{command}");
        assert_eq!(doc(&s).slices.list.len(), 2, "{command}");
        assert!(doc(&s).slices.get(u32::MAX).is_some(), "{command}");
        assert_eq!(s.active().unwrap().history.past_len(), past, "{command}");
        assert_eq!(s.active().unwrap().revision, revision, "{command}");
    }
}

#[test]
fn rejected_slice_options_change_nothing() {
    // #910: an invalid option on an auto slice used to promote it before failing.
    let mut s = session(8);
    let past = s.active().unwrap().history.past_len();
    assert!(s.execute("slice.set", json!({"number": 1, "kind": "bogus"})).is_err());
    assert!(doc(&s).slices.is_empty());
    assert_eq!(s.active().unwrap().history.past_len(), past);
    // A user slice moved off the canvas is rejected without committing the move.
    let id = s.execute("slice.new", json!({"rect": [0, 0, 10, 10]})).unwrap()["slice"].as_u64().unwrap() as u32;
    let past = s.active().unwrap().history.past_len();
    assert!(s.execute("slice.set", json!({"slice": id, "rect": [500, 500, 10, 10]})).is_err());
    assert_eq!(doc(&s).slices.get(id).unwrap().rect, Rect::new(0, 0, 10, 10));
    assert_eq!(s.active().unwrap().history.past_len(), past);
}

#[test]
fn dividing_an_auto_slice_is_one_step() {
    // #911: one undo restores the original auto slice.
    let mut s = session(8);
    let past = s.active().unwrap().history.past_len();
    let r = s.execute("slice.divide", json!({"number": 1, "vertical": 2})).unwrap();
    assert_eq!(r["count"], 2);
    assert_eq!(doc(&s).slices.list.len(), 2);
    assert_eq!(s.active().unwrap().history.past_len(), past + 1);
    assert!(s.undo());
    assert!(doc(&s).slices.is_empty());
    // A rejected division doesn't promote either.
    assert!(s.execute("slice.divide", json!({"number": 1, "vertical": 1000})).is_err());
    assert!(doc(&s).slices.is_empty());
    assert_eq!(s.active().unwrap().history.past_len(), past);
}

#[test]
fn slice_options_after_an_explicit_promote_is_its_own_step() {
    // #912: Slice Options used to fold into an earlier, separate Promote.
    let mut s = session(8);
    s.execute("slice.promote", json!({"number": 1})).unwrap();
    s.execute("slice.set", json!({"number": 1, "name": "renamed"})).unwrap();
    assert!(s.undo());
    assert_eq!(doc(&s).slices.list.len(), 1, "the promotion survives undoing the rename");
    assert_eq!(doc(&s).slices.list[0].name, "");
    assert!(s.undo());
    assert!(doc(&s).slices.is_empty());
}

#[test]
fn out_of_range_slice_ids_are_rejected_not_wrapped() {
    // #914: 2^32 + 1 used to wrap to slice 1 and delete it.
    let mut s = session(8);
    let id = s.execute("slice.new", json!({"rect": [0, 0, 10, 10]})).unwrap()["slice"].as_u64().unwrap();
    assert_eq!(id, 1);
    let wrapped = (1u64 << 32) + 1;
    assert!(s.execute("slice.delete", json!({"slice": wrapped})).is_err());
    assert!(s.execute("slice.delete", json!({"slices": [wrapped]})).is_err());
    assert!(s.execute("slice.set", json!({"slice": wrapped, "name": "x"})).is_err());
    assert!(s.execute("slice.delete", json!({"slice": -1})).is_err());
    assert_eq!(doc(&s).slices.list.len(), 1);
    assert_eq!(doc(&s).slices.list[0].name, "");
}

#[test]
fn divide_refuses_more_parts_than_one_division_makes() {
    // #1020: 1000 x 1000 was accepted (a million slices), and resolving the list is quadratic in
    // the slice count.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 1000, "height": 1000})).unwrap();
    s.execute("slice.new", json!({"rect": [0, 0, 1000, 1000]})).unwrap();
    let id = doc(&s).slices.list[0].id;
    for (h, v) in [(101, 100), (100, 101), (1000, 1000)] {
        let (before, past) = (doc(&s).slices.clone(), s.active().unwrap().history.past_len());
        let err = s.execute("slice.divide", json!({"slice": id, "horizontal": h, "vertical": v})).unwrap_err();
        assert!(err.to_string().contains("10000"), "{h} x {v}: {err}");
        assert_eq!((&doc(&s).slices, s.active().unwrap().history.past_len()), (&before, past), "{h} x {v}: nothing changed");
    }
    // At the limit it divides, and the list resolves.
    let r = s.execute("slice.divide", json!({"slice": id, "horizontal": 100, "vertical": 100})).unwrap();
    assert_eq!(r["count"], 10_000);
    assert_eq!(doc(&s).slices.list.len(), 10_000);
    assert!(s.execute("slice.list", json!({})).is_ok());
}
