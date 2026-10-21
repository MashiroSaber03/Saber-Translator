use super::*;

/// A smart object with three smart filters, the middle one blended at half opacity.
fn filtered(depth: u64) -> Session {
    let mut s = session(depth);
    paint(&mut s);
    s.execute("filter.convertForSmartFilters", json!({})).unwrap();
    s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
    s.execute("filter.noise.addNoise", json!({"amount": 20, "distribution": "gaussian", "seed": 5})).unwrap();
    s.execute("filter.blur.motionBlur", json!({"angle": 30, "distance": 6})).unwrap();
    s.execute("layer.smartFilter.blendingOptions", json!({"index": 1, "blend": "multiply", "opacity": 0.5})).unwrap();
    s
}

/// The active smart object rendered without the stage cache.
fn uncached(s: &Session) -> Surface {
    let doc = &s.active().unwrap().doc;
    let sm = active_smart(s);
    let (name, bytes) = source_bytes(&doc.metadata, &sm.source).unwrap();
    apply_smart_filters(&place(&name, &bytes, doc.pixel_format(), &sm).unwrap(), &sm, doc.bounds())
}

#[test]
fn renders_from_cached_stages_match_full_renders_at_every_depth() {
    for depth in DEPTHS {
        let mut s = filtered(depth);
        let edits = [
            ("layer.smartFilter.setVisible", json!({"index": 2, "visible": false})),
            ("layer.smartFilter.setVisible", json!({"index": 2, "visible": true})),
            ("layer.smartFilter.setVisible", json!({"index": 0, "visible": false})),
            ("layer.smartFilter.setVisible", json!({"index": 0, "visible": true})),
            ("layer.smartFilter.setParams", json!({"index": 2, "params": {"distance": 9}})),
            ("layer.smartFilter.blendingOptions", json!({"index": 1, "opacity": 0.8})),
            ("layer.smartFilter.move", json!({"index": 0, "to": 2})),
            ("layer.smartFilter.disableSmartFilters", json!({"enabled": false})),
            ("layer.smartFilter.disableSmartFilters", json!({"enabled": true})),
            ("layer.smartFilter.delete", json!({"index": 1})),
        ];
        for (id, params) in edits {
            s.execute(id, params.clone()).unwrap();
            assert_eq!(active_smart(&s).cache.unwrap(), uncached(&s), "{id} {params} at {depth}-bit");
        }
    }
}

#[test]
fn a_cancelled_render_stops_before_an_uncached_filter() {
    let s = filtered(8);
    let mut sm = active_smart(&s);
    // A radius no other render uses, so the stack below it is cached but this stage isn't.
    sm.smart_filters[0].params["radius"] = json!(2.375);
    let ctx = JobCtx::new();
    ctx.cancel();
    assert!(matches!(render_in(&s.active().unwrap().doc, &sm, &ctx), Err(EngineError::Cancelled)));
}

#[test]
fn bad_smart_filter_edits_fail_at_once_with_start() {
    let mut s = filtered(8);
    let before = active_smart(&s);
    let r = s.start("layer.smartFilter.setVisible", json!({"index": 9}));
    assert!(matches!(r, Err(EngineError::BadParams { .. })), "{r:?}");
    assert_eq!(active_smart(&s), before);
}
