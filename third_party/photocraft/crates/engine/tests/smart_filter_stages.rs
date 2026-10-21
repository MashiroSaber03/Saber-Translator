//! One integration-test process keeps the global smart filter stage cache isolated from parallel
//! unit tests, so these tests can tell a cached stage from a re-render.
use std::sync::Arc;

use photocraft_doc::LayerContent;
use photocraft_engine::{Session, smart_cmds};
use photocraft_raster::Surface;
use serde_json::json;

fn rendered(s: &Session) -> Surface {
    let d = s.active().unwrap();
    match &d.doc.layer(d.active_layer.unwrap()).unwrap().content {
        LayerContent::Smart(sm) => sm.cache.clone().unwrap(),
        other => panic!("not a smart object: {}", other.kind_name()),
    }
}

/// Same tiles, not just the same pixels: the surface came from the cache rather than a re-render.
fn same_tiles(a: &Surface, b: &Surface) -> bool {
    a.tile_count() == b.tile_count() && a.tiles().zip(b.tiles()).all(|((ca, ta), (cb, tb))| ca == cb && Arc::ptr_eq(ta, tb))
}

#[test]
fn toggling_the_top_filter_reuses_the_stages_below_it_until_purged() {
    for depth in [8, 16, 32] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 64, "depth": depth})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("paint.gradient", json!({"from": [0, 0], "to": [96, 64], "colors": ["#ff0000", "#0000ff"]})).unwrap();
        s.execute("filter.convertForSmartFilters", json!({})).unwrap();
        s.execute("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
        let blurred = rendered(&s);
        s.execute("filter.noise.addNoise", json!({"amount": 12, "seed": 1})).unwrap();
        let full = rendered(&s);

        s.execute("layer.smartFilter.setVisible", json!({"index": 1, "visible": false})).unwrap();
        let hidden = rendered(&s);
        assert!(same_tiles(&hidden, &blurred), "hiding the top filter shows the cached stage below it at {depth}-bit");
        s.execute("layer.smartFilter.setVisible", json!({"index": 1, "visible": true})).unwrap();
        assert!(same_tiles(&rendered(&s), &full), "showing it again reuses the full stack at {depth}-bit");

        assert!(smart_cmds::stack_cache_bytes() > 0);
        assert!(s.is_enabled("edit.purge.all"));
        let purged = s.execute("edit.purge.all", json!({})).unwrap();
        assert!(purged["purged"].as_array().unwrap().contains(&json!("smart filter cache")), "{purged}");
        assert_eq!(smart_cmds::stack_cache_bytes(), 0);

        s.execute("layer.smartFilter.setVisible", json!({"index": 1, "visible": false})).unwrap();
        let rerendered = rendered(&s);
        assert_eq!(rerendered, hidden, "a cold render gives the same pixels at {depth}-bit");
        assert!(!same_tiles(&rerendered, &hidden), "after a purge the stage is rendered again");
    }
}
