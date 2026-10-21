//! One integration-test process keeps its global mask cache isolated from parallel unit tests.
use std::sync::Arc;

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_compose::masks;
use photocraft_doc::{Layer, LayerMask};
use photocraft_engine::Session;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::json;

#[test]
fn purge_all_is_enabled_by_cached_masks_and_releases_them_at_every_depth() {
    let mut session = Session::new();
    assert!(!session.is_enabled("edit.purge.all"));
    let canvas = Rect::new(0, 0, 32, 32);
    for sample in SampleType::ALL {
        let mut input = Surface::with_default(PixelFormat { mode: ColorMode::Grayscale, sample, alpha: false }, &[1.0]);
        input.fill_rect(Rect::new(8, 8, 24, 24), &[0.0]);
        let mut layer = Layer::raster("mask", PixelFormat::RGBA8);
        layer.mask = Some(LayerMask { surface: input, feather: 2.0, ..LayerMask::reveal_all() });
        let before = masks::combined_mask(&layer, canvas).unwrap();
        let accounted = masks::cache_bytes();
        assert!(accounted > 0, "{sample:?}");
        assert!(session.is_enabled("edit.purge.all"), "only the mask cache enables purge");

        let result = session.execute("edit.purge.all", json!({})).unwrap();
        assert_eq!(result["purged"], json!(["mask cache"]));
        assert_eq!(result["bytes"].as_u64(), Some(accounted as u64));
        assert_eq!(masks::cache_bytes(), 0);
        assert!(!session.is_enabled("edit.purge.all"));

        let after = masks::combined_mask(&layer, canvas).unwrap();
        assert_eq!(before, after, "purge preserves pixels at {sample:?}");
        let old_tile = before.tiles().next().unwrap().1;
        let new_tile = after.tiles().next().unwrap().1;
        assert!(!Arc::ptr_eq(old_tile, new_tile), "the result was rebuilt after purge");
        session.execute("edit.purge.all", json!({})).unwrap();
        assert_eq!(masks::cache_bytes(), 0);
    }
}
