use std::sync::Arc;

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Adjustment, Document, Layer, LayerContent, LayerMask, Path, VectorMask};
use photocraft_geom::{Rect, Size};
use photocraft_raster::Surface;

use super::*;

fn session(depth: SampleType, adjustment: bool) -> (Session, LayerId, LayerId) {
    let mut doc = Document::new("Masks", Size::new(8, 8), ColorMode::Rgb, depth);
    let background = Layer::raster("Unmasked", doc.pixel_format());
    let background_id = background.id;
    doc.layers.push(background);
    let mut layer =
        if adjustment { Layer::new("Adjustment", LayerContent::Adjustment(Adjustment::Invert)) } else { Layer::raster("Pixels", doc.pixel_format()) };
    let mut surface = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, depth, false), &[1.0]);
    surface.fill_rect(Rect::new(1, 2, 5, 6), &[0.25]);
    layer.mask = Some(LayerMask { surface, enabled: !adjustment, linked: false, density: 1.0, feather: 0.0 });
    let path = Path { inverted: true, ..Default::default() };
    layer.vector_mask = Some(VectorMask::new(path));
    let id = layer.id;
    doc.layers.push(layer);
    doc.selection = Some(Surface::with_default(PixelFormat::GRAY8, &[0.5]));
    let mut s = Session::new();
    s.add_document(doc, None);
    (s, id, background_id)
}

#[test]
fn edits_preserve_mask_pixels_and_layer_at_all_depths_with_undo_redo() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for adjustment in [false, true] {
            let (mut s, id, background) = session(depth, adjustment);
            // Explicit targets work even when another layer is active.
            s.select_layer(background).unwrap();
            let before = s.active().unwrap().doc.clone();
            let count = s.active().unwrap().history.entries().len();
            let result = s.execute(EDIT, json!({"layer": id.0, "document": before.id.0, "density": 25, "feather": 2.5})).unwrap();
            assert_eq!(result, json!({"layer": id.0, "density": 25.0, "feather": 2.5}));
            let mut expected = (*before).clone();
            let mask = expected.layer_mut(id).unwrap().mask.as_mut().unwrap();
            (mask.density, mask.feather) = (0.25, 2.5);
            assert_eq!(*s.active().unwrap().doc, expected, "only density and feather change");
            assert_eq!(s.active().unwrap().active_layer, Some(background));
            assert_eq!(s.active().unwrap().history.entries().len(), count + 1);
            // Properties invalidate the composite as far as their undo would (the masked layer is
            // empty here; the adjustment reaches everything).
            let damage = s.active().unwrap().last_damage;
            assert_eq!(damage, crate::layer_multi_cmds::step_damage(&before, &s.active().unwrap().doc));
            assert_eq!(damage.is_none(), adjustment);
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(*s.active().unwrap().doc, *before);
            s.execute("edit.redo", json!({})).unwrap();
            assert_eq!(*s.active().unwrap().doc, expected);
            let unchanged = s.active().unwrap().doc.clone();
            s.execute(EDIT, json!({"layer": id.0, "density": 25})).unwrap();
            assert!(Arc::ptr_eq(&unchanged, &s.active().unwrap().doc), "same value is not an edit");
            assert_eq!(s.active().unwrap().history.entries().len(), count + 1);
        }
    }
}

#[test]
fn property_commands_update_rendered_edges_without_rewriting_the_mask() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for adjustment in [false, true] {
            let mut doc = Document::new("Mask edge", Size::new(64, 8), ColorMode::Rgb, depth);
            let mut background = Layer::raster("White", doc.pixel_format());
            background.surface_mut().unwrap().fill_rect(doc.bounds(), &[1.0; 4]);
            doc.layers.push(background);
            let mut layer = if adjustment {
                Layer::new("Invert", LayerContent::Adjustment(Adjustment::Invert))
            } else {
                let mut layer = Layer::raster("Black", doc.pixel_format());
                layer.surface_mut().unwrap().fill_rect(doc.bounds(), &[0.0, 0.0, 0.0, 1.0]);
                layer
            };
            let mut surface = Surface::new(PixelFormat::new(ColorMode::Grayscale, depth, false));
            surface.fill_rect(Rect::new(0, 0, 32, 8), &[1.0]);
            let raw = surface.clone();
            layer.mask = Some(LayerMask { surface, enabled: true, linked: true, density: 1.0, feather: 0.0 });
            let id = layer.id;
            doc.layers.push(layer);
            let mut s = Session::new();
            s.add_document(doc, None);
            let render = |s: &Session| photocraft_compose::flatten(&s.active().unwrap().doc);
            let original = render(&s);
            assert!(original.get(31, 4)[0] < 0.01 && original.get(32, 4)[0] > 0.99);

            s.execute(EDIT, json!({"density": 50})).unwrap();
            assert!((render(&s).get(60, 4)[0] - 0.5).abs() < 0.01, "density reveals the hidden side");
            s.execute(EDIT, json!({"feather": 8})).unwrap();
            let soft_half = render(&s); // Warms the combined-mask cache before the next edit.
            assert!(soft_half.get(31, 4)[0] > 0.15 && soft_half.get(32, 4)[0] < 0.35, "feather softens both sides of the edge");
            s.execute(EDIT, json!({"density": 100})).unwrap();
            let soft_full = render(&s);
            for x in [31, 32, 60] {
                assert!((soft_full.get(x, 4)[0] - 2.0 * soft_half.get(x, 4)[0]).abs() < 0.01, "density refreshes the cached feathered mask");
            }
            s.execute(EDIT, json!({"feather": 0})).unwrap();
            assert_eq!(render(&s), original, "clearing feather restores the sharp edge");
            assert_eq!(s.active().unwrap().doc.layer(id).unwrap().mask.as_ref().unwrap().surface, raw);
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(render(&s), soft_full);
            s.execute("edit.redo", json!({})).unwrap();
            assert_eq!(render(&s), original);
        }
    }
}

#[test]
fn slider_updates_coalesce_without_merging_separate_gestures() {
    let (mut s, id, _) = session(SampleType::U16, false);
    let before = s.active().unwrap().doc.clone();
    let count = s.active().unwrap().history.entries().len();
    for density in [80, 50, 0] {
        s.execute(EDIT, json!({"density": density, "coalesce": "mask-density-drag"})).unwrap();
    }
    let density_only = s.active().unwrap().doc.clone();
    assert_eq!(s.active().unwrap().history.entries().len(), count + 1);
    for feather in [2.5, 1000.0] {
        s.execute(EDIT, json!({"feather": feather, "coalesce": "mask-feather-drag"})).unwrap();
    }
    let done = s.active().unwrap().doc.clone();
    assert_eq!(s.active().unwrap().history.entries().len(), count + 2);
    assert_eq!(done.layer(id).unwrap().mask.as_ref().map(|m| (m.density, m.feather)), Some((0.0, 1000.0)));
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(*s.active().unwrap().doc, *density_only);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(*s.active().unwrap().doc, *before);
    s.execute("edit.redo", json!({})).unwrap();
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(*s.active().unwrap().doc, *done);
}

#[test]
fn malformed_params_missing_masks_and_stale_documents_do_not_edit() {
    let (mut s, _, background) = session(SampleType::U8, false);
    let before = s.active().unwrap().doc.clone();
    let count = s.active().unwrap().history.entries().len();
    let revision = s.active().unwrap().revision;
    for params in [
        Value::Null,
        json!([]),
        json!({}),
        json!({"density": "50"}),
        json!({"density": null}),
        json!({"density": -0.01}),
        json!({"density": 100.01}),
        json!({"feather": false}),
        json!({"feather": -0.01}),
        json!({"feather": 1000.01}),
        json!({"feather": 1e100}),
        json!({"density": 25, "feather": []}),
        json!({"layer": "bad", "density": 25}),
        json!({"layer": null, "density": 25}),
        json!({"layer": -1, "density": 25}),
        json!({"layer": u64::MAX, "density": 25}),
        json!({"layer": background.0, "density": 25}),
        json!({"document": "bad", "density": 25}),
        json!({"document": u64::MAX, "density": 25}),
    ] {
        assert!(s.execute(EDIT, params.clone()).is_err(), "{params}");
        let st = s.active().unwrap();
        assert!(Arc::ptr_eq(&before, &st.doc), "{params}");
        assert_eq!((st.history.entries().len(), st.revision), (count, revision), "{params}");
    }
    let mut empty = Session::new();
    assert!(!empty.is_enabled(EDIT));
    assert!(empty.execute(EDIT, json!({"density": 50})).is_err());
}
