//! A native document and its PSD must quantize the same later mask edit alike.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, LayerMask};
use photocraft_engine::Session;
use photocraft_raster::Surface;
use serde_json::json;

fn session(depth: u64, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48, "depth": depth, "mode": mode})).unwrap();
    s
}

fn mask(doc: &Document) -> &LayerMask {
    doc.walk().into_iter().find_map(|(_, _, l)| l.mask.as_ref()).unwrap()
}

#[test]
fn newly_created_layer_masks_follow_document_depth() {
    for (depth, sample) in [(8, SampleType::U8), (16, SampleType::U16), (32, SampleType::F32)] {
        for mode in ["rgb", "gray", "cmyk", "lab"] {
            for (command, selected) in [
                ("layer.newAdjustmentLayer.invert", false),
                ("layer.newAdjustmentLayer.invert", true),
                ("layer.newFillLayer.solidColor", false),
                ("layer.newFillLayer.solidColor", true),
                ("layer.layerMask.revealAll", false),
                ("layer.layerMask.hideAll", false),
                ("layer.layerMask.revealSelection", true),
            ] {
                let mut s = session(depth, mode);
                if selected {
                    s.execute("select.rect", json!({"x": 8, "y": 6, "width": 40, "height": 32, "antiAlias": false})).unwrap();
                }
                let before = s.active().unwrap().doc.clone();
                let history = s.active().unwrap().history.past_len();
                s.execute(command, json!({"color": "#000000"})).unwrap();
                let d = s.active().unwrap();
                let m = mask(&d.doc);
                assert_eq!(m.surface.format(), PixelFormat::new(ColorMode::Grayscale, sample, false), "{command}, {depth}-bit {mode}");
                assert!(m.enabled && m.linked);
                assert_eq!((m.density, m.feather), (1.0, 0.0));
                let white = if command.ends_with("hideAll") { 0.0 } else { 1.0 };
                assert_eq!(m.value(20, 20), white);
                if selected {
                    assert_eq!(m.value(0, 0), 0.0);
                    if command == "layer.layerMask.revealSelection" {
                        // The selection became the mask: Photoshop drops it, as Hide Selection does.
                        assert!(d.doc.selection.is_none(), "{command}: the selection is dropped");
                    } else {
                        assert_eq!(d.doc.selection, before.selection, "the selection is unchanged");
                    }
                } else {
                    assert_eq!(m.surface.tile_count(), 0, "uniform masks stay sparse");
                    assert_eq!(m.value(-300_000, 300_000), white);
                }
                assert_eq!(d.history.past_len(), history + 1);
                let after = d.doc.clone();
                s.undo();
                assert_eq!(*s.active().unwrap().doc, *before);
                s.redo();
                assert_eq!(*s.active().unwrap().doc, *after);
            }
        }
    }
}

#[test]
fn soft_mask_edits_match_after_native_and_psd_reopen() {
    for depth in [8, 16, 32] {
        let mut s = session(depth, "rgb");
        s.execute("select.rect", json!({"x": 8, "y": 6, "width": 40, "height": 32, "antiAlias": false})).unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({"color": "#000000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let native = photocraft_format::save_to_bytes(&before, &Default::default()).unwrap();
        let psd = photocraft_io::export(&before, "psd", &Default::default()).unwrap();
        assert!(psd.warnings.is_empty(), "{:?}", psd.warnings);
        let mut branches = Vec::new();
        for reopened in [photocraft_format::load_from_bytes(&native).unwrap(), photocraft_io::import("mask.psd", &psd.bytes).unwrap().document] {
            assert_eq!(mask(&reopened).surface.format(), mask(&before).surface.format(), "{depth}-bit mask depth survives reopening");
            assert_eq!(mask(&reopened).surface.read_region(before.bounds()), mask(&before).surface.read_region(before.bounds()));
            assert_eq!(photocraft_compose::flatten(&reopened), photocraft_compose::flatten(&before));
            let layer = reopened.walk().into_iter().find_map(|(_, _, l)| l.mask.as_ref().map(|_| l.id.0)).unwrap();
            let mut edited = Session::new();
            edited.add_document(reopened, None);
            let undo_doc = edited.active().unwrap().doc.clone();
            edited
                .execute(
                    "paint.stroke",
                    json!({"layer": layer, "target": "mask", "color": "#000000", "points": [[20,18,1], [36,30,1]], "size": 18, "hardness": 0.4, "seed": 17}),
                )
                .unwrap();
            let changed = edited.active().unwrap().doc.clone();
            let values = mask(&changed).surface.read_region(changed.bounds());
            assert!(values.iter().any(|v| *v > 0.0 && *v < 1.0), "a soft mask was painted");
            edited.undo();
            assert_eq!(*edited.active().unwrap().doc, *undo_doc);
            edited.redo();
            assert_eq!(*edited.active().unwrap().doc, *changed);
            branches.push(changed);
        }
        let a = mask(&branches[0]).surface.read_region(before.bounds());
        let b = mask(&branches[1]).surface.read_region(before.bounds());
        let changed = a.iter().zip(&b).filter(|(a, b)| a != b).count();
        let worst = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert_eq!(changed, 0, "{depth}-bit mask edit depends on the save format (worst coverage difference {worst})");
        assert_eq!(photocraft_compose::flatten(&branches[0]), photocraft_compose::flatten(&branches[1]), "{depth}-bit composite depends on the save format");
        if depth > 8 {
            assert!(a.iter().any(|v| (v * 255.0 - (v * 255.0).round()).abs() > 0.01), "{depth}-bit mask edits retain sub-8-bit coverage");
        }
    }
}

#[test]
fn mask_creation_preserves_selection_coverage_and_existing_masks() {
    for depth in [8, 16, 32] {
        for command in ["layer.newAdjustmentLayer.invert", "layer.layerMask.revealSelection", "gradient.fill.create"] {
            let mut s = session(depth, "rgb");
            s.execute("layer.new.layer", json!({})).unwrap();
            s.edit("synthetic coverage", |doc, _| {
                let mut selection = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
                selection.write_region(photocraft_geom::Rect::new(0, 0, 16, 16), &(0..256).map(|i| i as f32 / 255.0).collect::<Vec<_>>());
                doc.selection = Some(selection);
                let mut existing = LayerMask::reveal_all();
                existing.surface = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false), &[0.123456]);
                doc.layers[0].mask = Some(existing);
                Ok(())
            })
            .unwrap();
            let before = s.active().unwrap().doc.clone();
            s.execute(command, json!({"from": [0, 0], "to": [64, 48]})).unwrap();
            let after = &s.active().unwrap().doc;
            let added = after.layer(s.active().unwrap().active_layer.unwrap()).unwrap().mask.as_ref().unwrap();
            assert_eq!(
                added.surface.read_region(after.bounds()),
                before.selection.as_ref().unwrap().read_region(before.bounds()),
                "{depth}-bit coverage is unchanged"
            );
            let kept = if command == "layer.layerMask.revealSelection" { None } else { before.selection.clone() };
            assert_eq!(after.selection, kept, "{command}");
            assert_eq!(after.layers[0].mask, before.layers[0].mask, "creating a mask does not requantize an existing mask");
        }
    }
}
