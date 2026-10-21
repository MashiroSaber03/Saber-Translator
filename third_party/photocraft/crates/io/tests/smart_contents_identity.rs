//! Shared contents and independent placed layers through PSD/PSB persistence.

use std::sync::Arc;

use photocraft_color::{BlendMode, ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent, SmartContentsId, SmartFilter, SmartObject, SmartSource};
use photocraft_geom::{Affine, Size};
use photocraft_io::{ExportOptions, export, import, linked, smart_map};
use photocraft_psd::{PsdFile, TaggedBlock};
use photocraft_raster::Surface;

fn fixture() -> Document {
    let mut source = Document::new("contents", Size::new(2, 2), ColorMode::Rgb, SampleType::U8);
    let mut pixels = Surface::new(source.pixel_format());
    pixels.fill_rect(source.bounds(), &[1.0, 0.0, 0.0, 1.0]);
    source.layers.push(Layer::new("red", LayerContent::Raster(pixels.clone())));
    let bytes = photocraft_format::save_to_bytes(&source, &Default::default()).unwrap();
    let mut doc = Document::new("instances", Size::new(12, 4), ColorMode::Rgb, SampleType::U8);
    let smart = SmartObject::new(SmartSource::Embedded { file_name: "contents.pcraft".into(), bytes: Arc::new(bytes) }, Affine::IDENTITY, Some(pixels));
    doc.layers.push(Layer::new("original", LayerContent::Smart(smart)));
    doc
}

fn smart<'a>(doc: &'a Document, name: &str) -> &'a SmartObject {
    let layer = doc.layers.iter().find(|l| l.name == name).unwrap();
    let LayerContent::Smart(sm) = &layer.content else { panic!("{name} is no longer a smart object") };
    sm
}

fn smart_mut(layer: &mut Layer) -> &mut SmartObject {
    let LayerContent::Smart(sm) = &mut layer.content else { panic!("expected smart object") };
    sm
}

fn round_trip(doc: &Document, extension: &str) -> (PsdFile, Document) {
    let out = export(doc, extension, &ExportOptions::default()).unwrap();
    assert!(!out.warnings.iter().any(|w| w.contains("smart object written as pixels")), "{:?}", out.warnings);
    let file = PsdFile::from_bytes(&out.bytes).unwrap();
    let back = import(extension, &out.bytes).unwrap().document;
    (file, back)
}

fn sold(file: &PsdFile, name: &str) -> smart_map::Placed {
    let layer = file.layers().iter().find(|l| l.name() == name).unwrap();
    smart_map::parse_sold(&layer.block(b"SoLd").unwrap().data).unwrap()
}

fn gaussian(radius: f64) -> SmartFilter {
    SmartFilter {
        command: "filter.blur.gaussianBlur".into(),
        params: serde_json::json!({ "radius": radius }),
        blend: BlendMode::Normal,
        opacity: 1.0,
        visible: true,
    }
}

#[test]
fn psd_and_psb_keep_shared_contents_and_identical_independent_copies_distinct() {
    let mut doc = fixture();
    let mut shared = doc.layers[0].duplicate();
    shared.name = "shared".into();
    let sm = smart_mut(&mut shared);
    sm.transform = Affine::translate(4.0, 0.0);
    sm.cache = sm.cache.as_ref().map(|c| c.translated(4, 0, c.content_bounds()));
    sm.smart_filters.push(gaussian(2.0));
    let mut independent = doc.layers[0].duplicate();
    independent.name = "independent".into();
    smart_mut(&mut independent).contents_id = SmartContentsId::fresh();
    doc.layers.extend([shared, independent]);

    for extension in ["psd", "psb"] {
        let (file, back) = round_trip(&doc, extension);
        let a = sold(&file, "original");
        let b = sold(&file, "shared");
        let c = sold(&file, "independent");
        assert_eq!(a.idnt, b.idnt);
        assert_ne!(a.idnt, c.idnt);
        assert_eq!([a.placed, b.placed, c.placed].into_iter().collect::<std::collections::HashSet<_>>().len(), 3);
        let payloads: Vec<_> = file.global_blocks.iter().filter(|b| &b.key == b"lnk2").flat_map(|b| linked::parse_linked_files(&b.data)).collect();
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].bytes, payloads[1].bytes, "independent identical contents still have separate payload identities");
        assert_eq!(smart(&back, "original").contents_id, smart(&back, "shared").contents_id);
        assert_ne!(smart(&back, "original").contents_id, smart(&back, "independent").contents_id);
        assert_eq!(smart(&back, "shared").transform, Affine::translate(4.0, 0.0));
        assert_eq!(smart(&back, "shared").smart_filters[0].params["radius"].as_f64(), Some(2.0));
        assert!(smart(&back, "original").smart_filters.is_empty());
        assert!(smart(&back, "independent").smart_filters.is_empty());
    }
}

#[test]
fn duplicated_imported_layer_keeps_source_but_gets_its_own_placement() {
    let (before, mut doc) = round_trip(&fixture(), "psd");
    let original_files = before.global_blocks.iter().find(|b| &b.key == b"lnk2").unwrap().data.clone();
    let mut duplicate = doc.layers[0].duplicate();
    duplicate.name = "duplicate".into();
    smart_mut(&mut duplicate).smart_filters.push(gaussian(3.0));
    // An imported UUID may have been generated under a different allocator lifetime and equal
    // the duplicate's first candidate. Preserve it for the original even when emitted second.
    let old = sold(&before, "original");
    let colliding = smart_map::uuid_from(format!("{}:{}", old.idnt, duplicate.id.0).as_bytes());
    let spec = smart_map::PlacedSpec {
        idnt: &old.idnt,
        placed: &colliding,
        transform: Affine::IDENTITY,
        perspective: None,
        size: (2.0, 2.0),
        dpi: 72.0,
        warp: None,
        filter_fx: None,
    };
    let original_raw = smart_map::sold_bytes(Some(&old.descriptor), &spec, &mut Vec::new());
    for layer in [&mut doc.layers[0], &mut duplicate] {
        smart_mut(layer).psd_raw = Some(Arc::new(original_raw.clone()));
        for (key, data) in &mut layer.psd_blocks {
            if key == b"SoLd" {
                *data = Arc::new(original_raw.clone());
            }
        }
    }
    doc.layers.push(duplicate);

    for reversed in [false, true] {
        let mut ordered = doc.clone();
        if reversed {
            ordered.layers.reverse();
        }
        let (after, back) = round_trip(&ordered, "psd");
        let original = sold(&after, "original");
        let duplicate = sold(&after, "duplicate");
        assert_eq!(original.idnt, duplicate.idnt);
        assert_eq!(original.placed, colliding);
        assert_ne!(original.placed, duplicate.placed);
        let filter_cache = after.global_blocks.iter().find(|b| &b.key == b"FEid").unwrap();
        let filter_cache = photocraft_psd::filter_effects::FilterEffects::parse(&filter_cache.data).unwrap();
        assert_eq!(filter_cache.items.len(), 1);
        assert_eq!(filter_cache.items[0].id, duplicate.placed);
        assert_eq!(after.layers().iter().find(|l| l.name() == "original").unwrap().block(b"SoLd").unwrap().data, original_raw);
        assert_eq!(after.global_blocks.iter().find(|b| &b.key == b"lnk2").unwrap().data, original_files);
        assert_eq!(smart(&back, "original").contents_id, smart(&back, "duplicate").contents_id);
        assert_eq!(smart(&back, "duplicate").smart_filters[0].params["radius"].as_f64(), Some(3.0));
        assert!(smart(&back, "original").smart_filters.is_empty());
    }
}

#[test]
fn missing_source_identifiers_do_not_join_imported_contents() {
    let (mut file, _) = round_trip(&fixture(), "psd");
    let spec = smart_map::PlacedSpec {
        idnt: "",
        placed: "placement",
        transform: Affine::IDENTITY,
        perspective: None,
        size: (2.0, 2.0),
        dpi: 72.0,
        warp: None,
        filter_fx: None,
    };
    let data = smart_map::sold_bytes(None, &spec, &mut Vec::new());
    let mut record = file.layers()[0].clone();
    record.blocks.retain(|b| !matches!(&b.key, b"SoLd" | b"SoLE" | b"PlLd"));
    record.blocks.push(TaggedBlock::new(*b"SoLd", data));
    *file.layers_mut() = vec![record.clone(), record];
    let (doc, _) = photocraft_io::psd_to_document(&file);
    let LayerContent::Smart(a) = &doc.layers[0].content else { panic!("first smart object missing") };
    let LayerContent::Smart(b) = &doc.layers[1].content else { panic!("second smart object missing") };
    assert_ne!(a.contents_id, b.contents_id);
    let (_, back) = round_trip(&doc, "psd");
    let LayerContent::Smart(a) = &back.layers[0].content else { panic!("first smart object lost on save") };
    let LayerContent::Smart(b) = &back.layers[1].content else { panic!("second smart object lost on save") };
    assert_ne!(a.contents_id, b.contents_id);
}

#[test]
fn independent_unavailable_link_retains_separate_preserved_record() {
    let (mut file, _) = round_trip(&fixture(), "psd");
    let block = file.global_blocks.iter_mut().find(|b| &b.key == b"lnk2").unwrap();
    // An opaque external item: the source UUID is known, but no embedded bytes can be loaded.
    block.data[8..12].copy_from_slice(b"liFE");
    let original_record = block.data.clone();
    let (mut doc, _) = photocraft_io::psd_to_document(&file);
    let mut independent = doc.layers[0].duplicate();
    independent.name = "independent".into();
    let sm = smart_mut(&mut independent);
    sm.contents_id = SmartContentsId::fresh();
    sm.psd_raw = None;
    independent.psd_blocks.retain(|(key, _)| !matches!(key, b"SoLd" | b"SoLE" | b"PlLd"));
    // Encounter the independent copy first: the original imported identity must keep its UUID.
    doc.layers.insert(0, independent);

    let (after, back) = round_trip(&doc, "psd");
    assert_eq!(sold(&file, "original").idnt, sold(&after, "original").idnt);
    assert_ne!(sold(&after, "original").idnt, sold(&after, "independent").idnt);
    let links = &after.global_blocks.iter().find(|b| &b.key == b"lnk2").unwrap().data;
    assert!(links.starts_with(&original_record));
    assert_eq!(linked::block_uuids(links).len(), 2);
    assert_ne!(smart(&back, "original").contents_id, smart(&back, "independent").contents_id);
    let (again, _) = round_trip(&doc, "psd");
    assert_eq!(after.to_bytes().unwrap(), again.to_bytes().unwrap());
}

/// A `PlLd`-only placed layer (Photoshop CS2, #2465) whose embedded file is missing still gets its
/// source id and placement (at an assumed 100 % scale); an unreadable `PlLd` is reported.
#[test]
fn plld_only_layers_without_their_file_or_with_a_broken_record() {
    let (mut file, _) = round_trip(&fixture(), "psd");
    file.global_blocks.retain(|b| !matches!(&b.key, b"lnk2" | b"lnk3" | b"lnkD"));
    let rotated = Affine { m: [0.0, 1.0, -1.0, 0.0, 6.0, 1.0] };
    let spec =
        smart_map::PlacedSpec { idnt: "missing", placed: "", transform: rotated, perspective: None, size: (3.0, 2.0), dpi: 72.0, warp: None, filter_fx: None };
    let mut record = file.layers()[0].clone();
    record.blocks.retain(|b| !matches!(&b.key, b"SoLd" | b"SoLE" | b"PlLd"));
    let mut broken = record.clone();
    record.blocks.push(TaggedBlock::new(*b"PlLd", smart_map::plld_bytes(&spec, &mut Vec::new())));
    broken.blocks.push(TaggedBlock::new(*b"PlLd", b"plcL\0\0\0\x03\x05ab\0\0".to_vec()));
    *file.layers_mut() = vec![record, broken];
    let (doc, warnings) = photocraft_io::psd_to_document(&file);
    let LayerContent::Smart(a) = &doc.layers[0].content else { panic!("PlLd layer is not a smart object") };
    assert_eq!(a.source, SmartSource::Linked { path: "missing".into() });
    for (x, y) in a.transform.m.iter().zip(rotated.m) {
        assert!((x - y).abs() < 1e-12, "{:?} vs {rotated:?}", a.transform);
    }
    let LayerContent::Smart(b) = &doc.layers[1].content else { panic!("broken PlLd layer is not a smart object") };
    assert_eq!(b.source, SmartSource::Linked { path: String::new() });
    assert_eq!(warnings.iter().filter(|w| w.contains("(PlLd) could not be read")).count(), 1, "{warnings:?}");
}

/// Photoshop CS2 can warp a smart object and keeps the warp in `PlLd`, after the corners. A
/// `PlLd`-only layer imports with that warp, so re-rendering it from its now-available source
/// (Free Transform, Image Size, mode changes) keeps it warped. A warp that can't be read leaves
/// the contents unavailable, so the warped pixels stay the rendering (#2465).
#[test]
fn plld_only_layers_keep_their_warp() {
    use photocraft_geom::warp::{Warp, WarpStyle};
    let (mut file, _) = round_trip(&fixture(), "psd");
    let bulge = Warp::preset(WarpStyle::Bulge, 35.0, [0.0, 0.0, 2.0, 2.0]);
    let spec = |warp| smart_map::PlacedSpec {
        idnt: "missing",
        placed: "",
        transform: Affine { m: [2.0, 0.0, 0.0, 2.0, 1.0, 0.0] },
        perspective: None,
        size: (2.0, 2.0),
        dpi: 72.0,
        warp,
        filter_fx: None,
    };
    let mut record = file.layers()[0].clone();
    record.blocks.retain(|b| !matches!(&b.key, b"SoLd" | b"SoLE" | b"PlLd"));
    let (mut warped, mut flat, mut broken) = (record.clone(), record.clone(), record);
    warped.blocks.push(TaggedBlock::new(*b"PlLd", smart_map::plld_bytes(&spec(Some(&bulge)), &mut Vec::new())));
    flat.blocks.push(TaggedBlock::new(*b"PlLd", smart_map::plld_bytes(&spec(None), &mut Vec::new())));
    // The warp descriptor cut short: a warp is stored but can't be read.
    let mut cut = smart_map::plld_bytes(&spec(Some(&bulge)), &mut Vec::new());
    cut.truncate(cut.len() - 40);
    broken.blocks.push(TaggedBlock::new(*b"PlLd", cut));
    *file.layers_mut() = vec![warped, flat, broken];
    let (doc, warnings) = photocraft_io::psd_to_document(&file);
    let sm = |i: usize| match &doc.layers[i].content {
        LayerContent::Smart(s) => s,
        _ => panic!("layer {i} is not a smart object"),
    };
    assert_eq!(sm(0).warp, Some(bulge));
    assert_eq!(sm(0).source, SmartSource::Linked { path: "missing".into() });
    assert_eq!((sm(1).warp.as_ref(), &sm(1).source), (None, &SmartSource::Linked { path: "missing".into() }));
    assert_eq!(sm(2).source, SmartSource::Linked { path: String::new() }, "unreadable warp: contents unavailable");
    assert!(sm(2).cache.is_some());
    assert_eq!(warnings.iter().filter(|w| w.contains("warp (PlLd) could not be read")).count(), 1, "{warnings:?}");
}

/// A linked smart object with both `SoLE` and `PlLd` is read from `SoLE`: `PlLd` alone is
/// Photoshop CS2's record (#2465).
#[test]
fn sole_wins_over_plld() {
    let (mut file, _) = round_trip(&fixture(), "psd");
    let spec = smart_map::PlacedSpec {
        idnt: "plld-only",
        placed: "",
        transform: Affine::IDENTITY,
        perspective: None,
        size: (2.0, 2.0),
        dpi: 72.0,
        warp: None,
        filter_fx: None,
    };
    let mut record = file.layers()[0].clone();
    let sold = record.block(b"SoLd").expect("the fixture has SoLd").data.clone();
    let (id, _) = photocraft_io::blocks::parse_smart(b"SoLd", &sold);
    assert!(!id.is_empty() && id != "plld-only");
    record.blocks.retain(|b| !matches!(&b.key, b"SoLd" | b"SoLE" | b"PlLd"));
    record.blocks.push(TaggedBlock::new(*b"PlLd", smart_map::plld_bytes(&spec, &mut Vec::new())));
    record.blocks.push(TaggedBlock::new(*b"SoLE", sold.clone()));
    *file.layers_mut() = vec![record];
    let (doc, _) = photocraft_io::psd_to_document(&file);
    let LayerContent::Smart(sm) = &doc.layers[0].content else { panic!("not a smart object") };
    assert_eq!(sm.source, SmartSource::Linked { path: id });
    assert_eq!(sm.psd_raw.as_deref(), Some(&sold));
}

/// A CS2 `PlLd`-only layer with a custom warp whose control points spill past its source frames
/// that control-point box with its corners, as `SoLd`'s `Trnf` does (#3677): it imports with the
/// same placement as the equivalent `SoLd` layer. Unedited, it saves its `PlLd` record verbatim
/// (no `SoLd` added) and reopens in the same place (#2465).
#[test]
fn plld_only_spilled_custom_warp_places_like_sold_and_round_trips_verbatim() {
    use photocraft_geom::warp::{BezierMesh, Warp};
    let (mut file, _) = round_trip(&fixture(), "psd");
    let mut record = file.layers()[0].clone();
    let (id, _) = photocraft_io::blocks::parse_smart(b"SoLd", &record.block(b"SoLd").expect("the fixture has SoLd").data);
    assert!(!id.is_empty());
    // The embedded source is 2 × 2; the mesh reaches from (-1, -0.5) to (3, 2.5).
    let bounds = [0.0, 0.0, 2.0, 2.0];
    let mut mesh = BezierMesh::identity(bounds, 1, 1);
    mesh.points[0] = [-1.0, -0.5];
    mesh.points[15] = [3.0, 2.5];
    let warp = Warp::custom(mesh, bounds);
    let t = Affine { m: [1.5, 0.0, 0.0, 1.5, 4.0, 1.0] };
    let spec =
        smart_map::PlacedSpec { idnt: &id, placed: "p", transform: t, perspective: None, size: (2.0, 2.0), dpi: 72.0, warp: Some(&warp), filter_fx: None };
    record.blocks.retain(|b| !matches!(&b.key, b"SoLd" | b"SoLE" | b"PlLd"));
    let (mut with_sold, mut cs2) = (record.clone(), record);
    with_sold.blocks.push(TaggedBlock::new(*b"SoLd", smart_map::sold_bytes(None, &spec, &mut Vec::new())));
    let plld = smart_map::plld_bytes(&spec, &mut Vec::new());
    cs2.blocks.push(TaggedBlock::new(*b"PlLd", plld.clone()));
    *file.layers_mut() = vec![with_sold, cs2];

    let (doc, warnings) = photocraft_io::psd_to_document(&file);
    assert!(!warnings.iter().any(|w| w.contains("PlLd")), "{warnings:?}");
    let sm = |d: &Document, i: usize| match &d.layers[i].content {
        LayerContent::Smart(s) => s.clone(),
        _ => panic!("layer {i} is not a smart object"),
    };
    let close = |a: Affine, b: Affine| a.m.iter().zip(b.m).all(|(x, y)| (x - y).abs() < 1e-9);
    let (a, b) = (sm(&doc, 0), sm(&doc, 1));
    assert!(close(a.transform, t), "SoLd: {:?} vs {t:?}", a.transform);
    assert!(close(b.transform, a.transform), "PlLd {:?} vs SoLd {:?}", b.transform, a.transform);
    assert_eq!((&b.source, &b.warp, b.perspective), (&a.source, &a.warp, a.perspective));
    assert_eq!(b.warp.as_ref(), Some(&warp));
    assert_eq!(b.source, SmartSource::Linked { path: id.clone() });

    // Unedited: the CS2 record is written back as it was, with no SoLd, and reopens in place.
    let out = export(&doc, "psd", &ExportOptions::default()).unwrap();
    let saved = PsdFile::from_bytes(&out.bytes).unwrap();
    let rec = &saved.layers()[1];
    assert_eq!(rec.block(b"PlLd").map(|b| &b.data), Some(&plld));
    assert!(rec.block(b"SoLd").is_none(), "an unedited PlLd-only layer gained a SoLd");
    let again = import("psd", &out.bytes).unwrap().document;
    let c = sm(&again, 1);
    assert!(close(c.transform, t), "{:?} vs {t:?}", c.transform);
    assert_eq!(c.warp.as_ref(), Some(&warp));
}
