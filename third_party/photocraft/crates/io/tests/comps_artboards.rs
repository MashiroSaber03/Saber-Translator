//! Layer comps (resource 1065 + per-layer `cmls`) and artboards (`artb`) through PSD and
//! `.pcraft`: byte-exact while unchanged, regenerated after edits.

#[cfg(feature = "corpus")]
use std::path::PathBuf;

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::comps::capture_states;
use photocraft_doc::{Artboard, ArtboardBackground, Document, Layer, LayerComp, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_io::comps_map::LAYER_COMPS;
use photocraft_io::{ExportOptions, export, import};
use photocraft_psd::PsdFile;

#[cfg(feature = "corpus")]
fn corpus(rel: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd").join(rel);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}: run `cargo xtask corpus --all`", p.display()))
}

fn to_psd(doc: &Document) -> Vec<u8> {
    export(doc, "x.psd", &ExportOptions::default()).expect("export").bytes
}

fn layer_block<'a>(f: &'a PsdFile, name: &str, key: &[u8; 4]) -> Option<&'a [u8]> {
    f.layers().iter().find(|r| r.name() == name).and_then(|r| r.block(key)).map(|b| b.data.as_slice())
}

#[cfg(feature = "corpus")]
#[test]
fn corpus_layer_comps_import_and_verbatim_export() {
    let bytes = corpus("ag-psd/read-write/layer-comps/src.psd");
    let src = PsdFile::from_bytes(&bytes).unwrap();
    let doc = import("src.psd", &bytes).unwrap().document;
    let names: Vec<_> = doc.layer_comps.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["Layer Comp 1", "Layer Comp 2", "Layer Comp 3", "Layer Comp 4", "Layer Comp 5"]);
    assert_eq!(doc.layer_comps[3].comment, "some comments");
    assert_eq!(doc.layer_comps[1].captured_info(), 1);
    assert_eq!(doc.layer_comps[4].captured_info(), 7);
    assert_eq!(doc.last_applied_comp, Some(doc.layer_comps[3].id));
    // Comp 2 hides some layers; comp 1 moves one by (-48, -35).
    assert!(doc.layer_comps[1].states.iter().any(|s| s.visible == Some(false)));
    let moved: Vec<_> = doc.layer_comps[0].states.iter().filter(|s| s.position.is_some()).collect();
    assert_eq!(moved.len(), 1);
    let l = doc.layer(moved[0].layer).unwrap();
    let here = photocraft_doc::comps::layer_position(l).unwrap();
    assert_eq!(moved[0].position, Some((here.0 - 48, here.1 - 35)));

    // Unchanged: resource 1065 and every layer's shmd block are written back byte for byte.
    let out = PsdFile::from_bytes(&to_psd(&doc)).unwrap();
    assert_eq!(out.resource(LAYER_COMPS).unwrap().data, src.resource(LAYER_COMPS).unwrap().data);
    for r in src.layers().iter().filter(|r| r.block(b"shmd").is_some()) {
        assert_eq!(layer_block(&out, &r.name(), b"shmd"), r.block(b"shmd").map(|b| b.data.as_slice()), "layer {}", r.name());
    }

    // Edited: regenerated, and it decodes back to the edited comps.
    let mut edited = doc.clone();
    edited.layer_comps[0].name = "Renamed".into();
    edited.layer_comps.remove(2);
    let bytes2 = to_psd(&edited);
    let out2 = PsdFile::from_bytes(&bytes2).unwrap();
    assert_ne!(out2.resource(LAYER_COMPS).unwrap().data, src.resource(LAYER_COMPS).unwrap().data);
    let back = import("x.psd", &bytes2).unwrap().document;
    assert_eq!(back.layer_comps.len(), 4);
    assert_eq!(back.layer_comps[0].name, "Renamed");
    for (a, b) in edited.layer_comps.iter().zip(&back.layer_comps) {
        assert_eq!((a.id, a.captured_info(), &a.comment), (b.id, b.captured_info(), &b.comment));
        let pick = |c: &LayerComp| c.states.iter().map(|s| (s.visible, s.position)).collect::<Vec<_>>();
        assert_eq!(pick(a), pick(b), "comp {}", a.name);
    }
    assert_eq!(back.last_applied_comp, edited.last_applied_comp);
    // Other metadata items in shmd (`cust`) survive regeneration.
    let r = out2.layers().iter().find(|r| r.block(b"shmd").is_some()).unwrap();
    let items = photocraft_psd::metadata::parse_shmd(&r.block(b"shmd").unwrap().data).unwrap();
    assert!(items.iter().any(|i| &i.key == b"cust"));
    assert!(items.iter().any(|i| &i.key == b"cmls"));

    // All comps deleted: the resource and the cmls items go away.
    let mut none = doc.clone();
    none.layer_comps.clear();
    none.last_applied_comp = None;
    none.last_document_state = None;
    let out3 = PsdFile::from_bytes(&to_psd(&none)).unwrap();
    assert!(out3.resource(LAYER_COMPS).is_none());
    for r in out3.layers() {
        if let Some(b) = r.block(b"shmd") {
            assert!(photocraft_psd::metadata::parse_shmd(&b.data).unwrap().iter().all(|i| &i.key != b"cmls"));
        }
    }
}

#[cfg(feature = "corpus")]
#[test]
fn corpus_artboards_import_and_verbatim_export() {
    let bytes = corpus("psd-tools/gradient-sizes.psd");
    let src = PsdFile::from_bytes(&bytes).unwrap();
    let doc = import("g.psd", &bytes).unwrap().document;
    let boards = doc.artboards();
    assert_eq!(boards.len(), 3);
    let rects: Vec<Rect> = boards.iter().map(|b| b.2.rect).collect();
    assert!(rects.contains(&Rect::new(0, 0, 64, 64)));
    assert!(rects.contains(&Rect::new(0, 73, 64, 105)));
    assert!(rects.contains(&Rect::new(73, 0, 105, 64)));
    assert!(boards.iter().all(|b| b.2.background == ArtboardBackground::White));

    let out = PsdFile::from_bytes(&to_psd(&doc)).unwrap();
    for r in src.layers().iter().filter(|r| r.block(b"artb").is_some()) {
        assert_eq!(layer_block(&out, &r.name(), b"artb"), r.block(b"artb").map(|b| b.data.as_slice()), "artboard {}", r.name());
    }
    // A changed board is regenerated.
    let mut edited = doc.clone();
    let id = edited.artboards()[0].0;
    let a = edited.layer_mut(id).unwrap().artboard_mut().unwrap();
    a.background = ArtboardBackground::Black;
    a.rect = a.rect.translate(5, 0);
    let want = a.clone();
    let back = import("x.psd", &to_psd(&edited)).unwrap().document;
    assert!(back.artboards().iter().any(|b| *b.2 == want));
}

fn synthetic(depth: SampleType) -> Document {
    let mut d = Document::new("s", Size::new(80, 40), ColorMode::Rgb, depth);
    let fmt = PixelFormat::new(ColorMode::Rgb, depth, true);
    let mut a = Layer::raster("A", fmt);
    a.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 20, 20), &[1.0, 0.0, 0.0, 1.0]);
    let mut b = Layer::raster("B", fmt);
    b.surface_mut().unwrap().fill_rect(Rect::new(44, 4, 60, 20), &[0.0, 0.0, 1.0, 1.0]);
    let mut g1 = Layer::group("Board 1", vec![a]);
    let mut g2 = Layer::group("Board 2", vec![b]);
    if let LayerContent::Group(g) = &mut g1.content {
        g.artboard = Some(Artboard { rect: Rect::new(0, 0, 40, 40), background: ArtboardBackground::White, preset: "Custom".into() });
    }
    if let LayerContent::Group(g) = &mut g2.content {
        g.artboard =
            Some(Artboard { rect: Rect::new(40, 0, 80, 40), background: ArtboardBackground::Custom(Color::rgb(0.0, 1.0, 0.0)), preset: String::new() });
    }
    d.layers = vec![g1, g2];
    let states = capture_states(&d);
    d.layer_comps.push(LayerComp {
        id: 11,
        name: "Both".into(),
        comment: "all".into(),
        apply_visibility: true,
        apply_position: true,
        apply_appearance: false,
        states,
    });
    d.layers[1].visible = false;
    if let Some(ch) = d.layers[0].children_mut() {
        ch[0].blend = BlendMode::Multiply;
    }
    let mut states = capture_states(&d);
    states.retain(|s| s.layer != d.layers[0].id);
    d.layer_comps.push(LayerComp {
        id: 12,
        name: "One".into(),
        comment: String::new(),
        apply_visibility: true,
        apply_position: false,
        apply_appearance: true,
        states,
    });
    d.last_applied_comp = Some(12);
    d
}

#[test]
fn synthetic_psd_roundtrip_all_depths() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let d = synthetic(depth);
        let back = import("x.psd", &to_psd(&d)).unwrap().document;
        let boards: Vec<_> = back.artboards().into_iter().map(|b| (b.1.to_string(), b.2.clone())).collect();
        let want: Vec<_> = d.artboards().into_iter().map(|b| (b.1.to_string(), b.2.clone())).collect();
        assert_eq!(boards, want, "{depth:?}");
        assert_eq!(back.layer_comps.len(), 2);
        assert_eq!(back.last_applied_comp, Some(12));
        for (a, b) in d.layer_comps.iter().zip(&back.layer_comps) {
            assert_eq!((a.id, &a.name, &a.comment, a.captured_info()), (b.id, &b.name, &b.comment, b.captured_info()));
            // Layer ids differ after import: compare by layer name.
            let named = |doc: &Document, c: &LayerComp| {
                let mut v: Vec<_> = c.states.iter().map(|s| (doc.layer(s.layer).unwrap().name.clone(), s.visible, s.position)).collect();
                v.sort_by(|x, y| x.0.cmp(&y.0));
                v
            };
            assert_eq!(named(&d, a), named(&back, b), "{depth:?} comp {}", a.name);
        }
        // Re-exporting the imported document is byte-stable for comps and artboards.
        let f1 = PsdFile::from_bytes(&to_psd(&back)).unwrap();
        let f2 = PsdFile::from_bytes(&to_psd(&import("x.psd", &to_psd(&back)).unwrap().document)).unwrap();
        assert_eq!(f1.resource(LAYER_COMPS).map(|r| &r.data), f2.resource(LAYER_COMPS).map(|r| &r.data));
        for r in f1.layers() {
            for key in [b"artb", b"shmd"] {
                assert_eq!(r.block(key).map(|b| &b.data), layer_block(&f2, &r.name(), key).map(|d| d.to_vec()).as_ref(), "{}", r.name());
            }
        }
    }
}

#[test]
fn synthetic_pcraft_roundtrip_keeps_everything() {
    let mut d = synthetic(SampleType::U16);
    d.last_document_state = Some(LayerComp {
        id: 0,
        name: "Last Document State".into(),
        comment: String::new(),
        apply_visibility: true,
        apply_position: true,
        apply_appearance: true,
        states: capture_states(&d),
    });
    let bytes = photocraft_format::save_to_bytes(&d, &Default::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    assert_eq!(back.layer_comps, d.layer_comps);
    assert_eq!(back.last_applied_comp, d.last_applied_comp);
    assert_eq!(back.last_document_state, d.last_document_state);
    assert_eq!(back.artboards().len(), 2);
    assert_eq!(back.layers[1].artboard(), d.layers[1].artboard());
    // Remapped ids still point at the same layers.
    let fresh = photocraft_format::load_from_bytes_with(&bytes, &photocraft_format::LoadOptions { preserve_ids: false, ..Default::default() }).unwrap();
    let c = &fresh.layer_comps[0];
    assert_eq!(c.states.len(), fresh.layer_count());
    assert!(c.states.iter().all(|s| fresh.layer(s.layer).is_some()));

    // Native identity is independent of ZIP entry order and the filename extension.
    let zip = photocraft_format::zip::ZipReader::new(&bytes).unwrap();
    let mut reordered = photocraft_format::zip::ZipWriter::new();
    reordered.add("extra.txt", b"not the manifest").unwrap();
    for entry in &zip.entries {
        reordered.add(&entry.name, &zip.read(entry, bytes.len()).unwrap()).unwrap();
    }
    let reordered = reordered.finish().unwrap();
    for name in ["x.PCRAFT", "no-extension"] {
        let imported = import(name, &reordered).unwrap().document;
        assert_eq!(imported.layer_comps, d.layer_comps);
        assert_eq!(imported.layers[1].artboard(), d.layers[1].artboard());
    }

    let png = export(&d, "x.png", &ExportOptions::default()).unwrap().bytes;
    assert!(import("pcraft", &png).is_ok());
    assert!(matches!(import("broken.pcraft", b"not a native bundle"), Err(photocraft_io::IoError::Pcraft(_))));
}

/// A layer at (4, 4) whose preserved `shmd` block offsets it by `(dx, 0)` in comp 1, plus the
/// matching resource 1065: what a PSD with that comp data imports to before decoding comps.
fn doc_with_raw_comp_offset(dx: i32) -> Document {
    use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value, VersionedDescriptor};
    use photocraft_psd::metadata::{MetadataItem, write_shmd};
    let obj = |class: &str| Descriptor { name: UnicodeString(vec![0]), ..Descriptor::new(class) };
    let mut d = Document::new("s", Size::new(32, 32), ColorMode::Rgb, SampleType::U8);
    let mut l = Layer::raster("A", PixelFormat::RGBA8);
    l.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 8, 8), &[1.0, 0.0, 0.0, 1.0]);
    let setting = obj("null")
        .with("enab", Value::Boolean(false))
        .with("compList", Value::List(vec![Value::Integer(1)]))
        .with("Ofst", Value::Descriptor(obj("null").with("Hrzn", Value::Integer(dx)).with("Vrtc", Value::Integer(0))));
    let cmls = VersionedDescriptor::new(Descriptor::new("null").with("layerSettings", Value::List(vec![Value::Descriptor(setting)]))).to_bytes();
    l.psd_blocks = vec![(*b"shmd", std::sync::Arc::new(write_shmd(&[MetadataItem::new(*b"cmls", cmls)])))];
    d.layers.push(l);
    let comp = Descriptor::new("Comp")
        .with("Nm  ", Value::Text(UnicodeString::new_nul("One")))
        .with("compID", Value::Integer(1))
        .with("capturedInfo", Value::Integer(3));
    let res = VersionedDescriptor::new(Descriptor::new("CompList").with("list", Value::List(vec![Value::Descriptor(comp)]))).to_bytes();
    d.metadata.psd_resources.push((LAYER_COMPS, String::new(), std::sync::Arc::new(res)));
    d
}

#[test]
fn out_of_range_comp_offsets_import_and_export_without_panic() {
    // #1017: an `Ofst` that puts the layer past i32 range panicked on import and on export (the
    // unchanged-comps check decodes the preserved data). The position is dropped with a warning,
    // the rest of the comp survives, and the file data is still written back verbatim.
    let mut src = doc_with_raw_comp_offset(i32::MAX);
    (src.layer_comps, src.last_applied_comp, src.last_document_state) =
        photocraft_io::comps_map::comps_from_psd(src.metadata.psd_resources.last().map(|r| r.2.as_slice()), &src);
    let bytes = to_psd(&src);
    let imported = import("x.psd", &bytes).unwrap();
    assert!(imported.warnings.iter().any(|w| w.contains("layer comp position")), "{:?}", imported.warnings);
    let c = &imported.document.layer_comps[0];
    assert_eq!((c.name.as_str(), c.states.len()), ("One", 1));
    assert_eq!((c.states[0].visible, c.states[0].position), (Some(false), None));
    let a = PsdFile::from_bytes(&bytes).unwrap();
    let b = PsdFile::from_bytes(&to_psd(&imported.document)).unwrap();
    assert_eq!(layer_block(&a, "A", b"shmd"), layer_block(&b, "A", b"shmd"));

    // Export of a stored position whose offset from the layer is out of range (no preserved
    // data, so `cmls` is regenerated): the offset is left out and reads back as no position.
    let mut d = Document::new("s", Size::new(32, 32), ColorMode::Rgb, SampleType::U8);
    let mut l = Layer::raster("A", PixelFormat::RGBA8);
    l.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 8, 8), &[1.0, 0.0, 0.0, 1.0]);
    let st = photocraft_doc::CompLayerState { layer: l.id, visible: Some(false), position: Some((i32::MIN, 0)), appearance: None };
    d.layers.push(l);
    d.layer_comps.push(LayerComp {
        id: 1,
        name: "One".into(),
        comment: String::new(),
        apply_visibility: true,
        apply_position: true,
        apply_appearance: false,
        states: vec![st],
    });
    let back = import("x.psd", &to_psd(&d)).unwrap().document;
    let s = &back.layer_comps[0].states[0];
    assert_eq!((s.visible, s.position), (Some(false), None));

    // Control: an ordinary offset imports as layer origin + offset, without a warning.
    let mut ok = doc_with_raw_comp_offset(6);
    (ok.layer_comps, ok.last_applied_comp, ok.last_document_state) =
        photocraft_io::comps_map::comps_from_psd(ok.metadata.psd_resources.last().map(|r| r.2.as_slice()), &ok);
    let imported = import("x.psd", &to_psd(&ok)).unwrap();
    assert!(!imported.warnings.iter().any(|w| w.contains("layer comp")), "{:?}", imported.warnings);
    assert_eq!(imported.document.layer_comps[0].states[0].position, Some((10, 4)));
}
