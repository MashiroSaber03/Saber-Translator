//! Clipboard copies carry only the linked smart-object records their layers reference.
use photocraft_doc::{Document, LayerContent, LayerId, SmartSource};
use photocraft_engine::{Session, smart_cmds::source_bytes};
use photocraft_geom::Rect;
use serde_json::json;
use std::sync::Arc;

fn fixture(depth: u64) -> Document {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 6, "depth": depth})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("synthetic color", |doc, active| {
        doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 8, 6), &[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    let bytes = photocraft_io::export(&s.active().unwrap().doc, "psd", &Default::default()).unwrap().bytes;
    let doc = photocraft_io::import("synthetic.psd", &bytes).unwrap().document;
    assert!(doc.walk().iter().any(|(_, _, l)| matches!(l.content, LayerContent::Smart(_))));
    doc
}
fn smart_id(doc: &Document) -> LayerId {
    doc.walk().into_iter().find_map(|(_, _, l)| matches!(l.content, LayerContent::Smart(_)).then_some(l.id)).unwrap()
}
fn bytes(doc: &Document, id: LayerId) -> Arc<Vec<u8>> {
    let LayerContent::Smart(sm) = &doc.layer(id).unwrap().content else { panic!("expected smart object") };
    source_bytes(&doc.metadata, &sm.source).unwrap().1
}
fn hash(bytes: &[u8]) -> (usize, String) {
    (bytes.len(), blake3::hash(bytes).to_hex().to_string())
}
fn source_session(doc: Document) -> (Session, LayerId) {
    let mut s = Session::new();
    let id = smart_id(&doc);
    s.add_document(doc, None);
    s.select_layer(id).unwrap();
    (s, id)
}
fn roundtrip(doc: &Document, format: &str) -> Document {
    if format == "pcraft" {
        photocraft_format::load_from_bytes(&photocraft_format::save_to_bytes(doc, &Default::default()).unwrap()).unwrap()
    } else {
        let out = photocraft_io::export(doc, format, &Default::default()).unwrap();
        assert!(out.warnings.is_empty(), "{format}: {:?}", out.warnings);
        photocraft_io::import(&format!("clipboard.{format}"), &out.bytes).unwrap().document
    }
}

#[test]
fn clipboard_valid_psd_smart_sources_open_reopen_and_undo() {
    for depth in [8, 16, 32] {
        for key in [*b"lnk2", *b"lnk3", *b"lnkD"] {
            let mut doc = fixture(depth);
            for (_, k, _) in &mut doc.metadata.psd_global_blocks {
                if matches!(&*k, b"lnk2" | b"lnk3" | b"lnkD") {
                    *k = key;
                }
            }
            let (mut s, original) = source_session(doc);
            let expected = bytes(&s.active().unwrap().doc, original);
            s.execute("edit.copy", json!({})).unwrap();
            s.execute("file.new", json!({"width": 8, "height": 6, "depth": depth})).unwrap();
            let before = s.active().unwrap().doc.clone();
            let pasted = LayerId(s.execute("edit.paste", json!({})).unwrap()["layer"].as_u64().unwrap());
            assert_eq!(hash(&bytes(&s.active().unwrap().doc, pasted)), hash(&expected), "depth {depth}, key {key:?}");
            let after = s.active().unwrap().doc.clone();
            assert!(s.undo());
            assert_eq!(s.active().unwrap().doc, before);
            assert!(s.redo());
            assert_eq!(s.active().unwrap().doc, after);
            s.execute("layer.smartObjects.editContents", json!({"layer": pasted.0})).unwrap();
            assert_eq!(photocraft_compose::flatten(&s.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0]);
            s.set_active(1);
            for format in ["pcraft", "psd", "psb"] {
                let back = roundtrip(&s.active().unwrap().doc, format);
                let (mut reopened, id) = source_session(back);
                reopened.execute("layer.smartObjects.editContents", json!({"layer": id.0})).unwrap();
                assert_eq!(photocraft_compose::flatten(&reopened.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0], "depth {depth}, {format}");
            }
        }
    }
}

#[test]
fn clipboard_batch_and_same_document_duplicates_keep_sources() {
    for depth in [8, 16, 32] {
        let (mut s, original) = source_session(fixture(depth));
        let duplicate = s.execute("layer.duplicate", json!({"layer": original.0})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.select", json!({"layer": original.0})).unwrap();
        s.execute("layer.select", json!({"layer": duplicate, "mode": "add"})).unwrap();
        let expected = bytes(&s.active().unwrap().doc, original);
        let count = s.active().unwrap().doc.metadata.psd_global_blocks.len();
        s.execute("edit.copy", json!({})).unwrap();
        let copied = s.clipboard.as_ref().unwrap().layers.as_ref().unwrap();
        assert!(!copied.metadata.psd_global_blocks.is_empty());
        assert!(
            copied.metadata.psd_global_blocks.iter().all(|(_, key, data)| s
                .active()
                .unwrap()
                .doc
                .metadata
                .psd_global_blocks
                .iter()
                .any(|(_, dk, dd)| { dk == key && Arc::ptr_eq(dd, data) })),
            "an exclusive linked block should stay shared, including large PSBs"
        );
        s.execute("edit.paste", json!({})).unwrap();
        assert_eq!(s.active().unwrap().doc.metadata.psd_global_blocks.len(), count, "same-document paste shouldn't duplicate blocks");
        for (_, _, l) in s.active().unwrap().doc.walk() {
            if matches!(l.content, LayerContent::Smart(_)) {
                assert_eq!(hash(&bytes(&s.active().unwrap().doc, l.id)), hash(&expected));
            }
        }
        s.execute("file.new", json!({"width": 8, "height": 6, "depth": depth})).unwrap();
        s.execute("edit.paste", json!({})).unwrap();
        let ids: Vec<_> =
            s.active().unwrap().doc.walk().into_iter().filter_map(|(_, _, l)| matches!(l.content, LayerContent::Smart(_)).then_some(l.id)).collect();
        assert_eq!(ids.len(), 2);
        for id in ids {
            assert_eq!(hash(&bytes(&s.active().unwrap().doc, id)), hash(&expected));
            s.execute("layer.smartObjects.editContents", json!({"layer": id.0})).unwrap();
            assert_eq!(photocraft_compose::flatten(&s.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0]);
            s.set_active(1);
        }
    }
}

#[test]
fn clipboard_converts_parent_without_losing_contents_and_disk_reopen() {
    for mode in ["rgb", "grayscale", "cmyk", "lab"] {
        for depth in [8, 16, 32] {
            let (mut s, id) = source_session(fixture(16));
            let expected = bytes(&s.active().unwrap().doc, id);
            s.execute("edit.copy", json!({})).unwrap();
            s.execute("file.new", json!({"width": 8, "height": 6, "depth": depth, "mode": mode})).unwrap();
            let pasted = LayerId(s.execute("edit.paste", json!({})).unwrap()["layer"].as_u64().unwrap());
            assert_eq!(hash(&bytes(&s.active().unwrap().doc, pasted)), hash(&expected), "{mode}, depth {depth}");
            s.execute("layer.smartObjects.editContents", json!({"layer": pasted.0})).unwrap();
            assert_eq!(photocraft_compose::flatten(&s.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0]);
            s.set_active(1);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("pasted.pcraft");
            s.execute("file.saveACopy", json!({"path": path})).unwrap();
            let mut reopened = Session::new();
            reopened.execute("file.openAs", json!({"path": path})).unwrap();
            let reopened_id = smart_id(&reopened.active().unwrap().doc);
            assert_eq!(hash(&bytes(&reopened.active().unwrap().doc, reopened_id)), hash(&expected));
            reopened.execute("layer.smartObjects.editContents", json!({"layer": reopened_id.0})).unwrap();
            assert_eq!(photocraft_compose::flatten(&reopened.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0]);
        }
    }
}

#[test]
fn clipboard_group_keeps_nested_linked_sources() {
    let mut doc = fixture(16);
    let id = smart_id(&doc);
    let expected = bytes(&doc, id);
    let child = doc.remove(id).unwrap();
    let group = photocraft_doc::Layer::group("synthetic group", vec![child]);
    let group_id = group.id;
    doc.layers.push(group);
    let (mut s, _) = source_session(doc);
    s.select_layer(group_id).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    s.execute("file.new", json!({"width": 8, "height": 6, "depth": 16})).unwrap();
    s.execute("edit.paste", json!({})).unwrap();
    let pasted = smart_id(&s.active().unwrap().doc);
    assert_eq!(hash(&bytes(&s.active().unwrap().doc, pasted)), hash(&expected));
    s.execute("layer.smartObjects.editContents", json!({"layer": pasted.0})).unwrap();
    assert_eq!(photocraft_compose::flatten(&s.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn copy_excludes_unselected_linked_records_from_native_save() {
    let mut doc = fixture(8);
    let original = smart_id(&doc);
    let expected = bytes(&doc, original);
    let mut unselected = doc.layer(original).unwrap().duplicate();
    unselected.name = "Unselected private synthetic document".into();
    unselected.visible = false;
    let LayerContent::Smart(sm) = &mut unselected.content else { panic!("smart") };
    sm.source = SmartSource::Linked { path: "unselected-private-source".into() };
    doc.layers.push(unselected);
    let mut private_child = photocraft_engine::smart_cmds::decode_source("source.psb", &expected).unwrap();
    for layer in &mut private_child.layers {
        if let Some(surface) = layer.surface_mut() {
            surface.fill_rect(Rect::new(0, 0, 8, 6), &[0.0, 0.0, 1.0, 1.0]);
        }
    }
    let private_bytes = photocraft_io::export(&private_child, "psb", &Default::default()).unwrap().bytes;
    assert_ne!(hash(&expected), hash(&private_bytes));
    let private_source =
        photocraft_io::linked::LinkedFile { uuid: "unselected-private-source".into(), file_name: "private-synthetic.psb".into(), bytes: private_bytes.clone() };
    // Files may put both selected and unselected entries in one global linked-layer block.
    let (_, _, block) = doc.metadata.psd_global_blocks.iter_mut().find(|(_, k, _)| matches!(k, b"lnk2" | b"lnk3" | b"lnkD")).unwrap();
    let selected_record = block.clone();
    let mut combined = block.to_vec();
    combined.extend(photocraft_io::linked::encode_linked_file(&private_source));
    combined.extend([0, 0, 0, 0, 0, 0, 1, 0, 9]); // unrelated unreadable tail
    *block = Arc::new(combined);
    let (mut s, _) = source_session(doc);
    s.execute("edit.copy", json!({})).unwrap();
    s.execute("file.new", json!({"width": 8, "height": 6})).unwrap();
    s.execute("edit.paste", json!({})).unwrap();
    let back = roundtrip(&s.active().unwrap().doc, "pcraft");
    let retained = photocraft_io::linked::find_linked_file(&back.metadata, "unselected-private-source");
    if let Some(file) = &retained {
        assert_eq!(hash(&file.bytes), hash(&private_bytes));
    }
    assert!(retained.is_none(), "copying one selected layer must not retain another unselected smart object's embedded file in saved destination .pcraft");
    let linked: Vec<_> = back.metadata.psd_global_blocks.iter().filter(|(_, k, _)| matches!(k, b"lnk2" | b"lnk3" | b"lnkD" | b"lnkE")).collect();
    assert_eq!(linked.len(), 1);
    assert_eq!(hash(&linked[0].2), hash(&selected_record), "unidentified tail bytes must not follow the selected record");
}

#[test]
fn copy_keeps_selected_external_alias_and_unknown_records_without_reading_their_source() {
    for key in [*b"lnk2", *b"lnk3", *b"lnkD", *b"lnkE"] {
        for kind in [*b"liFE", *b"liFA", *b"next"] {
            let mut doc = fixture(8);
            let id = smart_id(&doc);
            let LayerContent::Smart(sm) = &doc.layer(id).unwrap().content else { panic!("smart") };
            let source = sm.source.clone();
            let (_, _, raw) = doc.metadata.psd_global_blocks.iter().find(|(_, k, _)| matches!(k, b"lnk2" | b"lnk3" | b"lnkD")).unwrap();
            let mut selected = raw.to_vec();
            selected[8..12].copy_from_slice(&kind);
            selected[12..16].copy_from_slice(&42u32.to_be_bytes());
            let mut data = photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile {
                uuid: "unselected-external".into(),
                file_name: "unselected.psb".into(),
                bytes: vec![9; 8],
            });
            data.extend(&selected);
            doc.metadata.psd_global_blocks.retain(|(_, k, _)| !matches!(k, b"lnk2" | b"lnk3" | b"lnkD" | b"lnkE"));
            doc.metadata.psd_global_blocks.push((*b"8BIM", key, Arc::new(data)));
            let (mut s, _) = source_session(doc);
            s.execute("edit.copy", json!({})).unwrap();
            s.execute("file.new", json!({"width": 8, "height": 6})).unwrap();
            let pasted = LayerId(s.execute("edit.paste", json!({})).unwrap()["layer"].as_u64().unwrap());
            let LayerContent::Smart(sm) = &s.active().unwrap().doc.layer(pasted).unwrap().content else { panic!("smart") };
            assert_eq!(sm.source, source, "source identity is retained; missing external contents aren't read during Copy");
            let back = roundtrip(&s.active().unwrap().doc, "pcraft");
            let linked: Vec<_> = back.metadata.psd_global_blocks.iter().filter(|(_, k, _)| matches!(k, b"lnk2" | b"lnk3" | b"lnkD" | b"lnkE")).collect();
            assert_eq!(linked.len(), 1);
            assert_eq!(linked[0].1, key);
            assert_eq!(hash(&linked[0].2), hash(&selected));
        }
    }
}

#[test]
fn copy_refuses_a_truncated_selected_record_without_changing_document_or_clipboard() {
    let (mut s, _) = source_session(fixture(8));
    s.execute("edit.copy", json!({})).unwrap();
    let before_clip = s.clipboard.clone().unwrap();
    s.edit("truncate selected linked record", |doc, _| {
        let (_, _, block) = doc.metadata.psd_global_blocks.iter_mut().find(|(_, k, _)| matches!(k, b"lnk2" | b"lnk3" | b"lnkD")).unwrap();
        let mut truncated = block.to_vec();
        truncated[..8].copy_from_slice(&u64::MAX.to_be_bytes());
        *block = Arc::new(truncated);
        Ok(())
    })
    .unwrap();
    let before_doc = s.active().unwrap().doc.clone();
    let error = s.execute("edit.copy", json!({})).unwrap_err();
    assert!(error.to_string().contains("truncated selected"), "{error}");
    assert_eq!(s.active().unwrap().doc, before_doc);
    let after_clip = s.clipboard.as_ref().unwrap();
    assert_eq!(after_clip.bounds, before_clip.bounds);
    assert_eq!(after_clip.layers, before_clip.layers);
}

#[test]
fn repeated_same_document_subset_pastes_do_not_duplicate_embedded_files() {
    let mut doc = fixture(16);
    let original = smart_id(&doc);
    let expected = bytes(&doc, original);
    let (_, _, block) = doc.metadata.psd_global_blocks.iter_mut().find(|(_, k, _)| matches!(k, b"lnk2" | b"lnk3" | b"lnkD")).unwrap();
    let selected_record = block.clone();
    let mut combined = block.to_vec();
    combined.extend(photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile {
        uuid: "unselected".into(),
        file_name: "another.psb".into(),
        bytes: expected.to_vec(),
    }));
    *block = Arc::new(combined);
    let (mut s, _) = source_session(doc);
    let metadata = s.active().unwrap().doc.metadata.psd_global_blocks.clone();
    s.execute("edit.copy", json!({})).unwrap();
    let copied = s.clipboard.as_ref().unwrap().layers.as_ref().unwrap();
    assert_eq!(copied.metadata.psd_global_blocks.len(), 1);
    assert_eq!(hash(&copied.metadata.psd_global_blocks[0].2), hash(&selected_record));
    for _ in 0..3 {
        let before = s.active().unwrap().doc.clone();
        let pasted = LayerId(s.execute("edit.paste", json!({})).unwrap()["layer"].as_u64().unwrap());
        assert_eq!(hash(&bytes(&s.active().unwrap().doc, pasted)), hash(&expected));
        let after = s.active().unwrap().doc.clone();
        assert_eq!(after.metadata.psd_global_blocks.len(), metadata.len(), "subsets should not append a second linked block");
        for ((signature, key, bytes), (expected_signature, expected_key, expected_bytes)) in after.metadata.psd_global_blocks.iter().zip(&metadata) {
            assert_eq!(signature, expected_signature);
            assert_eq!(key, expected_key);
            assert_eq!(hash(bytes), hash(expected_bytes));
        }
        assert!(s.undo());
        assert_eq!(s.active().unwrap().doc, before);
        assert!(s.redo());
        assert_eq!(s.active().unwrap().doc, after);
    }
    let back = roundtrip(&s.active().unwrap().doc, "pcraft");
    let ids: Vec<_> = back.metadata.psd_global_blocks.iter().flat_map(|(_, _, data)| photocraft_io::linked::block_uuids(data)).collect();
    assert_eq!(ids.len(), 2, "one record per original file after native save/reopen");
}

#[test]
fn empty_source_ids_do_not_carry_unidentified_global_files() {
    let mut private = Session::new();
    private.execute("file.new", json!({"width":8,"height":6,"background":"black"})).unwrap();
    let bytes = photocraft_format::save_to_bytes(&private.active().unwrap().doc, &Default::default()).unwrap();
    let mut source = Session::new();
    source.execute("file.new", json!({"width":8,"height":6})).unwrap();
    source.execute("layer.new.layer", json!({})).unwrap();
    source.execute("edit.fill", json!({"contents":"foreground"})).unwrap();
    source.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    source
        .edit("malformed empty source", |doc, active| {
            let LayerContent::Smart(sm) = &mut doc.layer_mut(active.unwrap()).unwrap().content else { panic!("smart") };
            sm.source = SmartSource::Linked { path: String::new() };
            doc.metadata.psd_global_blocks.push((
                *b"8BIM",
                *b"lnk2",
                Arc::new(photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile {
                    uuid: String::new(),
                    file_name: "unidentified-private.pcraft".into(),
                    bytes: bytes.clone(),
                })),
            ));
            Ok(())
        })
        .unwrap();
    let state = source.active().unwrap();
    let LayerContent::Smart(sm) = &state.doc.layer(state.active_layer.unwrap()).unwrap().content else { panic!("smart") };
    assert!(photocraft_engine::smart_cmds::source_bytes(&state.doc.metadata, &sm.source).is_none(), "an empty UUID cannot reference this record");
    source.execute("edit.copy", json!({})).unwrap();
    source.execute("file.new", json!({"width":8,"height":6})).unwrap();
    source.execute("edit.paste", json!({})).unwrap();
    let saved = photocraft_format::save_to_bytes(&source.active().unwrap().doc, &Default::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&saved).unwrap();
    assert!(back.metadata.psd_global_blocks.is_empty(), "unidentified empty-ID global bytes must not enter the copied document");
}

#[test]
fn ambiguous_invalid_utf8_ids_do_not_replace_the_clipboard() {
    let mut private = Session::new();
    private.execute("file.new", json!({"width":8,"height":6,"background":"black"})).unwrap();
    let private_bytes = photocraft_format::save_to_bytes(&private.active().unwrap().doc, &Default::default()).unwrap();
    let mut s = Session::new();
    s.execute("file.new", json!({"width":8,"height":6})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"contents":"foreground"})).unwrap();
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let before_clip = s.clipboard.clone().unwrap();
    s.edit("ambiguous source IDs", |doc, active| {
        let LayerContent::Smart(sm) = &mut doc.layer_mut(active.unwrap()).unwrap().content else { panic!("smart") };
        let SmartSource::Embedded { bytes, .. } = &sm.source else { panic!("embedded") };
        let mut first = photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile {
            uuid: "x".into(),
            file_name: "selected.pcraft".into(),
            bytes: bytes.to_vec(),
        });
        let mut second = photocraft_io::linked::encode_linked_file(&photocraft_io::linked::LinkedFile {
            uuid: "x".into(),
            file_name: "unrelated.pcraft".into(),
            bytes: private_bytes.clone(),
        });
        first[17] = 0xff;
        second[17] = 0xfe;
        first.extend(second);
        sm.source = SmartSource::Linked { path: "\u{fffd}".into() };
        doc.metadata.psd_global_blocks.push((*b"8BIM", *b"lnk2", Arc::new(first)));
        Ok(())
    })
    .unwrap();
    let before_doc = s.active().unwrap().doc.clone();
    let copied = s.execute("edit.copy", json!({}));
    if copied.is_ok() {
        let data = &s.clipboard.as_ref().unwrap().layers.as_ref().unwrap().metadata.psd_global_blocks[0].2;
        let files = photocraft_io::linked::parse_linked_files(data);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].uuid, files[1].uuid);
        eprintln!("two distinct malformed raw IDs were retained under one lossy source label");
    }
    let error = copied.expect_err("ambiguous source identity must fail before replacing the clipboard");
    assert!(error.to_string().contains("invalid UTF-8"), "{error}");
    assert_eq!(s.active().unwrap().doc, before_doc);
    assert_eq!(s.clipboard.as_ref().unwrap().layers, before_clip.layers);
}
