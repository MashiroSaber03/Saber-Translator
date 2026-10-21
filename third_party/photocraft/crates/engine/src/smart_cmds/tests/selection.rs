use super::*;
use photocraft_color::{ColorMode, SampleType};

fn pixel(doc: &Document, name: &str, bounds: Rect, rgba: [f32; 4]) -> Layer {
    let mut layer = Layer::raster(name, doc.pixel_format());
    layer.surface_mut().unwrap().fill_rect(bounds, &photocraft_raster::from_rgba(&doc.pixel_format(), rgba));
    layer
}

fn selected_session(doc: Document, ids: &[LayerId]) -> Session {
    let mut s = Session::new();
    s.add_document(doc, None);
    crate::layer_multi_cmds::set_selection(&mut s, ids.to_vec(), ids.last().copied(), None).unwrap();
    s
}

fn contents(s: &Session) -> Document {
    let SmartSource::Embedded { file_name, bytes } = active_smart(s).source else { panic!() };
    decode_source(&file_name, &bytes).unwrap()
}

#[test]
fn selected_pixels_become_one_editable_smart_object_in_every_mode_and_depth() {
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for depth in SampleType::ALL {
            let mut doc = Document::new("Selected", Size::new(W as u32, H as u32), mode, depth);
            let profile = match mode {
                ColorMode::Rgb => photocraft_cms::builtin::Builtin::Srgb,
                ColorMode::Grayscale => photocraft_cms::builtin::Builtin::SGray,
                ColorMode::Cmyk => photocraft_cms::builtin::Builtin::CoatedCmyk,
                _ => photocraft_cms::builtin::Builtin::LabD50,
            };
            doc.icc_profile = Some(profile.profile().to_bytes());
            doc.resolution_dpi = 300.0;
            let a = pixel(&doc, "Bottom", Rect::new(-3, 4, 24, 26), [0.8, 0.2, 0.1, 1.0]);
            let mut b = pixel(&doc, "Top", Rect::new(10, 12, 36, 32), [0.1, 0.4, 0.8, 0.7]);
            b.mask = Some(LayerMask::reveal_all());
            b.mask.as_mut().unwrap().surface.fill_rect(Rect::new(14, 16, 20, 22), &[0.5]);
            b.locks.all = true;
            b.effects.enabled = true;
            b.effects.items.push(photocraft_doc::Effect::ColorOverlay {
                common: photocraft_doc::FxCommon { enabled: true, blend: BlendMode::Normal, opacity: 0.3 },
                color: photocraft_color::Color::WHITE,
            });
            let mut hidden = pixel(&doc, "Hidden", Rect::new(50, 40, 58, 44), [0.2, 0.9, 0.3, 1.0]);
            hidden.visible = false;
            let ids = [a.id, b.id, hidden.id];
            doc.layers = vec![a, b, hidden];
            let mut selection = Surface::new(PixelFormat::GRAY8);
            selection.fill_rect(Rect::new(0, 0, 12, 12), &[0.5]);
            doc.selection = Some(selection);
            let original = doc.clone();
            // Selection order must not change stacking order.
            let mut s = selected_session(doc, &[ids[2], ids[1], ids[0]]);
            let before = flat(&s);
            let mut grouped = selected_session(original.clone(), &ids);
            grouped.execute("layer.groupLayers", json!({})).unwrap();
            convert(&mut grouped);
            let id = LayerId(convert(&mut s));
            assert_eq!(s.active().unwrap().doc.layers.len(), 1, "{mode:?} {depth:?}");
            assert_eq!(s.active().unwrap().selected_layers(), vec![id]);
            assert_eq!(s.active().unwrap().doc.selection, original.selection);
            // The existing cache conversion quantizes channels and round-trips CMYK through
            // RGB. Combining a selection must introduce no loss beyond grouping then converting.
            assert_eq!(flat(&s), flat(&grouped), "selection conversion matches Group Layers then Convert: {mode:?} {depth:?}");
            if matches!(mode, ColorMode::Rgb | ColorMode::Grayscale) {
                assert!(max_diff(&flat(&s), &before) <= 2.0 / 255.0, "{mode:?} {depth:?}");
            }
            let inner = contents(&s);
            assert_eq!((inner.mode, inner.depth), (mode, depth));
            assert_eq!(inner.icc_profile, original.icc_profile);
            assert_eq!(inner.resolution_dpi, original.resolution_dpi);
            // Like Photoshop, the contents hold the selected layers at their top level, in stacking
            // order, with no wrapper group.
            assert_eq!(inner.layers.iter().map(|l| l.id).collect::<Vec<_>>(), ids, "{mode:?} {depth:?}");
            let sm = active_smart(&s);
            for source in &original.layers {
                let mut shifted = source.clone();
                shift_layer(&mut shifted, -(sm.transform.m[4] as i32), -(sm.transform.m[5] as i32));
                assert_eq!(inner.layer(source.id).unwrap(), &shifted, "editable source preserved");
            }
            assert!(s.undo());
            assert_eq!(*s.active().unwrap().doc, original);
            assert_eq!(s.active().unwrap().selected_layers(), ids);
            assert!(!s.undo(), "conversion is one undo step");
            assert!(s.redo());
            assert_eq!(s.active().unwrap().selected_layers(), vec![id]);
            assert_eq!(contents(&s), inner);
        }
    }
}

#[test]
fn selected_group_and_descendant_are_embedded_once() {
    let mut doc = Document::new("Groups", Size::new(W as u32, H as u32), ColorMode::Rgb, SampleType::U16);
    let a = pixel(&doc, "Child", Rect::new(0, 0, 16, 16), [1.0, 0.0, 0.0, 1.0]);
    let a_id = a.id;
    let group = Layer::group("Original group", vec![a]);
    let group_id = group.id;
    let b = pixel(&doc, "Outside", Rect::new(16, 0, 32, 16), [0.0, 0.0, 1.0, 1.0]);
    let b_id = b.id;
    doc.layers = vec![group, b];
    let mut s = selected_session(doc, &[group_id, a_id, b_id]);
    convert(&mut s);
    assert_eq!(s.active().unwrap().doc.layers.len(), 1);
    let inner = contents(&s);
    assert_eq!(inner.walk().iter().filter(|(_, _, l)| l.id == a_id).count(), 1);
    assert_eq!(inner.layers.iter().map(|l| l.id).collect::<Vec<_>>(), vec![group_id, b_id], "a selected group stays a group");
    assert_eq!(inner.layer(group_id).unwrap().children().unwrap()[0].id, a_id);
}

#[test]
fn mixed_parent_selection_keeps_unselected_siblings_and_uses_the_top_selected_parent() {
    let mut doc = Document::new("Parents", Size::new(W as u32, H as u32), ColorMode::Rgb, SampleType::U8);
    let a = pixel(&doc, "Selected lower", Rect::new(0, 0, 8, 8), [1.0, 0.0, 0.0, 1.0]);
    let b = pixel(&doc, "Selected upper", Rect::new(8, 0, 16, 8), [0.0, 0.0, 1.0, 1.0]);
    let keep_a = pixel(&doc, "Keep lower", Rect::new(0, 8, 8, 16), [0.0, 1.0, 0.0, 1.0]);
    let keep_b = pixel(&doc, "Keep upper", Rect::new(8, 8, 16, 16), [1.0, 1.0, 0.0, 1.0]);
    let (a_id, b_id, keep_a_id, keep_b_id) = (a.id, b.id, keep_a.id, keep_b.id);
    let lower = Layer::group("Lower parent", vec![a, keep_a.clone()]);
    let upper = Layer::group("Upper parent", vec![b, keep_b.clone()]);
    let (lower_id, upper_id) = (lower.id, upper.id);
    doc.layers = vec![lower, upper];
    let original = doc.clone();
    let mut s = selected_session(doc, &[b_id, a_id]);
    let before = flat(&s);
    let so = LayerId(convert(&mut s));
    let doc = &s.active().unwrap().doc;
    assert_eq!(doc.layers.iter().map(|l| l.id).collect::<Vec<_>>(), vec![lower_id, upper_id]);
    assert_eq!(doc.layer(lower_id).unwrap().children().unwrap().iter().map(|l| l.id).collect::<Vec<_>>(), vec![keep_a_id]);
    assert_eq!(doc.layer(upper_id).unwrap().children().unwrap().iter().map(|l| l.id).collect::<Vec<_>>(), vec![so, keep_b_id]);
    assert_eq!(doc.layer(keep_a_id).unwrap(), &keep_a);
    assert_eq!(doc.layer(keep_b_id).unwrap(), &keep_b);
    assert_eq!(flat(&s), before);
    let inner = contents(&s);
    assert!(inner.layer(a_id).is_some() && inner.layer(b_id).is_some());
    assert!(s.undo());
    assert_eq!(*s.active().unwrap().doc, original);
}

#[test]
fn explicit_layer_converts_only_that_layer_despite_multi_selection() {
    let mut doc = Document::new("Explicit", Size::new(W as u32, H as u32), ColorMode::Rgb, SampleType::F32);
    let a = pixel(&doc, "A", Rect::new(0, 0, 8, 8), [1.0, 0.0, 0.0, 1.0]);
    let b = pixel(&doc, "B", Rect::new(8, 0, 16, 8), [0.0, 0.0, 1.0, 1.0]);
    let (a_id, b_id) = (a.id, b.id);
    doc.layers = vec![a, b.clone()];
    let mut s = selected_session(doc, &[a_id, b_id]);
    s.execute("layer.smartObjects.convertToSmartObject", json!({"layer": a_id.0})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), 2);
    assert_eq!(s.active().unwrap().doc.layer(b_id).unwrap(), &b);
    assert_eq!(contents(&s).layers.len(), 1);
    assert_eq!(contents(&s).layers[0].id, a_id);
}

#[test]
fn combined_contents_survive_native_and_psd_save_open() {
    for depth in SampleType::ALL {
        let mut doc = Document::new("Saved pair", Size::new(W as u32, H as u32), ColorMode::Rgb, depth);
        let a = pixel(&doc, "Bottom", Rect::new(-2, 4, 20, 20), [1.0, 0.0, 0.0, 1.0]);
        let b = pixel(&doc, "Top", Rect::new(10, 8, 24, 24), [0.0, 0.0, 1.0, 1.0]);
        let ids = [a.id, b.id];
        doc.layers = vec![a, b];
        let mut s = selected_session(doc, &ids);
        let id = LayerId(convert(&mut s));
        let original = s.active().unwrap().doc.clone();
        let native = photocraft_format::save_to_bytes(&original, &Default::default()).unwrap();
        let native_back = photocraft_format::load_from_bytes(&native).unwrap();
        assert_eq!(smart(&original, id).unwrap(), smart(&native_back, id).unwrap());
        let psd = photocraft_io::export(&original, "pair.psd", &Default::default()).unwrap();
        let back = photocraft_io::import("pair.psd", &psd.bytes).unwrap().document;
        assert_eq!(back.depth, depth);
        assert_eq!(back.layers.len(), 1);
        let LayerContent::Smart(sm) = &back.layers[0].content else { panic!("PSD lost the smart object") };
        let (name, bytes) = source_bytes(&back.metadata, &sm.source).unwrap();
        let inner = decode_source(&name, &bytes).unwrap();
        let names: Vec<_> = inner.walk().into_iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Raster(_))).map(|(_, _, l)| l.name.clone()).collect();
        assert_eq!(names, vec!["Bottom", "Top"], "{depth:?}");
    }
}

#[test]
fn selected_ancestor_in_another_parent_cannot_remove_the_new_smart_object() {
    let mut doc = Document::new("Ancestor", Size::new(W as u32, H as u32), ColorMode::Rgb, SampleType::U8);
    let a = pixel(&doc, "A", Rect::new(0, 0, 8, 8), [1.0, 0.0, 0.0, 1.0]);
    let b = pixel(&doc, "B", Rect::new(8, 0, 16, 8), [0.0, 0.0, 1.0, 1.0]);
    let keep = pixel(&doc, "Unselected sibling", Rect::new(0, 8, 8, 16), [0.0, 1.0, 0.0, 1.0]);
    let (a_id, b_id, keep_id) = (a.id, b.id, keep.id);
    let lower = Layer::group("Selected parent", vec![a]);
    let upper = Layer::group("Unselected parent", vec![b, keep]);
    let (lower_id, upper_id) = (lower.id, upper.id);
    doc.layers = vec![lower, upper];
    let mut s = selected_session(doc, &[lower_id, a_id, b_id]);
    let so = LayerId(convert(&mut s));
    let outer = &s.active().unwrap().doc;
    assert_eq!(outer.layers.len(), 1);
    assert_eq!(outer.layers[0].id, upper_id);
    assert_eq!(outer.layer(upper_id).unwrap().children().unwrap().iter().map(|l| l.id).collect::<Vec<_>>(), vec![so, keep_id]);
    assert!(outer.layer(keep_id).is_some());
    let inner = contents(&s);
    assert!(inner.layer(lower_id).is_some() && inner.layer(a_id).is_some() && inner.layer(b_id).is_some());
    assert!(inner.layer(keep_id).is_none());
}

#[test]
fn an_all_hidden_selection_is_rejected_without_changing_document_selection_or_history() {
    let mut doc = Document::new("Hidden pair", Size::new(W as u32, H as u32), ColorMode::Rgb, SampleType::U8);
    let mut a = pixel(&doc, "Hidden A", Rect::new(0, 0, 8, 8), [1.0, 0.0, 0.0, 1.0]);
    let mut b = pixel(&doc, "Hidden B", Rect::new(8, 0, 16, 8), [0.0, 0.0, 1.0, 1.0]);
    a.visible = false;
    b.visible = false;
    let ids = [a.id, b.id];
    doc.layers = vec![a, b];
    let original = doc.clone();
    let mut s = selected_session(doc, &ids);
    let err = s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap_err();
    assert!(err.to_string().contains("empty"), "{err}");
    assert_eq!(*s.active().unwrap().doc, original);
    assert_eq!(s.active().unwrap().selected_layers(), ids);
    assert_eq!(s.active().unwrap().active_layer, Some(ids[1]));
    assert_eq!(s.active().unwrap().history.past_len(), 0);
    assert!(!s.undo());
}
