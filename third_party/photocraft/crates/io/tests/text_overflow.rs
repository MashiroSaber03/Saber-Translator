//! Visible comic text stays complete and editable in standard Photoshop paragraph layers.
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{
    Document, Layer, LayerContent, TextLayer,
    text::{Orientation, ParagraphRun, ParagraphStyle, TextAlign, TextShape},
};
use photocraft_geom::{Affine, Point, Size};

#[test]
fn editable_comic_export_projects_text_even_with_project_text_metadata() {
    let mut doc = Document::new("comic", Size::new(300, 300), ColorMode::Rgb, SampleType::U8);
    let mut text = TextLayer {
        text: "田‼⁉田⁈⁇田...田".into(),
        font_family: "Inter".into(),
        orientation: Orientation::Vertical,
        shape: TextShape::Box { x: 0.0, y: 0.0, width: 80.0, height: 260.0 },
        size_pt: 23.0,
        overflow_visible: true,
        ..Default::default()
    };
    let layout = photocraft_text::shared().lock().unwrap().layout(&text, doc.resolution_dpi);
    let native = photocraft_text::psd::build_layout_tysh(&text, doc.resolution_dpi, &layout);
    assert_eq!(photocraft_text::psd::text_layer_from_tysh(&native, doc.resolution_dpi).unwrap().text, text.text);
    text.psd_raw = Some(std::sync::Arc::new(native));
    doc.layers.push(Layer::new("comic", LayerContent::Text(text.clone())));

    let bytes = photocraft_io::export(&doc, "comic.psd", &Default::default()).unwrap().bytes;
    let file = photocraft_psd::PsdFile::from_bytes(&bytes).unwrap();
    let txt2 = file.global_blocks.iter().find(|block| block.key == *b"Txt2").unwrap();
    let txt2 = photocraft_text::psd::parse_txt2(&txt2.data).unwrap();
    let objects = txt2.path(&["1", "1"]).unwrap().as_array().unwrap();
    let expected = "田!!\u{200C}!?田?!\u{200C}??田…田";
    assert_eq!(objects[0].path(&["0", "0"]).and_then(photocraft_text::engine_data::Value::as_str), Some(format!("{expected}\r").as_str()));
    let reopened = photocraft_io::import("comic.psd", &bytes).unwrap().document;
    let LayerContent::Text(editable) = &reopened.layers[0].content else { panic!("expected editable text") };
    assert_eq!(editable.text, expected);
    assert_eq!(doc.layers[0].content, LayerContent::Text(text));
}

#[test]
fn narrow_cjk_psd_links_the_strict_rule_to_each_editable_paragraph() {
    let mut doc = Document::new("narrow", Size::new(640, 640), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = 72.0;
    for (orientation, width, height) in [(Orientation::Horizontal, 125.0, 190.0), (Orientation::Vertical, 190.0, 125.0)] {
        let text = TextLayer {
            text: "甲「乙」甲\n丙😀".into(),
            orientation,
            shape: TextShape::Box { x: 0.0, y: 0.0, width, height },
            size_pt: 36.0,
            overflow_visible: true,
            transform: Affine::translate(160.0, 80.0),
            ..Default::default()
        };
        doc.layers.push(Layer::new("narrow", LayerContent::Text(text)));
    }
    let bytes = photocraft_io::export(&doc, "narrow.psd", &Default::default()).unwrap().bytes;
    let file = photocraft_psd::PsdFile::from_bytes(&bytes).unwrap();
    let txt2 = file.global_blocks.iter().find(|block| block.key == *b"Txt2").unwrap();
    let txt2 = photocraft_text::psd::parse_txt2(&txt2.data).unwrap();
    use photocraft_text::engine_data::Value;
    let Value::Dict(resources) = txt2.get("0").unwrap() else { panic!("expected ordered document resources") };
    assert_eq!(resources.iter().map(|(key, _)| key.as_str()).collect::<Vec<_>>(), ["1", "2", "3", "4", "5", "6", "8"]);
    let Value::Dict(document) = txt2.get("1").unwrap() else { panic!("expected ordered text document") };
    assert_eq!(document.iter().map(|(key, _)| key.as_str()).collect::<Vec<_>>(), ["0", "1"]);
    let rules = txt2.path(&["0", "4", "0"]).unwrap().as_array().unwrap();
    let objects = txt2.path(&["1", "1"]).unwrap().as_array().unwrap();
    let frames = txt2.path(&["0", "8", "0"]).unwrap().as_array().unwrap();
    assert_eq!((objects.len(), frames.len()), (2, 2));
    for object in objects {
        let model = object.get("0").unwrap();
        assert_eq!(model.get("0").unwrap().as_str(), Some("甲「乙」甲\r丙😀\r"));
        let paragraphs = model.path(&["5", "0"]).unwrap().as_array().unwrap();
        assert_eq!(paragraphs.len(), 2);
        for (paragraph, len) in paragraphs.iter().zip([6, 4]) {
            assert_eq!(paragraph.get("1").unwrap().as_i64(), Some(len));
            let sheet = paragraph.path(&["0", "0"]).unwrap();
            assert_eq!(sheet.get("6").and_then(Value::as_i64), Some(0));
            let props = sheet.get("5").unwrap();
            let rule = props.get("27").unwrap().as_i64().unwrap() as usize;
            assert_eq!(rules[rule].path(&["0", "0"]).unwrap().as_str(), Some("Hard"));
            assert_eq!(props.get("29").and_then(|value| value.as_bool()), Some(false));
            assert_eq!(props.get("35").and_then(|value| value.as_i64()), Some(1));
        }
        let style = model.path(&["6", "0"]).unwrap().as_array().unwrap().first().unwrap();
        assert_eq!(style.path(&["0", "0", "5"]).and_then(Value::as_i64), Some(0));
    }
    let reopened = photocraft_io::import("narrow.psd", &bytes).unwrap().document;
    for (before, after) in doc.layers.iter().zip(&reopened.layers) {
        let (LayerContent::Text(before), LayerContent::Text(after)) = (&before.content, &after.content) else { panic!("expected editable text") };
        assert_eq!(before.orientation, after.orientation);
        assert_eq!(before.shape, after.shape);
        assert_eq!(before.text, after.text);
    }
}

#[test]
fn block_alignment_survives_native_and_editable_psd_roundtrips() {
    use photocraft_doc::text::BlockAlign;
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        for align in [BlockAlign::Center, BlockAlign::End] {
            for extent in [18.0, 240.0] {
                let shape = if orientation == Orientation::Vertical {
                    TextShape::Box { x: 0.0, y: 0.0, width: extent, height: 100.0 }
                } else {
                    TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: extent }
                };
                let mut text = TextLayer {
                    text: "Hello\nWorld".into(),
                    font_family: "Inter".into(),
                    size_pt: 28.0,
                    shape: shape.clone(),
                    orientation,
                    block_align: align,
                    overflow_visible: true,
                    transform: Affine { m: [1.25, 0.2, -0.1, 1.1, 220.0, 120.0] },
                    ..Default::default()
                };
                let mut doc = Document::new("aligned", Size::new(640, 640), ColorMode::Rgb, SampleType::U8);
                doc.resolution_dpi = 72.0;
                let expected = photocraft_text::shared().lock().unwrap().layout(&text, 72.0);
                photocraft_text::shared().lock().unwrap().render_layer(&mut text, 72.0, doc.pixel_format());
                let original = text.clone();
                doc.layers.push(Layer::new("translation", LayerContent::Text(text)));
                let project = photocraft_io::export(&doc, "aligned.pcraft", &Default::default()).unwrap();
                let loaded = photocraft_io::import("aligned.pcraft", &project.bytes).unwrap().document;
                let LayerContent::Text(native) = &loaded.layers[0].content else { panic!("native text") };
                assert_eq!((native.block_align, &native.shape), (align, &shape));
                let bytes = photocraft_io::export(&doc, "aligned.psd", &Default::default()).unwrap().bytes;
                let mut imported = photocraft_io::import("aligned.psd", &bytes).unwrap().document;
                let LayerContent::Text(psd) = &mut imported.layers[0].content else { panic!("PSD text") };
                let actual = photocraft_text::shared().lock().unwrap().layout(psd, 72.0);
                assert_eq!(actual.glyphs.len(), expected.glyphs.len());
                for (before, after) in expected.glyphs.iter().zip(&actual.glyphs) {
                    let a = original.transform.apply(Point::new(f64::from(before.x), f64::from(before.y)));
                    let b = psd.transform.apply(Point::new(f64::from(after.x), f64::from(after.y)));
                    assert!((a.x - b.x).abs() < 0.2 && (a.y - b.y).abs() < 0.2, "{orientation:?}/{align:?}/{extent}: {a:?} -> {b:?}");
                }
                let bounds = original.cache.as_ref().unwrap().content_bounds();
                assert_eq!(psd.cache.as_ref().unwrap().read_region(bounds), original.cache.as_ref().unwrap().read_region(bounds));
                psd.text.replace_range(0..1, "Z");
                psd.psd_raw = None;
                psd.cache = None;
                photocraft_text::shared().lock().unwrap().render_layer(psd, 72.0, doc.pixel_format());
                let saved = photocraft_io::export(&imported, "edited.psd", &Default::default()).unwrap();
                let reopened = photocraft_io::import("edited.psd", &saved.bytes).unwrap().document;
                let LayerContent::Text(edited) = &reopened.layers[0].content else { panic!("reopened text") };
                assert_eq!(edited.text.replace('\n', ""), "ZelloWorld");
                assert_eq!(doc.layers[0].content, LayerContent::Text(original));
            }
        }
    }
}

#[test]
fn visible_text_keeps_its_project_frame_and_survives_psd_edit_save_reopen() {
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        for align in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
            let mut doc = Document::new("comic", Size::new(480, 480), ColorMode::Rgb, SampleType::U8);
            doc.resolution_dpi = 72.0;
            let shape = if orientation == Orientation::Vertical {
                TextShape::Box { x: 0.0, y: 0.0, width: 18.0, height: 100.0 }
            } else {
                TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 18.0 }
            };
            let content = "One two three four five six";
            let mut original = TextLayer {
                text: content.into(),
                font_family: "Inter".into(),
                size_pt: 36.0,
                shape: shape.clone(),
                orientation,
                overflow_visible: true,
                transform: Affine::translate(240.0, 140.0),
                paragraphs: vec![ParagraphRun { len: content.len(), style: ParagraphStyle { align, ..Default::default() } }],
                ..Default::default()
            };
            let expected = photocraft_text::shared().lock().unwrap().layout(&original, 72.0);
            photocraft_text::shared().lock().unwrap().render_layer(&mut original, 72.0, doc.pixel_format());
            doc.layers.push(Layer::new("translation", LayerContent::Text(original.clone())));
            let project = photocraft_io::export(&doc, "comic.pcraft", &Default::default()).unwrap();
            let loaded = photocraft_io::import("comic.pcraft", &project.bytes).unwrap().document;
            let LayerContent::Text(native) = &loaded.layers[0].content else { panic!("native text missing") };
            assert!(native.overflow_visible);
            assert_eq!(native.shape, shape);

            let bytes = photocraft_io::export(&doc, "comic.psd", &Default::default()).unwrap().bytes;
            assert_eq!(doc.layers[0].content, LayerContent::Text(original.clone()), "export must never edit the source project");
            let mut psd = photocraft_io::import("comic.psd", &bytes).unwrap().document;
            let LayerContent::Text(text) = &mut psd.layers[0].content else { panic!("editable PSD text missing") };
            assert!(!text.overflow_visible, "PSD uses standard bounded paragraph text");
            assert!(matches!(text.shape, TextShape::Box { x: 0.0, y: 0.0, .. }));
            assert_eq!(text.text.replace('\n', ""), content);
            assert_eq!(
                text.cache.as_ref().unwrap().read_region(original.cache.as_ref().unwrap().content_bounds()),
                original.cache.as_ref().unwrap().read_region(original.cache.as_ref().unwrap().content_bounds())
            );
            let actual = photocraft_text::shared().lock().unwrap().layout(text, 72.0);
            assert_eq!(actual.glyphs.len(), expected.glyphs.len());
            for (a, b) in expected.glyphs.iter().zip(&actual.glyphs) {
                assert_eq!(a.id, b.id);
                let before = original.transform.apply(Point::new(f64::from(a.x), f64::from(a.y)));
                let after = text.transform.apply(Point::new(f64::from(b.x), f64::from(b.y)));
                assert!((before.x - after.x).abs() < 0.1 && (before.y - after.y).abs() < 0.1, "{orientation:?} {align:?}: {before:?} -> {after:?}");
            }
            text.text.replace_range(0..1, "Z");
            text.psd_raw = None;
            text.cache = None;
            let edited = text.text.clone();
            photocraft_text::shared().lock().unwrap().render_layer(text, 72.0, doc.pixel_format());
            let saved = photocraft_io::export(&psd, "edited.psd", &Default::default()).unwrap();
            let reopened = photocraft_io::import("edited.psd", &saved.bytes).unwrap().document;
            let LayerContent::Text(text) = &reopened.layers[0].content else { panic!("reopened editable text missing") };
            assert_eq!(text.text, edited);
            assert!(!text.cache.as_ref().unwrap().content_bounds().is_empty());
        }
    }
}
