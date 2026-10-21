//! Visible comic text stays complete and editable in standard Photoshop paragraph layers.
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{
    Document, Layer, LayerContent, TextLayer,
    text::{Orientation, ParagraphRun, ParagraphStyle, TextAlign, TextShape},
};
use photocraft_geom::{Affine, Size};

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
                let world = |m: [f64; 6], x: f32, y: f32| (m[0] * f64::from(x) + m[2] * f64::from(y) + m[4], m[1] * f64::from(x) + m[3] * f64::from(y) + m[5]);
                for (before, after) in expected.glyphs.iter().zip(&actual.glyphs) {
                    let (a, b) = (world(original.transform.m, before.x, before.y), world(psd.transform.m, after.x, after.y));
                    assert!((a.0 - b.0).abs() < 0.2 && (a.1 - b.1).abs() < 0.2, "{orientation:?}/{align:?}/{extent}: {a:?} -> {b:?}");
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
            assert_eq!(text.text.replace('\n', ""), content);
            assert_eq!(
                text.cache.as_ref().unwrap().read_region(original.cache.as_ref().unwrap().content_bounds()),
                original.cache.as_ref().unwrap().read_region(original.cache.as_ref().unwrap().content_bounds())
            );
            let actual = photocraft_text::shared().lock().unwrap().layout(text, 72.0);
            assert_eq!(actual.glyphs.len(), expected.glyphs.len());
            for (a, b) in expected.glyphs.iter().zip(&actual.glyphs) {
                assert_eq!(a.id, b.id);
                assert!((a.x - b.x).abs() < 0.1 && (a.y - b.y).abs() < 0.1, "{orientation:?} {align:?}: {a:?} -> {b:?}");
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
