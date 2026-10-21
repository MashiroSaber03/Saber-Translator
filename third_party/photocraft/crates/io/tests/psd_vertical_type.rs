//! Vertical type (#199): orientation survives a PSD round trip (descriptor `Ornt` and the
//! engine data's writing direction), point and paragraph text, and the re-rendered pixels after
//! import are the vertical layout's (taller than wide).

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::text::{CharStyle, Orientation, TextRun, TextShape};
use photocraft_doc::{Document, Layer, LayerContent, Size, TextLayer};
use photocraft_geom::Affine;

fn doc_with(shape: TextShape) -> Document {
    let mut doc = Document::new("v", Size::new(300, 400), ColorMode::Rgb, SampleType::U8);
    let text = "Vertical type\nColumns";
    let mut t = TextLayer {
        text: text.into(),
        runs: vec![TextRun { len: text.len(), style: CharStyle { font_family: "Inter".into(), size_pt: 24.0, ..Default::default() } }],
        orientation: Orientation::Vertical,
        shape,
        transform: Affine::translate(250.0, 20.0),
        ..Default::default()
    };
    t.sync_summary();
    photocraft_text::TextEngine::new().render_layer(&mut t, doc.resolution_dpi, doc.pixel_format());
    doc.layers.push(Layer::new("v", LayerContent::Text(t)));
    doc
}

fn text(doc: &Document) -> &TextLayer {
    match &doc.layers[0].content {
        LayerContent::Text(t) => t,
        _ => panic!("not text"),
    }
}

#[test]
fn vertical_orientation_round_trips_through_psd() {
    for shape in [TextShape::Point, TextShape::Box { x: -200.0, y: 0.0, width: 220.0, height: 300.0 }] {
        let doc = doc_with(shape.clone());
        let ink = text(&doc).cache.as_ref().unwrap().content_bounds();
        assert!(ink.height() > ink.width(), "{shape:?}: {ink:?}");
        let out = photocraft_io::export(&doc, "v.psd", &Default::default()).unwrap();
        let back = photocraft_io::import("v.psd", &out.bytes).unwrap().document;
        let t = text(&back);
        assert_eq!(t.orientation, Orientation::Vertical, "{shape:?}");
        assert_eq!(t.shape, shape);
        assert_eq!(t.text, text(&doc).text);
        // Re-rendering the imported model gives the same vertical layout.
        let mut again = t.clone();
        photocraft_text::TextEngine::new().render_layer(&mut again, back.resolution_dpi, back.pixel_format());
        let r = again.cache.as_ref().unwrap().content_bounds();
        assert!(
            (r.x0 - ink.x0).abs() <= 1 && (r.y0 - ink.y0).abs() <= 1 && (r.x1 - ink.x1).abs() <= 1 && (r.y1 - ink.y1).abs() <= 1,
            "{shape:?}: {ink:?} → {r:?}"
        );
        // A second trip keeps it, and horizontal stays horizontal.
        let out2 = photocraft_io::export(&back, "v.psd", &Default::default()).unwrap();
        assert_eq!(text(&photocraft_io::import("v.psd", &out2.bytes).unwrap().document).orientation, Orientation::Vertical);
    }
    let mut doc = doc_with(TextShape::Point);
    if let LayerContent::Text(t) = &mut doc.layers[0].content {
        t.orientation = Orientation::Horizontal;
        t.psd_raw = None;
    }
    let out = photocraft_io::export(&doc, "h.psd", &Default::default()).unwrap();
    assert_eq!(text(&photocraft_io::import("h.psd", &out.bytes).unwrap().document).orientation, Orientation::Horizontal);
}
