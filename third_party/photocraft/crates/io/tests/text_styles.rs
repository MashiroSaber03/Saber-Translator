//! Character and paragraph styles through PSD: named sheets in the type layers' engine data,
//! run references, and untouched files staying byte-exact.

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::text::{CharStyle, ParagraphRun, ParagraphStyle, TextAlign, TextRun};
use photocraft_doc::text_styles::{CharacterStyleDef, ParagraphStyleDef, diff_attrs};
use photocraft_doc::{Document, Layer, LayerContent, Size, TextLayer, TextStyles};

fn styles() -> TextStyles {
    let mut s = TextStyles::default();
    s.character.push(CharacterStyleDef {
        id: 3,
        name: "Loud".into(),
        attrs: diff_attrs(&CharStyle { size_pt: 30.0, underline: true, ..Default::default() }, &CharStyle::default()),
    });
    s.paragraph.push(ParagraphStyleDef {
        id: 1,
        name: "Middle".into(),
        para_attrs: diff_attrs(&ParagraphStyle { align: TextAlign::Center, ..Default::default() }, &ParagraphStyle::default()),
        ..Default::default()
    });
    s
}

fn doc_with(styles: TextStyles, link: bool) -> Document {
    let mut doc = Document::new("t", Size::new(200, 80), ColorMode::Rgb, SampleType::U8);
    let base = CharStyle { font_family: "Inter".into(), ..Default::default() };
    let loud = if link { styles.resolve_char(None, Some(3), "Inter") } else { base.clone() };
    let para = if link { styles.resolve_para(Some(1)) } else { ParagraphStyle::default() };
    let mut t = TextLayer {
        text: "calm LOUD".into(),
        runs: vec![TextRun { len: 5, style: base }, TextRun { len: 4, style: loud }],
        paragraphs: vec![ParagraphRun { len: 9, style: para }],
        transform: photocraft_geom::Affine::translate(4.0, 40.0),
        ..Default::default()
    };
    t.sync_summary();
    photocraft_text::TextEngine::new().render_layer(&mut t, doc.resolution_dpi, doc.pixel_format());
    doc.layers.push(Layer::new("t", LayerContent::Text(t)));
    doc.text_styles = styles;
    doc
}

fn text(doc: &Document) -> &TextLayer {
    match &doc.layers[0].content {
        LayerContent::Text(t) => t,
        _ => panic!("not text"),
    }
}

#[test]
fn named_styles_round_trip_through_psd() {
    let doc = doc_with(styles(), true);
    let out = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap();
    let back = photocraft_io::import("t.psd", &out.bytes).unwrap().document;
    let st = &back.text_styles;
    assert_eq!(st.character.len(), 1);
    assert_eq!(st.character[0].name, "Loud");
    assert_eq!(st.character[0].attrs.get("size_pt").and_then(|v| v.as_f64()), Some(30.0));
    assert_eq!(st.character[0].attrs.get("underline").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(st.paragraph[0].name, "Middle");
    assert_eq!(st.paragraph[0].para_attrs.get("align").and_then(|v| v.as_str()), Some("Center"));
    let t = text(&back);
    let runs = t.char_runs();
    assert_eq!(runs[0].style.style_sheet, None);
    assert_eq!(runs[1].style.style_sheet, Some(st.character[0].id));
    assert_eq!(t.paragraph_runs()[0].style.style_sheet, Some(st.paragraph[0].id));
    // Export again: stable.
    let out2 = photocraft_io::export(&back, "t.psd", &Default::default()).unwrap();
    let back2 = photocraft_io::import("t.psd", &out2.bytes).unwrap().document;
    assert_eq!(back2.text_styles.character.len(), 1);
    assert_eq!(back2.text_styles.paragraph.len(), 1);
}

#[test]
fn documents_without_styles_keep_their_type_data() {
    let doc = doc_with(TextStyles::default(), false);
    let out = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap();
    let back = photocraft_io::import("t.psd", &out.bytes).unwrap().document;
    assert!(back.text_styles.is_empty());
    // Re-export of an imported file without styles keeps TySh byte-exact.
    let raw = text(&back).psd_raw.clone().unwrap();
    let out2 = photocraft_io::export(&back, "t.psd", &Default::default()).unwrap();
    let back2 = photocraft_io::import("t.psd", &out2.bytes).unwrap().document;
    assert_eq!(text(&back2).psd_raw.as_deref(), Some(&*raw));
    // Deleting every style removes the named sheets on the next save.
    let styled = doc_with(styles(), true);
    let out = photocraft_io::export(&styled, "t.psd", &Default::default()).unwrap();
    let mut back = photocraft_io::import("t.psd", &out.bytes).unwrap().document;
    back.text_styles = TextStyles::default();
    let out = photocraft_io::export(&back, "t.psd", &Default::default()).unwrap();
    let back = photocraft_io::import("t.psd", &out.bytes).unwrap().document;
    assert!(back.text_styles.character.is_empty() && back.text_styles.paragraph.is_empty());
}
