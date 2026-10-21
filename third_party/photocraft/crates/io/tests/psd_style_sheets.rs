//! #3795: a PSD type layer's alignment and character attributes can come from its style sheets
//! rather than its runs. A run's `Properties` / `StyleSheetData` hold only its overrides; the rest
//! comes from the paragraph style its `Name` names in `ParagraphSheetSet`, the character style in
//! `StyleSheetSet` (by name, or the paragraph's `DefaultStyleSheet`), then the Normal sheets.
//! Reading only the Normal sheets left a centred paragraph style read as Left, so starting an
//! edit re-laid the text out half a line to the side.

use photocraft_doc::text::TextAlign;
use photocraft_geom::Affine;
use photocraft_psd::descriptor::{Descriptor, Id, UnicodeString, Value as D};
use photocraft_text::engine_data::Value as E;
use photocraft_text::psd::{TySh, text_layer_from_tysh, write_tysh};

fn real(v: f64) -> E {
    E::Real(v)
}

fn d(kv: Vec<(&str, E)>) -> E {
    E::Dict(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Photoshop-shaped EngineData for one paragraph of point text. `para` is the paragraph run's
/// `ParagraphSheet`, `style` the character run's `StyleSheet`; the sheet sets are the
/// `ResourceDict`'s, with Normal at index 0.
fn engine(text: &str, para: E, style: E, para_set: Vec<E>, style_set: Vec<E>) -> E {
    let ed_text = format!("{text}\r");
    let n = ed_text.encode_utf16().count() as i64;
    let res = d(vec![
        (
            "FontSet",
            E::Array(vec![d(vec![("Name", E::String("Inter-Regular".into())), ("Script", E::Int(0)), ("FontType", E::Int(1)), ("Synthetic", E::Int(0))])]),
        ),
        ("StyleSheetSet", E::Array(style_set)),
        ("ParagraphSheetSet", E::Array(para_set)),
        ("TheNormalStyleSheet", E::Int(0)),
        ("TheNormalParagraphSheet", E::Int(0)),
    ]);
    d(vec![
        (
            "EngineDict",
            d(vec![
                ("Editor", d(vec![("Text", E::String(ed_text))])),
                ("ParagraphRun", d(vec![("RunArray", E::Array(vec![d(vec![("ParagraphSheet", para)])])), ("RunLengthArray", E::Array(vec![E::Int(n)]))])),
                ("StyleRun", d(vec![("RunArray", E::Array(vec![d(vec![("StyleSheet", style)])])), ("RunLengthArray", E::Array(vec![E::Int(n)]))])),
                (
                    "Rendered",
                    d(vec![(
                        "Shapes",
                        d(vec![(
                            "Children",
                            E::Array(vec![d(vec![
                                ("ShapeType", E::Int(0)),
                                ("Cookie", d(vec![("Photoshop", d(vec![("ShapeType", E::Int(0)), ("PointBase", E::Array(vec![real(0.0), real(0.0)]))]))])),
                            ])]),
                        )]),
                    )]),
                ),
            ]),
        ),
        ("ResourceDict", res),
    ])
}

fn normal_char(size: f64) -> E {
    d(vec![
        ("Name", E::String("Normal RGB".into())),
        ("StyleSheetData", d(vec![("Font", E::Int(0)), ("FontSize", real(size)), ("AutoLeading", E::Bool(true))])),
    ])
}

fn para_sheet(name: &str, justification: i64, default_style: i64) -> E {
    d(vec![("Name", E::String(name.into())), ("DefaultStyleSheet", E::Int(default_style)), ("Properties", d(vec![("Justification", E::Int(justification))]))])
}

fn layer(text: &str, ed: &E) -> photocraft_doc::TextLayer {
    let desc = Descriptor::new("TxLr")
        .with("Txt ", D::Text(UnicodeString::new_nul(text)))
        .with("Ornt", D::Enumerated { type_id: Id::new("Ornt"), value: Id::new("Hrzn") })
        .with("TextIndex", D::Integer(0))
        .with("EngineData", D::RawData(photocraft_text::engine_data::write(ed)));
    let data = write_tysh(&TySh { transform: Affine::translate(450.0, 300.0), text: desc, warp: None, bounds: [0.0; 4] });
    let mut t = text_layer_from_tysh(&data, 72.0).expect("parses");
    // Import keeps the block, as for a layer read from a file.
    t.psd_raw = Some(std::sync::Arc::new(data));
    t
}

/// The paragraph run names a centred (or right-aligned) paragraph style and overrides nothing.
#[test]
fn alignment_from_the_paragraph_style_the_run_names() {
    for (j, want) in [(2, TextAlign::Center), (1, TextAlign::Right)] {
        let ed = engine(
            "LOREM IPSUM",
            d(vec![("Name", E::String("Title".into())), ("DefaultStyleSheet", E::Int(0)), ("Properties", d(vec![]))]),
            d(vec![("StyleSheetData", d(vec![]))]),
            vec![para_sheet("Normal RGB", 0, 0), para_sheet("Title", j, 0)],
            vec![normal_char(48.0)],
        );
        let t = layer("LOREM IPSUM", &ed);
        assert_eq!(t.paragraphs[0].style.align, want, "Justification {j} from the paragraph style");
        // Laid out around the anchor like Photoshop draws it.
        let b = photocraft_text::TextEngine::new().layout(&t, 72.0).bounds().unwrap();
        match want {
            TextAlign::Center => assert!((b[0] + b[2]).abs() < 1.0, "centred on the anchor: {b:?}"),
            _ => assert!(b[2].abs() < 1.0, "ends at the anchor: {b:?}"),
        }
    }
}

/// A run's own value still wins over its paragraph style, and an unknown style name or an
/// out-of-range sheet index falls back to Normal.
#[test]
fn run_overrides_win_and_bad_references_fall_back_to_normal() {
    let ed = engine(
        "Text",
        d(vec![("Name", E::String("Title".into())), ("DefaultStyleSheet", E::Int(0)), ("Properties", d(vec![("Justification", E::Int(1))]))]),
        d(vec![("StyleSheetData", d(vec![]))]),
        vec![para_sheet("Normal RGB", 0, 0), para_sheet("Title", 2, 0)],
        vec![normal_char(48.0)],
    );
    assert_eq!(layer("Text", &ed).paragraphs[0].style.align, TextAlign::Right);
    let ed = engine(
        "Text",
        d(vec![("Name", E::String("Missing".into())), ("DefaultStyleSheet", E::Int(9)), ("Properties", d(vec![]))]),
        d(vec![("Name", E::String("Missing".into())), ("StyleSheetData", d(vec![]))]),
        vec![para_sheet("Normal RGB", 2, 0)],
        vec![normal_char(30.0)],
    );
    let t = layer("Text", &ed);
    assert_eq!(t.paragraphs[0].style.align, TextAlign::Center, "Normal's alignment");
    assert!((t.runs[0].style.size_pt - 30.0).abs() < 0.01, "Normal's size: {}", t.runs[0].style.size_pt);
}

/// Character attributes from the paragraph's default character style (`DefaultStyleSheet`) and
/// from a character style the run names.
#[test]
fn character_attributes_from_the_style_sheets() {
    let big = d(vec![("Name", E::String("Big".into())), ("StyleSheetData", d(vec![("FontSize", real(72.0))]))]);
    let ed = engine(
        "Text",
        d(vec![("DefaultStyleSheet", E::Int(1)), ("Properties", d(vec![]))]),
        d(vec![("StyleSheetData", d(vec![]))]),
        vec![para_sheet("Normal RGB", 0, 0)],
        vec![normal_char(12.0), big.clone()],
    );
    assert!((layer("Text", &ed).runs[0].style.size_pt - 72.0).abs() < 0.01, "the paragraph's default character style");
    let ed = engine(
        "Text",
        d(vec![("DefaultStyleSheet", E::Int(0)), ("Properties", d(vec![]))]),
        d(vec![("Name", E::String("Big".into())), ("StyleSheetData", d(vec![]))]),
        vec![para_sheet("Normal RGB", 0, 0)],
        vec![normal_char(12.0), big],
    );
    let t = layer("Text", &ed);
    assert!((t.runs[0].style.size_pt - 72.0).abs() < 0.01, "the named character style: {}", t.runs[0].style.size_pt);
    // Other attributes still come from Normal.
    assert_eq!(
        t.runs[0].style.font_family,
        layer(
            "Text",
            &engine(
                "Text",
                d(vec![("Properties", d(vec![]))]),
                d(vec![("StyleSheetData", d(vec![]))]),
                vec![para_sheet("Normal RGB", 0, 0)],
                vec![normal_char(12.0)]
            )
        )
        .runs[0]
            .style
            .font_family
    );
}

/// Export writes every run fully resolved and keeps the sheet sets: a round trip reads the same
/// alignment and size, with one entry per sheet.
#[test]
fn styles_from_sheets_survive_a_round_trip() {
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_doc::{Document, Layer, LayerContent, Size};
    let ed = engine(
        "LOREM IPSUM",
        d(vec![("Name", E::String("Title".into())), ("DefaultStyleSheet", E::Int(1)), ("Properties", d(vec![]))]),
        d(vec![("StyleSheetData", d(vec![]))]),
        vec![para_sheet("Normal RGB", 0, 0), para_sheet("Title", 2, 1)],
        vec![normal_char(12.0), d(vec![("Name", E::String("Big".into())), ("StyleSheetData", d(vec![("FontSize", real(48.0))]))])],
    );
    let mut t = layer("LOREM IPSUM", &ed);
    let mut doc = Document::new("t", Size::new(900, 600), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = 72.0;
    photocraft_text::TextEngine::new().render_layer(&mut t, 72.0, doc.pixel_format());
    doc.layers.push(Layer::new("t", LayerContent::Text(t)));
    let bytes = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("t.psd", &bytes).unwrap().document;
    let LayerContent::Text(bt) = &back.layers[0].content else { panic!("not a type layer") };
    assert_eq!(bt.paragraphs[0].style.align, TextAlign::Center);
    assert!((bt.runs[0].style.size_pt - 48.0).abs() < 0.01, "{}", bt.runs[0].style.size_pt);
}

/// A layer that inherits its alignment and size from sheets it no longer links to (a document
/// without those styles) keeps them through export: runs are written with what they inherited.
#[test]
fn unlinked_runs_keep_their_inherited_style_on_export() {
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_doc::{Document, Layer, LayerContent, Size};
    let ed = engine(
        "LOREM IPSUM",
        d(vec![("Name", E::String("Title".into())), ("DefaultStyleSheet", E::Int(1)), ("Properties", d(vec![]))]),
        d(vec![("StyleSheetData", d(vec![]))]),
        vec![para_sheet("Normal RGB", 0, 0), para_sheet("Title", 2, 1)],
        vec![normal_char(12.0), d(vec![("Name", E::String("Big".into())), ("StyleSheetData", d(vec![("FontSize", real(48.0))]))])],
    );
    let mut t = layer("LOREM IPSUM", &ed);
    assert!(t.psd_raw.is_some());
    let mut doc = Document::new("t", Size::new(900, 600), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = 72.0;
    photocraft_text::TextEngine::new().render_layer(&mut t, 72.0, doc.pixel_format());
    doc.layers.push(Layer::new("t", LayerContent::Text(t)));
    // No `text_styles_map::import`: the document has no styles and the runs link to none.
    let bytes = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("t.psd", &bytes).unwrap().document;
    let LayerContent::Text(bt) = &back.layers[0].content else { panic!("not a type layer") };
    assert_eq!(bt.paragraphs[0].style.align, TextAlign::Center);
    assert!((bt.runs[0].style.size_pt - 48.0).abs() < 0.01, "{}", bt.runs[0].style.size_pt);
}
