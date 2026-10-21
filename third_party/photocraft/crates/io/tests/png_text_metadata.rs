//! PNG text survives open, native save/reopen and PNG save without a false loss warning.

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType, decode, encode};
use photocraft_io::{ExportOptions, XmpEmbed, export, import};

fn source(sample: SampleType) -> Image {
    let mut image = Image::from_u8(1, 1, ChannelLayout::Rgb, vec![31, 127, 223]).unwrap().convert(ChannelLayout::Rgb, sample);
    image.meta.text = vec![
        ("Creation Time".into(), "Fri, 09 Oct 2026 07:00:00 GMT".into()),
        ("Author".into(), "Synthetic author".into()),
        ("Description".into(), "Synthetic description é".into()),
        ("Title".into(), "Unicode 雪".into()),
        ("Comment".into(), "first".into()),
        ("Comment".into(), "second".into()),
    ];
    image.meta.xmp = Some("<x:xmpmeta xmlns:x='adobe:ns:meta/'/>".into());
    image
}

#[test]
fn png_text_survives_native_and_png_saves_at_both_depths() {
    for sample in [SampleType::U8, SampleType::U16] {
        let image = source(sample);
        let png = encode(&image, Format::Png, &EncodeOptions::default()).unwrap();
        let original = decode(&png).unwrap();
        let opened = import("synthetic.png", &png).unwrap();
        assert!(opened.warnings.is_empty(), "{:?}", opened.warnings);
        let native = export(&opened.document, "pcraft", &ExportOptions::default()).unwrap();
        let reopened = import("synthetic.pcraft", &native.bytes).unwrap();
        let saved = export(&reopened.document, "png", &ExportOptions::default()).unwrap();
        assert!(saved.warnings.is_empty(), "{:?}", saved.warnings);
        let back = decode(&saved.bytes).unwrap();
        assert_eq!(back.meta.text, original.meta.text, "duplicates and Unicode must survive");
        assert_eq!(back.meta.xmp, original.meta.xmp);
        assert_eq!(back.sample_type(), sample);
        let back = back.convert(original.layout(), sample);
        assert_eq!(back.data(), original.data(), "pixels must not change");
    }
}

#[test]
fn metadata_none_and_embed_flag_remove_png_text_without_mutating_document() {
    let image = source(SampleType::U8);
    let png = encode(&image, Format::Png, &EncodeOptions::default()).unwrap();
    let doc = import("synthetic.png", &png).unwrap().document;
    for opts in [
        ExportOptions { xmp: XmpEmbed::None, ..Default::default() },
        ExportOptions { encode: EncodeOptions { embed_metadata: false, ..Default::default() }, ..Default::default() },
    ] {
        let saved = export(&doc, "png", &opts).unwrap();
        let back = decode(&saved.bytes).unwrap();
        assert!(back.meta.text.is_empty());
        assert!(back.meta.xmp.is_none());
    }
    let saved = export(&doc, "png", &ExportOptions::default()).unwrap();
    assert_eq!(decode(&saved.bytes).unwrap().meta.text, decode(&png).unwrap().meta.text);
}

#[test]
fn formats_without_text_support_report_the_loss_on_export() {
    let image = source(SampleType::U8);
    let png = encode(&image, Format::Png, &EncodeOptions::default()).unwrap();
    let doc = import("synthetic.png", &png).unwrap().document;
    for ext in ["jpg", "psd", "psb"] {
        let saved = export(&doc, ext, &ExportOptions::default()).unwrap();
        assert!(saved.warnings.iter().any(|w| w.contains("text metadata")), "{ext}: {:?}", saved.warnings);
    }
}

#[test]
fn palette_png_reports_unsupported_text_only_when_embedding_is_requested() {
    let image = source(SampleType::U8);
    let png = encode(&image, Format::Png, &EncodeOptions::default()).unwrap();
    let mut doc = import("synthetic.png", &png).unwrap().document;
    doc.mode = photocraft_color::ColorMode::Indexed;
    doc.color_table = Some(photocraft_doc::ColorTable { colors: vec![[31, 127, 223]], transparent: None });
    let saved = export(&doc, "png", &ExportOptions::default()).unwrap();
    assert!(saved.warnings.iter().any(|w| w.contains("text metadata")));
    let opts = ExportOptions { xmp: XmpEmbed::None, ..Default::default() };
    let saved = export(&doc, "png", &opts).unwrap();
    assert!(!saved.warnings.iter().any(|w| w.contains("text metadata")));
}

#[test]
fn tiff_text_survives_flat_and_layered_saves_and_respects_metadata_none() {
    let image = source(SampleType::U16);
    let png = encode(&image, Format::Png, &EncodeOptions::default()).unwrap();
    let mut doc = import("synthetic.png", &png).unwrap().document;
    // TIFF's supported ASCII tag names, rather than arbitrary PNG keywords.
    doc.metadata.text = vec![("Description".into(), "Synthetic description".into()), ("Artist".into(), "Synthetic author".into())];
    let mut overlay = doc.layers[0].clone();
    overlay.id = photocraft_doc::LayerId::fresh();
    doc.layers.push(overlay);
    for tiff_layers in [false, true] {
        let opts = ExportOptions { tiff_layers, ..Default::default() };
        let saved = export(&doc, "tif", &opts).unwrap();
        let opened = import("synthetic.tif", &saved.bytes).unwrap();
        assert_eq!(opened.document.metadata.text, doc.metadata.text);
        assert!(!opened.warnings.iter().any(|w| w.contains("text metadata")), "{:?}", opened.warnings);
        let opts = ExportOptions { xmp: XmpEmbed::None, ..opts };
        let stripped = export(&doc, "tif", &opts).unwrap();
        assert!(decode(&stripped.bytes).unwrap().meta.text.is_empty());
        assert_eq!(doc.metadata.text.len(), 2, "export must not mutate the document");
    }
}
