//! #3833: hyphens inside an installed family must survive PSD save/reopen and editing.
//! The fixtures rename the OFL bundled fonts in memory; no proprietary fonts are needed.

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{
    Document, Layer, LayerContent, TextLayer,
    text::{CharStyle, TextRun},
};
use photocraft_geom::Size;
use photocraft_text::fonts::{INTER_REGULAR, JETBRAINS_MONO_REGULAR};

fn renamed_font(source: &[u8], family: &str, ps: &str) -> Vec<u8> {
    font_with_names(source, &[(1, family), (2, "Regular"), (6, ps), (16, family), (17, "Regular")])
}

fn font_with_names(source: &[u8], names: &[(u16, &str)]) -> Vec<u8> {
    let mut bytes = source.to_vec();
    let count = u16::from_be_bytes(bytes[4..6].try_into().unwrap()) as usize;
    let record = (0..count).map(|i| 12 + i * 16).find(|r| &bytes[*r..*r + 4] == b"name").unwrap();
    let mut table = Vec::new();
    let mut strings = Vec::new();
    for value in [0u16, names.len() as u16, (6 + names.len() * 12) as u16] {
        table.extend(value.to_be_bytes());
    }
    for &(id, value) in names {
        let encoded: Vec<_> = value.encode_utf16().flat_map(u16::to_be_bytes).collect();
        for n in [3, 1, 0x409, id, encoded.len() as u16, strings.len() as u16] {
            table.extend(n.to_be_bytes());
        }
        strings.extend(encoded);
    }
    table.extend(strings);
    let offset = bytes.len() as u32;
    bytes[record + 8..record + 12].copy_from_slice(&offset.to_be_bytes());
    bytes[record + 12..record + 16].copy_from_slice(&(table.len() as u32).to_be_bytes());
    bytes.extend(table);
    bytes
}

fn roundtrip(family: &str, ps: Option<&str>, install_prefix: bool) {
    let actual_ps = ps.map(str::to_string).unwrap_or_else(|| format!("{family}-Book").replace(' ', ""));
    {
        let mut engine = photocraft_text::shared().lock().unwrap();
        engine.fonts.register_font_data(renamed_font(&JETBRAINS_MONO_REGULAR, family, &actual_ps));
        if install_prefix {
            let prefix = family.split_once('-').unwrap().0;
            engine.fonts.register_font_data(renamed_font(&INTER_REGULAR, prefix, &format!("{prefix}-Regular")));
        }
    }
    roundtrip_registered_fonts(family, ps);
}

fn roundtrip_registered_fonts(family: &str, ps: Option<&str>) {
    let mut doc = Document::new("fonts", Size::new(256, 64), ColorMode::Rgb, SampleType::U8);
    let mut original = TextLayer {
        text: "Wide WWW iii".into(),
        runs: vec![TextRun {
            len: 12,
            style: CharStyle { font_family: family.into(), postscript_name: ps.map(str::to_string), size_pt: 24.0, ..Default::default() },
        }],
        ..Default::default()
    };
    original.sync_summary();
    photocraft_text::shared().lock().unwrap().render_layer(&mut original, doc.resolution_dpi, doc.pixel_format());
    doc.layers.push(Layer::new("type", LayerContent::Text(original.clone())));
    let bytes = photocraft_io::export(&doc, "fonts.psd", &Default::default()).unwrap().bytes;
    let mut reopened = photocraft_io::import("fonts.psd", &bytes).unwrap().document;
    let LayerContent::Text(text) = &mut reopened.layers[0].content else { panic!("expected editable text") };
    assert!(text.cache.is_some(), "opening retains the original appearance");
    let before = original.cache.as_ref().unwrap();
    let opened = text.cache.as_ref().unwrap();
    assert_eq!(opened.read_region(before.content_bounds()), before.read_region(before.content_bounds()), "PSD pixels retain the original font before editing");
    assert_eq!(text.font_family, family, "the Character panel must show the installed family");
    assert_eq!(text.char_runs()[0].style.font_family, family);
    assert_eq!(text.char_runs()[0].style.weight, original.char_runs()[0].style.weight);
    assert_eq!(text.char_runs()[0].style.italic, original.char_runs()[0].style.italic);
    if let Some(ps) = ps {
        assert_eq!(text.char_runs()[0].style.postscript_name.as_deref(), Some(ps));
    }
    // Editing drops the PSD pixels and regenerates them from the reopened model.
    text.text.push('!');
    text.runs[0].len += 1;
    text.cache = None;
    text.psd_raw = None;
    original.text.push('!');
    original.runs[0].len += 1;
    let mut engine = photocraft_text::shared().lock().unwrap();
    let expected = engine.layout(&original, doc.resolution_dpi);
    let actual = engine.layout(text, doc.resolution_dpi);
    assert!((expected.lines[0].x1 - actual.lines[0].x1).abs() < 0.01, "editing must keep the same font outlines");
    engine.render_layer(text, doc.resolution_dpi, doc.pixel_format());
    engine.render_layer(&mut original, doc.resolution_dpi, doc.pixel_format());
    let edited = text.cache.as_ref().unwrap();
    let expected_pixels = original.cache.as_ref().unwrap();
    assert_eq!(edited.content_bounds(), expected_pixels.content_bounds());
    assert_eq!(edited.read_region(expected_pixels.content_bounds()), expected_pixels.read_region(expected_pixels.content_bounds()));
}

#[test]
fn family_only_hyphenated_font_survives_psd_reopen_and_edit() {
    roundtrip("FOT-NewRodin Pro", None, false);
}

#[test]
fn exact_postscript_identity_wins_over_installed_prefix_family() {
    roundtrip("Prefix-Hyphenated Pro", Some("Prefix-Book"), true);
}

#[test]
fn installed_family_survives_a_face_indexed_under_its_typographic_family() {
    const FAMILY: &str = "Indexed Alias";
    const PS: &str = "IndexedAlias-Regular";
    {
        let mut engine = photocraft_text::shared().lock().unwrap();
        // The requested family exists, but its exact PostScript face is listed under the
        // typographic family, as with system fonts that separate legacy and typographic names.
        engine.fonts.register_font_data(renamed_font(&INTER_REGULAR, FAMILY, "IndexedAlias-Other"));
        engine.fonts.register_font_data(font_with_names(&JETBRAINS_MONO_REGULAR, &[(1, FAMILY), (2, "Regular"), (6, PS), (16, "Indexed"), (17, "Regular")]));
        assert!(engine.fonts.has_family(FAMILY));
        assert!(engine.fonts.faces(FAMILY).iter().all(|face| face.postscript_name.as_deref() != Some(PS)));
        let exact = engine.fonts.resolve_postscript(PS);
        assert!(exact.exact);
        assert_eq!(exact.family, "Indexed");
    }
    roundtrip_registered_fonts(FAMILY, Some(PS));
}
