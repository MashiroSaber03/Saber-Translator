//! Notes (`Anno`) and the measurement scale (resource 1074) through PSD: verbatim while
//! unchanged, regenerated after edits.

#[cfg(feature = "corpus")]
use std::path::PathBuf;

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, MeasurementScale, Note};
use photocraft_geom::Size;
use photocraft_io::annotations_map::MEASUREMENT_SCALE;
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

#[cfg(feature = "corpus")]
#[test]
fn corpus_notes_import_and_verbatim_export() {
    let bytes = corpus("ag-psd/read-write/annotations/src.psd");
    let src = PsdFile::from_bytes(&bytes).unwrap();
    let mut doc = import("src.psd", &bytes).unwrap().document;
    assert_eq!(doc.notes.len(), 2);
    assert_eq!(doc.notes[1].text, "open note");
    let out = to_psd(&doc);
    let f = PsdFile::from_bytes(&out).unwrap();
    assert_eq!(f.global_block(b"Anno").unwrap().data, src.global_block(b"Anno").unwrap().data);

    // Edit a note: the block is regenerated and reads back with the change.
    doc.notes[0].text = "changed".into();
    doc.notes.push(Note { author: "agent".into(), text: "third".into(), position: [5.0, 6.0], ..Default::default() });
    let back = import("x.psd", &to_psd(&doc)).unwrap().document;
    assert_eq!(back.notes.len(), 3);
    assert_eq!(back.notes[0].text, "changed");
    assert_eq!(back.notes[2].position, [5.0, 6.0]);
    // Deleting every note drops the block.
    doc.notes.clear();
    let f = PsdFile::from_bytes(&to_psd(&doc)).unwrap();
    assert!(f.global_block(b"Anno").is_none());
}

#[test]
fn measurement_scale_roundtrips_through_psd() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut doc = Document::with_background("m", Size::new(16, 8), ColorMode::Rgb, depth, photocraft_color::Color::rgb(1.0, 1.0, 1.0));
        let f = PsdFile::from_bytes(&to_psd(&doc)).unwrap();
        assert!(f.resource(MEASUREMENT_SCALE).is_none());
        doc.measurement.scale = MeasurementScale { pixel_length: 120.0, logical_length: 3.0, units: "cm".into() };
        doc.notes.push(Note { text: "n".into(), ..Default::default() });
        let back = import("x.psd", &to_psd(&doc)).unwrap().document;
        assert_eq!(back.measurement.scale, doc.measurement.scale);
        assert_eq!(back.notes.len(), 1);
        assert_eq!((back.notes[0].text.as_str(), back.notes[0].position), ("n", [0.0, 0.0]));
        // Unchanged → the preserved resource is written byte-exact (once).
        let again = PsdFile::from_bytes(&to_psd(&back)).unwrap();
        assert_eq!(again.resources.iter().filter(|r| r.id == MEASUREMENT_SCALE).count(), 1);
    }
}

/// A PSD `Anno` block with one text note whose icon rectangle is `[top, left, 0, 0]` and whose
/// popup is all zeros (issue #939's fixture).
fn anno_with_icon(top: i32, left: i32) -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(b"txtA");
    entry.push(1); // open
    entry.push(0x1C); // flags
    entry.extend_from_slice(&1u16.to_be_bytes()); // optional blocks
    for v in [top, left, 0, 0] {
        entry.extend_from_slice(&v.to_be_bytes()); // icon: top, left, bottom, right
    }
    for v in [0i32; 4] {
        entry.extend_from_slice(&v.to_be_bytes()); // popup
    }
    entry.extend_from_slice(&0u16.to_be_bytes()); // colour space: RGB
    for _ in 0..4 {
        entry.extend_from_slice(&0u16.to_be_bytes());
    }
    entry.extend_from_slice(&[0u8, 0]); // author: empty Pascal string
    entry.extend_from_slice(&[0u8, 0]); // name
    entry.extend_from_slice(&[0u8, 0]); // modification date
    let text = [0xFEu8, 0xFF, 0x00, b'A']; // UTF-16BE "A"
    entry.extend_from_slice(&((text.len() + 12) as u32).to_be_bytes());
    entry.extend_from_slice(b"txtC");
    entry.extend_from_slice(&(text.len() as u32).to_be_bytes());
    entry.extend_from_slice(&text);
    let mut out = Vec::new();
    out.extend_from_slice(&2u16.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&1u32.to_be_bytes()); // one annotation
    out.extend_from_slice(&((entry.len() + 4) as u32).to_be_bytes());
    out.extend_from_slice(&entry);
    out
}

fn psd_with_anno(anno: Vec<u8>) -> Vec<u8> {
    let mut doc = Document::new("notes", Size::new(16, 16), ColorMode::Rgb, SampleType::U8);
    doc.layers.push(photocraft_doc::Layer::raster("Background", doc.pixel_format()));
    let mut file = PsdFile::from_bytes(&to_psd(&doc)).unwrap();
    let mut tb = photocraft_psd::TaggedBlock::new(*b"Anno", anno.clone());
    tb.padding = Some(vec![0; (4 - anno.len() % 4) % 4]);
    file.global_blocks.push(tb);
    file.to_bytes().unwrap()
}

/// Icon rectangle `[top, left, bottom, right]` of the first note in the exported `Anno` block.
fn exported_icon(doc: &Document) -> [i32; 4] {
    let f = PsdFile::from_bytes(&to_psd(doc)).unwrap();
    let d = &f.global_block(b"Anno").unwrap().data;
    // header (2 + 2 + 4), entry length (4), `txtA` (4), open (1), flags (1), optional (2).
    let mut out = [0i32; 4];
    for (i, v) in out.iter_mut().enumerate() {
        let at = 20 + i * 4;
        *v = i32::from_be_bytes(d[at..at + 4].try_into().unwrap());
    }
    out
}

/// Issue #939: a note whose stored icon top sits near `i32::MAX` imported fine and then
/// overflowed `y + 20` on every PSD save (a panic in debug, a wrapped coordinate in release). The
/// far edge saturates instead, so the note survives a round trip with representable geometry.
#[test]
fn a_note_icon_near_i32_max_saves_with_representable_geometry() {
    for (top, left) in [(i32::MAX - 5, 0), (0, i32::MAX - 3), (i32::MIN, i32::MIN)] {
        let imp = import("notes.psd", &psd_with_anno(anno_with_icon(top, left))).expect("import");
        assert_eq!(imp.document.notes[0].position, [f64::from(left), f64::from(top)]);
        let icon = exported_icon(&imp.document);
        assert_eq!(icon, [top, left, top.saturating_add(20), left.saturating_add(16)], "{top},{left}");
        let back = import("notes.psd", &to_psd(&imp.document)).unwrap().document;
        assert_eq!((back.notes[0].text.as_str(), back.notes[0].position), ("A", [f64::from(left), f64::from(top)]));
    }
    // The same through the model: a position a command could set beyond i32 saturates, NaN is 0.
    let mut doc = Document::new("notes", Size::new(16, 16), ColorMode::Rgb, SampleType::U8);
    doc.notes.push(Note { text: "far".into(), position: [1e300, f64::INFINITY], ..Default::default() });
    assert_eq!(exported_icon(&doc), [i32::MAX, i32::MAX, i32::MAX, i32::MAX]);
    doc.notes[0].position = [f64::NAN, -1e300];
    assert_eq!(exported_icon(&doc), [i32::MIN, 0, i32::MIN + 20, 16]);
    // Control: an ordinary note is unchanged.
    doc.notes[0].position = [42.0, 28.0];
    assert_eq!(exported_icon(&doc), [28, 42, 48, 58]);
    let back = import("x.psd", &to_psd(&doc)).unwrap().document;
    assert_eq!(back.notes[0].position, [42.0, 28.0]);
}

/// Issue #944: an `Anno` block the reader rejects opened as "no notes" without a word, and the
/// next save deleted it. It now opens with a warning and is written back byte for byte.
#[test]
fn an_unreadable_anno_block_warns_and_survives_a_save() {
    // The block declares two annotations but holds one.
    let mut broken = anno_with_icon(28, 42);
    broken[4..8].copy_from_slice(&2u32.to_be_bytes());
    let imp = import("notes.psd", &psd_with_anno(broken.clone())).expect("import");
    assert!(imp.document.notes.is_empty());
    assert!(imp.warnings.iter().any(|w| w.contains("Anno")), "{:?}", imp.warnings);
    let f = PsdFile::from_bytes(&to_psd(&imp.document)).unwrap();
    assert_eq!(f.global_block(b"Anno").expect("Anno block kept").data, broken);
    // Control: the intact block reads with no warning.
    let ok = import("notes.psd", &psd_with_anno(anno_with_icon(28, 42))).expect("import");
    assert_eq!(ok.document.notes.len(), 1);
    assert!(!ok.warnings.iter().any(|w| w.contains("Anno")));
}
