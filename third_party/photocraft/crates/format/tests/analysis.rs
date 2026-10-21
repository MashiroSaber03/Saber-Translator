//! Image › Analysis data and notes survive a `.pcraft` round trip; old manifests default them.

use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{CountGroup, Document, MeasurementScale, Note, Ruler, Size};
use photocraft_format::*;

#[test]
fn measurement_and_notes_roundtrip_and_default_when_absent() {
    let mut doc = Document::new("a", Size::new(40, 30), ColorMode::Rgb, SampleType::U16);
    doc.measurement.scale = MeasurementScale { pixel_length: 72.0, logical_length: 1.0, units: "inches".into() };
    doc.measurement.count_groups = vec![CountGroup {
        name: "Cells".into(),
        color: Color::rgb(0.0, 0.5, 1.0),
        marker_size: 3,
        label_size: 12,
        visible: true,
        points: vec![[1.0, 2.0], [5.5, 6.5]],
    }];
    doc.measurement.ruler = Some(Ruler { start: [1.0, 1.0], end: [30.0, 20.0], protractor: Some([1.0, 25.0]) });
    doc.notes = vec![Note {
        author: "A".into(),
        text: "Fix the sky\nplease".into(),
        position: [10.0, 12.0],
        popup: [10.0, 30.0, 200.0, 150.0],
        open: true,
        ..Default::default()
    }];
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let back = load_from_bytes(&bytes).unwrap();
    assert_eq!(back.measurement, doc.measurement);
    assert_eq!(back.notes, doc.notes);

    let m = read_manifest(&bytes).unwrap();
    let mut v = serde_json::to_value(&m.document).unwrap();
    v.as_object_mut().unwrap().remove("measurement");
    v.as_object_mut().unwrap().remove("notes");
    let old: photocraft_format::manifest::DocM = serde_json::from_value(v).unwrap();
    assert!(old.measurement.is_empty() && old.notes.is_empty());
}
