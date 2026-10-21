//! Current .af artboards stay separate, editable and clipped through native/PSD saves.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "support/current_affinity.rs"]
mod fixture;

use fixture::{Marker, document};
use photocraft_affinity::synth::Method;
use photocraft_doc::Document;
use photocraft_geom::Rect;
use photocraft_io::{ExportOptions, export, import};

fn boards(doc: &Document) -> Vec<(String, Rect)> {
    doc.artboards().into_iter().map(|(_, name, a)| (name.to_string(), a.rect)).collect()
}

fn pixels(doc: &Document) -> Vec<u8> {
    let bytes = export(doc, "boards.png", &ExportOptions::default()).unwrap().bytes;
    photocraft_codecs::decode(&bytes).unwrap().data().to_vec()
}

#[test]
fn current_property_artboards_keep_bounds_clipping_and_roundtrips() {
    for marker in [Marker::Legacy, Marker::Current] {
        for method in [Method::Stored, Method::Zlib, Method::Zstd] {
            let loaded = import("synthetic.af", &document(marker, false, method)).unwrap();
            assert!(!loaded.preview_only && loaded.source_read_only);
            let d = &loaded.document;
            assert_eq!(boards(d), [("Board 100".into(), Rect::new(0, 0, 16, 16)), ("Board 200".into(), Rect::new(24, 0, 40, 16))]);
            let expected = pixels(d);
            assert_eq!(expected.len(), 40 * 16 * 4);
            for (i, px) in expected.as_chunks::<4>().0.iter().enumerate() {
                let x = i % 40;
                assert_eq!(px, if (16..24).contains(&x) { &[0, 0, 0, 0] } else { &[255, 0, 0, 255] }, "overflow must be clipped at the board edges");
            }
            for name in ["copy.pcraft", "copy.psd"] {
                let bytes = export(d, name, &ExportOptions::default()).unwrap().bytes;
                let reopened = import(name, &bytes).unwrap().document;
                assert_eq!(boards(&reopened), boards(d), "{name}");
                assert_eq!(pixels(&reopened), expected, "{name}");
            }
        }
    }
}

#[test]
fn current_properties_mark_converted_curve_artboards_too() {
    let loaded = import("curves.af", &document(Marker::Current, true, Method::Stored)).unwrap();
    assert_eq!(loaded.document.artboards().len(), 2);
    assert!(loaded.warnings.iter().any(|w| w.contains("converted artboard outline")));
}

#[test]
fn other_properties_do_not_turn_ordinary_shapes_into_artboards() {
    let d = import("shapes.af", &document(Marker::OtherProperties, false, Method::Stored)).unwrap().document;
    assert!(!d.has_artboards());
}
