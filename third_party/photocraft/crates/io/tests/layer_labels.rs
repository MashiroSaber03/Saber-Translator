use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, LabelColor, Layer, Size};

#[test]
fn layer_labels_keep_the_psd_indices() {
    use photocraft_io::blocks::{label_from_index, label_index};
    let labels = [
        LabelColor::None,
        LabelColor::Red,
        LabelColor::Orange,
        LabelColor::Yellow,
        LabelColor::Green,
        LabelColor::Blue,
        LabelColor::Violet,
        LabelColor::Gray,
        LabelColor::Seafoam,
        LabelColor::Indigo,
        LabelColor::Magenta,
        LabelColor::Fuchsia,
    ];
    for (i, color) in labels.into_iter().enumerate() {
        assert_eq!(label_index(color), i as u16);
        assert_eq!(label_from_index(i as u16), color);
    }
    for i in [12, 255, u16::MAX] {
        assert_eq!(label_from_index(i), LabelColor::None);
    }
}

#[test]
fn layer_labels_psd_roundtrip_all_colors_and_depths() {
    for depth in SampleType::ALL {
        let mut doc = Document::new("Labels", Size::new(2, 2), ColorMode::Rgb, depth);
        for color in LabelColor::ALL {
            let mut child = Layer::raster("Child", doc.pixel_format());
            child.label = color;
            let mut group = Layer::group(color.label(), vec![child]);
            group.label = color;
            doc.layers.push(group);
        }
        let bytes = photocraft_io::export(&doc, "labels.psd", &Default::default()).unwrap().bytes;
        let back = photocraft_io::import("labels.psd", &bytes).unwrap().document;
        let labels = |d: &Document| d.walk().into_iter().map(|(_, _, l)| (l.name.clone(), l.label)).collect::<Vec<_>>();
        assert_eq!(labels(&back), labels(&doc), "{depth:?}");
        assert_eq!(back.depth, depth);
    }
}

#[test]
fn layer_labels_truncated_sheet_color_is_not_a_label() {
    let block = photocraft_psd::TaggedBlock::new(*b"lclr", vec![0]);
    assert!(matches!(block.parsed(), Some(Err(_))));
}
