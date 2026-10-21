//! Opening a file whose adjustment layer can't be read warns, naming the layer and the reason,
//! instead of opening it silently without its effect (#1763).

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Adjustment, Document, Layer, LayerContent};
use photocraft_geom::Size;
use photocraft_io::*;

fn reopen(adj: Adjustment, path: &str, opts: &ExportOptions) -> (Document, Vec<String>) {
    let mut d = Document::new("a", Size::new(16, 16), ColorMode::Rgb, SampleType::U8);
    d.layers.push(Layer::new("Teal Look", LayerContent::Adjustment(adj)));
    let r = export(&d, path, opts).expect("export");
    let imp = import(path, &r.bytes).expect("import");
    (imp.document, imp.warnings)
}

#[test]
fn unreadable_adjustments_warn_on_open() {
    // A Color Lookup that uses a profile, not a LUT file: kept verbatim, not applied.
    let profile = Adjustment::Unsupported { psd_key: "clrL".into(), raw: vec![0, 2, 0, 0] };
    for (path, opts) in [("x.psd", ExportOptions::default()), ("x.psb", ExportOptions { force_psb: true, ..Default::default() })] {
        let (doc, warnings) = reopen(profile.clone(), path, &opts);
        assert!(matches!(&doc.layers[0].content, LayerContent::Adjustment(Adjustment::Unsupported { .. })), "{path}");
        let w: Vec<&String> = warnings.iter().filter(|w| w.contains("Teal Look")).collect();
        assert_eq!(w.len(), 1, "{path}: {warnings:?}");
        assert!(w[0].contains("Color Lookup") && w[0].contains("version 2") && w[0].contains("no effect"), "{path}: {}", w[0]);
    }
    // Another kind without a specific reason still names the layer and the adjustment.
    let (_, warnings) = reopen(Adjustment::Unsupported { psd_key: "levl".into(), raw: vec![0, 9] }, "x.psd", &ExportOptions::default());
    assert!(warnings.iter().any(|w| w.contains("Teal Look") && w.contains("Levels settings could not be read;")), "{warnings:?}");
}

#[test]
fn readable_adjustments_open_without_warnings() {
    let lut: Vec<f32> = (0..8).flat_map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32]).collect();
    let lookup = Adjustment::ColorLookup { name: "id.cube".into(), size: 2, lut: Some(std::sync::Arc::new(lut)), tetrahedral: false, dither: false };
    for adj in [lookup, Adjustment::Invert] {
        let (doc, warnings) = reopen(adj, "x.psd", &ExportOptions::default());
        assert!(!matches!(&doc.layers[0].content, LayerContent::Adjustment(Adjustment::Unsupported { .. })));
        assert!(!warnings.iter().any(|w| w.contains("Teal Look")), "{warnings:?}");
    }
}
