//! Background opens (#210): `import_with` stops when cancelled and reports progress.

use std::sync::atomic::{AtomicU32, Ordering};

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent, Size};
use photocraft_geom::Rect;
use photocraft_raster::{Interrupt, Surface};

fn layered_psd() -> Vec<u8> {
    let mut d = Document::new("t", Size::new(32, 24), ColorMode::Rgb, SampleType::U8);
    for i in 0..5 {
        let mut s = Surface::new(d.pixel_format());
        s.fill_rect(Rect::new(i, i, 20 + i, 16 + i), &[0.2 * i as f32, 0.5, 0.7, 1.0]);
        d.layers.push(Layer::new(format!("L{i}"), LayerContent::Raster(s)));
    }
    photocraft_io::export(&d, "t.psd", &Default::default()).unwrap().bytes
}

#[test]
fn cancelled_import_fails_with_cancelled() {
    let bytes = layered_psd();
    let yes = || true;
    let r = photocraft_io::import_with("t.psd", &bytes, &Interrupt::cancel_only(&yes));
    assert!(matches!(r, Err(photocraft_io::IoError::Cancelled)), "{:?}", r.map(|_| ()));
}

#[test]
fn import_reports_monotonic_progress_per_layer() {
    let bytes = layered_psd();
    let last = AtomicU32::new(0);
    let calls = AtomicU32::new(0);
    let no = || false;
    let progress = |f: f32| {
        assert!(f >= f32::from_bits(last.load(Ordering::Relaxed)), "progress went backwards");
        last.store(f.to_bits(), Ordering::Relaxed);
        calls.fetch_add(1, Ordering::Relaxed);
    };
    let r = photocraft_io::import_with("t.psd", &bytes, &Interrupt::new(&no, &progress)).unwrap();
    assert_eq!(r.document.layers.len(), 5);
    assert_eq!(f32::from_bits(last.load(Ordering::Relaxed)), 1.0);
    assert!(calls.load(Ordering::Relaxed) >= 6, "one report per layer plus stages");
    // Same document as the plain import.
    let plain = photocraft_io::import("t.psd", &bytes).unwrap();
    assert_eq!(plain.document.layers.len(), r.document.layers.len());
}

#[test]
fn cancelled_brush_import_fails() {
    use photocraft_psd::abr::{LegacyBrush, LegacyTip, write_v12};
    let brush = |name: &str| LegacyBrush {
        name: name.into(),
        spacing: 10,
        anti_alias: true,
        tip: LegacyTip::Computed { diameter: 9, hardness: 100, angle: 0, roundness: 100 },
    };
    let bytes = write_v12(2, &[brush("a"), brush("b")], true).unwrap();
    assert_eq!(photocraft_io::abr_map::read_abr(&bytes, "g").unwrap().presets.len(), 2);
    let yes = || true;
    let r = photocraft_io::abr_map::read_abr_with(&bytes, "g", &Interrupt::cancel_only(&yes));
    assert_eq!(r.err().as_deref(), Some("cancelled"));
}
