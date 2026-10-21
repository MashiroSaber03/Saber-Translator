//! #3389: a background save's export reports its progress (so the status bar and the tab title
//! can read "Saving NN%") and can be cancelled, without changing a byte of what it writes.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_io::{ExportOptions, IoError};
use photocraft_raster::{Interrupt, Surface};

/// Three pixel layers (two in a group) of noise-like pixels at `depth`.
fn layered(depth: SampleType) -> Document {
    let mut d = Document::new("progress", Size::new(300, 200), ColorMode::Rgb, depth);
    let fmt = d.pixel_format();
    let layer = |name: &str, seed: usize, r: Rect| {
        let n = r.width() as usize * r.height() as usize * fmt.bytes_per_pixel();
        let px: Vec<u8> = (0..n).map(|i| ((i * 31 + seed * 17 + i / 7) % 251) as u8).collect();
        Layer::new(name, LayerContent::Raster(Surface::from_interleaved(fmt, r, &px)))
    };
    d.layers.push(layer("Back", 1, d.bounds()));
    d.layers.push(Layer::group("Group", vec![layer("A", 2, Rect::new(10, 10, 200, 150)), layer("B", 3, Rect::new(50, 40, 290, 190))]));
    d
}

/// Exports `doc` as `name` with progress recorded; the bytes and every fraction reported.
fn export_recorded(doc: &Document, name: &str, opts: &ExportOptions) -> (Vec<u8>, Vec<f32>) {
    let seen = Mutex::new(Vec::new());
    let cancel = || false;
    let progress = |f: f32| seen.lock().unwrap().push(f);
    let r = photocraft_io::export_with(doc, name, opts, &Interrupt::new(&cancel, &progress)).unwrap();
    (r.bytes, seen.into_inner().unwrap())
}

#[test]
fn export_reports_monotonic_progress_and_writes_the_same_bytes() {
    let layered_tiff = ExportOptions { tiff_layers: true, ..Default::default() };
    let cases: [(&str, ExportOptions, usize); 6] = [
        // Per layer record (3 layers + the group) and per composite band, then the bytes.
        ("a.psd", ExportOptions::default(), 6),
        ("a.psb", ExportOptions::default(), 6),
        ("a.tif", layered_tiff, 6),
        ("a.pcraft", ExportOptions::default(), 3),
        ("a.png", ExportOptions::default(), 2),
        ("a.jpg", ExportOptions::default(), 2),
    ];
    for depth in [SampleType::U8, SampleType::U16] {
        let doc = layered(depth);
        for (name, opts, min_reports) in &cases {
            if depth == SampleType::U16 && name.ends_with(".jpg") {
                continue;
            }
            let (bytes, seen) = export_recorded(&doc, name, opts);
            let plain = photocraft_io::export(&doc, name, opts).unwrap().bytes;
            // .pcraft bundles carry a save time, so only the formats without one compare exactly.
            if !name.ends_with(".pcraft") {
                assert!(bytes == plain, "{name} {depth:?}: progress reporting changed the bytes");
            }
            assert!(seen.len() >= *min_reports, "{name} {depth:?}: only {} reports: {seen:?}", seen.len());
            assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{name} {depth:?}: progress went backwards: {seen:?}");
            assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)), "{name} {depth:?}: {seen:?}");
            assert_eq!(seen.last().copied(), Some(1.0), "{name} {depth:?}: ends at 100%");
            assert!(seen.iter().any(|f| *f > 0.0 && *f < 1.0), "{name} {depth:?}: something between 0 and 100%: {seen:?}");
        }
    }
}

#[test]
fn psd_progress_moves_per_layer_before_the_composite() {
    let (_, seen) = export_recorded(&layered(SampleType::U8), "a.psd", &ExportOptions::default());
    // 4 layer records + the composite (weighing 2) = 6 units, the bytes after 90 %.
    let first = seen.first().copied().unwrap_or(0.0);
    assert!((first - 0.9 / 6.0).abs() < 1e-3, "the first layer is one sixth of the conversion: {seen:?}");
    let distinct = seen.iter().filter(|f| **f < 1.0).count();
    assert!(distinct >= 5, "{seen:?}");
}

#[test]
fn a_cancelled_export_stops_with_cancelled() {
    let doc = layered(SampleType::U8);
    for name in ["a.psd", "a.tif", "a.pcraft", "a.png"] {
        // Cancelled from the start, and cancelled after the first progress report.
        let cancelled = || true;
        let r = photocraft_io::export_with(&doc, name, &ExportOptions { tiff_layers: true, ..Default::default() }, &Interrupt::cancel_only(&cancelled));
        assert!(matches!(r, Err(IoError::Cancelled)), "{name}: {:?}", r.map(|r| r.bytes.len()));
        let stop = AtomicBool::new(false);
        let cancel = || stop.load(Ordering::Relaxed);
        let progress = |_: f32| stop.store(true, Ordering::Relaxed);
        let r = photocraft_io::export_with(&doc, name, &ExportOptions { tiff_layers: true, ..Default::default() }, &Interrupt::new(&cancel, &progress));
        assert!(matches!(r, Err(IoError::Cancelled)), "{name} midway: {:?}", r.map(|r| r.bytes.len()));
    }
}
