use super::*;
use std::io::Write;

enum Field {
    Int(i32),
    Long(i64),
    Byte(u8),
    Bool(bool),
    Ref(i32),
}

struct Writer(Vec<u8>);

impl Writer {
    fn int(&mut self, n: i32) {
        self.0.extend(n.to_le_bytes());
    }
    fn string(&mut self, s: &str) {
        let mut len = s.len();
        while len >= 128 {
            self.0.push((len as u8 & 127) | 128);
            len >>= 7;
        }
        self.0.push(len as u8);
        self.0.extend(s.as_bytes());
    }
    fn object(&mut self, id: i32, class: &str, fields: &[(&str, Field)]) {
        self.0.push(5);
        self.int(id);
        self.string(class);
        self.int(fields.len() as i32);
        for (name, _) in fields {
            self.string(name);
        }
        for (_, f) in fields {
            self.0.push(if matches!(f, Field::Ref(_)) { 2 } else { 0 });
        }
        for (_, f) in fields {
            match f {
                Field::Int(_) => self.0.push(8),
                Field::Long(_) => self.0.push(9),
                Field::Byte(_) => self.0.push(2),
                Field::Bool(_) => self.0.push(1),
                Field::Ref(_) => {}
            }
        }
        self.int(1);
        for (_, f) in fields {
            match f {
                Field::Int(n) => self.int(*n),
                Field::Long(n) => self.0.extend(n.to_le_bytes()),
                Field::Byte(n) => self.0.push(*n),
                Field::Bool(b) => self.0.push(u8::from(*b)),
                Field::Ref(id) => {
                    self.0.push(9);
                    self.int(*id);
                }
            }
        }
    }
}

/// Independent synthetic PDN writer: padded rows, forward references, a root cycle,
/// spare ArrayList capacity, and reversed gzip/raw chunks. There is no production PDN writer.
fn fixture(modes: &[i32], legacy: bool, compressed: bool) -> Vec<u8> {
    use Field::*;
    let mut w = Writer(b"PDN3\0\0\0\0\x01".to_vec());
    w.0.push(0);
    for n in [1, -1, 1, 0] {
        w.int(n);
    }
    w.0.push(12);
    w.int(1);
    w.string("PaintDotNet.Data");
    w.object(1, "PaintDotNet.Document", &[("width", Int(2)), ("height", Int(2)), ("layers", Ref(2))]);
    w.object(2, "PaintDotNet.LayerList", &[("parent", Ref(1)), ("ArrayList+_size", Int(modes.len() as i32)), ("ArrayList+_items", Ref(3))]);
    w.0.push(16);
    w.int(3);
    w.int(modes.len() as i32 + 2);
    for i in 0..modes.len() {
        w.0.push(9);
        w.int(16 + i as i32 * 10);
    }
    w.0.extend([13, 2]);
    for (i, mode) in modes.iter().enumerate() {
        let id = 16 + i as i32 * 10;
        w.object(
            id,
            "PaintDotNet.BitmapLayer",
            &[("Layer+width", Int(2)), ("Layer+height", Int(2)), ("surface", Ref(id + 1)), ("Layer+properties", Ref(id + 2)), ("properties", Ref(id + 3))],
        );
        w.object(id + 1, "PaintDotNet.Surface", &[("width", Int(2)), ("height", Int(2)), ("stride", Int(12)), ("scan0", Ref(id + 7))]);
        let mut fields =
            vec![("name", Ref(id + 4)), ("opacity", Byte(if i == 0 { 255 } else { 128 })), ("visible", Bool(i != 1)), ("isBackground", Bool(i == 0))];
        if !legacy {
            fields.push(("blendMode", Ref(id + 5)));
        }
        w.object(id + 2, "PaintDotNet.Layer+LayerProperties", &fields);
        w.object(id + 3, "PaintDotNet.BitmapLayer+BitmapLayerProperties", &[("blendOp", Ref(id + 6))]);
        w.0.push(6);
        w.int(id + 4);
        w.string(&format!("Läyer {i}"));
        w.object(id + 5, "PaintDotNet.LayerBlendMode", &[("value__", Int(*mode))]);
        w.object(id + 6, &format!("PaintDotNet.UserBlendOps+{}BlendOp", OLD_MODES.get(*mode as usize).unwrap_or(&"Unknown")), &[]);
        w.object(id + 7, "PaintDotNet.MemoryBlock", &[("length64", Long(24)), ("hasParent", Bool(false)), ("deferred", Bool(true))]);
    }
    w.0.push(11);
    for _ in modes {
        w.0.push(if compressed { 0 } else { 1 });
        w.0.extend(12u32.to_be_bytes());
        for index in [1u32, 0] {
            let raw = if index == 0 { vec![30, 20, 10, 255, 60, 50, 40, 128, 99, 99, 99, 99] } else { vec![90, 80, 70, 64, 120, 110, 100, 0, 99, 99, 99, 99] };
            let data = if compressed {
                let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                enc.write_all(&raw).unwrap();
                enc.finish().unwrap()
            } else {
                raw
            };
            w.0.extend(index.to_be_bytes());
            w.0.extend((data.len() as u32).to_be_bytes());
            w.0.extend(data);
        }
    }
    w.0
}

#[test]
fn all_pdn_blend_modes_and_layer_properties_import_and_save_natively() {
    let modes: Vec<_> = (0..14).collect();
    for legacy in [false, true] {
        for compressed in [false, true] {
            let bytes = fixture(&modes, legacy, compressed);
            // Detection must work without an extension too.
            let imported = crate::import("sample", &bytes).unwrap();
            assert!(imported.source_read_only);
            assert!(!imported.preview_only);
            let doc = imported.document;
            assert_eq!(doc.layers.len(), 14);
            assert_eq!((doc.size.width, doc.size.height), (2, 2));
            for (i, layer) in doc.layers.iter().enumerate() {
                assert_eq!(layer.name, format!("Läyer {i}"));
                assert_eq!(layer.blend, MODES[i]);
                assert_eq!(layer.visible, i != 1);
                assert_eq!(layer.opacity, if i == 0 { 1.0 } else { 128.0 / 255.0 });
                assert!(!layer.locks.all && !layer.locks.pixels && !layer.locks.position);
                let surface = layer.surface().unwrap();
                for (x, y, expected) in [(0, 0, [10, 20, 30, 255]), (1, 0, [40, 50, 60, 128]), (0, 1, [70, 80, 90, 64]), (1, 1, [100, 110, 120, 0])] {
                    let pixel = surface.pixel(x, y);
                    assert_eq!(pixel.iter().map(|v| (v * 255.0).round() as u8).collect::<Vec<_>>(), expected);
                }
            }
            let saved = crate::export(&doc, "sample.pcraft", &Default::default()).unwrap();
            assert!(saved.warnings.is_empty());
            let back = crate::import("sample.pcraft", &saved.bytes).unwrap().document;
            assert_eq!(doc.layers, back.layers);
            assert_eq!(photocraft_compose::flatten(&doc).px, photocraft_compose::flatten(&back).px);
        }
    }
}

#[test]
fn truncation_unknown_modes_and_corrupt_chunks_fail_without_panicking() {
    let raw = fixture(&[0], false, false);
    for len in 0..raw.len() {
        assert!(crate::import("truncated.pdn", &raw[..len]).is_err(), "length {len}");
    }
    assert!(crate::import("unknown.pdn", &fixture(&[999], false, false)).is_err());
    let mut corrupt = raw.clone();
    let block = corrupt.len() - 45;
    // Zero chunk size.
    corrupt[block + 1..block + 5].fill(0);
    assert!(crate::import("corrupt.pdn", &corrupt).is_err());
    let mut duplicate = raw.clone();
    // Both row chunks claim index 1.
    duplicate[block + 25..block + 29].copy_from_slice(&1u32.to_be_bytes());
    assert!(crate::import("duplicate.pdn", &duplicate).is_err());
    let mut zipped = fixture(&[0], false, true);
    let last = zipped.len() - 1;
    zipped[last] ^= 1;
    assert!(crate::import("checksum.pdn", &zipped).is_err());
}

#[test]
fn pdn_import_can_be_cancelled_during_parsing_and_pixel_decode() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let bytes = fixture(&[0, 1], false, true);
    for limit in [0, 10] {
        let checks = AtomicUsize::new(0);
        let cancel = || checks.fetch_add(1, Ordering::Relaxed) >= limit;
        let result = crate::import_with("cancel.pdn", &bytes, &Interrupt::cancel_only(&cancel));
        assert!(matches!(result, Err(IoError::Cancelled)));
    }
    let cancel = AtomicUsize::new(0);
    let no = || cancel.load(Ordering::Relaxed) != 0;
    let progress = |_| {
        cancel.store(1, Ordering::Relaxed);
    };
    let result = crate::import_with("cancel.pdn", &bytes, &Interrupt::new(&no, &progress));
    assert!(matches!(result, Err(IoError::Cancelled)));
}

proptest::proptest! {
    #[test]
    fn arbitrary_serialized_data_returns_an_error_without_panicking(data in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..4096)) {
        let mut bytes = b"PDN3\0\0\0\0\x01".to_vec();
        bytes.extend(data);
        let _ = crate::import("fuzz.pdn", &bytes);
    }

    #[test]
    fn mutated_layered_documents_never_panic(edits in proptest::collection::vec((proptest::prelude::any::<usize>(), proptest::prelude::any::<u8>()), 1..8)) {
        let mut bytes = fixture(&[0, 5, 13], false, true);
        for (offset, value) in edits {
            let index = offset % bytes.len();
            bytes[index] = value;
        }
        let _ = crate::import("mutated.pdn", &bytes);
    }
}

#[test]
fn psd_export_reports_paint_net_blend_loss() {
    let mut doc = crate::import("sample.pdn", &fixture(&[5], false, false)).unwrap().document;
    doc.layers[0].effects.items.push(photocraft_doc::effects::Effect::ColorOverlay {
        common: photocraft_doc::effects::FxCommon::new(BlendMode::Glow, 1.0),
        color: photocraft_color::Color::BLACK,
    });
    let result = crate::export(&doc, "sample.psd", &Default::default()).unwrap();
    assert!(result.warnings.iter().any(|s| s.contains("Reflect") && s.contains("no PSD equivalent")));
    assert!(result.warnings.iter().any(|s| s.contains("Color Overlay") && s.contains("Glow") && s.contains("no PSD equivalent")));
    let native = crate::export(&doc, "sample.pcraft", &Default::default()).unwrap();
    assert_eq!(crate::import("sample.pcraft", &native.bytes).unwrap().document.layers, doc.layers);
}

#[test]
fn malformed_pdn_reports_a_format_error() {
    for bytes in [b"".as_slice(), b"PDN3", b"PDN3\xff\xff\xff", b"PDN4\0\0\0"] {
        assert!(matches!(crate::import("bad.pdn", bytes), Err(IoError::Pdn(_))));
    }
}

#[test]
fn serialized_counts_depth_and_references_are_bounded() {
    let header = || {
        let mut w = Writer(vec![0]);
        for n in [1, -1, 1, 0] {
            w.int(n);
        }
        w
    };
    let mut oversized = header();
    oversized.0.push(16);
    oversized.int(1);
    oversized.int(1_000_001);
    let mut nested = header();
    for id in 1..=66 {
        nested.0.push(16);
        nested.int(id);
        nested.int(1);
    }
    nested.0.extend([10, 11]);
    let mut duplicate = header();
    for _ in 0..2 {
        duplicate.0.push(6);
        duplicate.int(1);
        duplicate.string("duplicate");
    }
    duplicate.0.push(11);
    let mut null_run = header();
    null_run.0.push(16);
    null_run.int(1);
    null_run.int(1);
    null_run.0.extend([13, 2, 11]);
    let mut missing_class = header();
    missing_class.0.push(1);
    missing_class.int(1);
    missing_class.int(999);
    let mut missing_root = header();
    missing_root.0.push(11);
    for w in [oversized, nested, duplicate, null_run, missing_class, missing_root] {
        let mut reader = Reader { bytes: &w.0, pos: 0 };
        assert!(nrbf::parse(&mut reader, &Interrupt::NONE).is_err());
    }
}

#[test]
fn pdn_export_is_explicitly_unsupported() {
    let doc = Document::new("x", Size::new(1, 1), ColorMode::Rgb, SampleType::U8);
    assert!(matches!(crate::export(&doc, "x.pdn", &Default::default()), Err(IoError::Unsupported(_))));
}

#[test]
fn truncated_uncompressed_pixels_are_rejected_before_allocating() {
    // A raw (version 1) block of 256 MiB chunks that declares 1 GiB of pixels but holds a few bytes.
    let mut reader = Reader { bytes: &[1, 0x10, 0, 0, 0, 0, 0, 0, 0], pos: 0 };
    let e = pixels(&mut reader, 1 << 30, &Interrupt::NONE).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(e.contains("uncompressed pixel data is truncated"), "{e}");
}
