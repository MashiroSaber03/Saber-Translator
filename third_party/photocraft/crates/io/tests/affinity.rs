//! Affinity documents: native import (layers, artboards, shapes, paints, text, masks) and the
//! embedded preview only as a warned fallback that keeps its pixels and alpha; never written back.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType};
use photocraft_io::{ExportOptions, export, import};
use proptest::prelude::*;
use std::io::Write as _;

fn png() -> Vec<u8> {
    let img = Image::from_raw(2, 1, ChannelLayout::Rgba, SampleType::U8, vec![220, 40, 60, 255, 0, 0, 0, 0]).unwrap();
    photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).unwrap()
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut tagged = kind.to_vec();
    tagged.extend(data);
    let mut result = (data.len() as u32).to_be_bytes().to_vec();
    result.extend(&tagged);
    result.extend(crc32fast::hash(&tagged).to_be_bytes());
    result
}

/// A synthetic record envelope, not an Affinity document writer (there is no object graph).
fn file(png: &[u8]) -> Vec<u8> {
    let mut b = vec![0; 72];
    b[..4].copy_from_slice(photocraft_affinity::MAGIC);
    b[4..6].copy_from_slice(&12u16.to_le_bytes());
    b[8..12].copy_from_slice(b"nsrP");
    b[12..16].copy_from_slice(b"#Inf");
    b[24..32].copy_from_slice(&72u64.to_le_bytes());
    b[64..68].copy_from_slice(b"Prot");
    b.extend(b"\xff\xff\xff\xffThmb");
    b.extend(1u32.to_le_bytes());
    b.extend((png.len() as u32 + 13).to_le_bytes());
    b.extend(29u32.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend((png.len() as u32).to_le_bytes());
    b.push(1);
    b.extend(png);
    b
}

#[test]
fn preview_is_named_and_warns_at_its_actual_size() {
    let r = import("art.af", &file(&png())).unwrap();
    assert_eq!((r.document.size.width, r.document.size.height), (2, 1));
    assert_eq!(r.document.layers.len(), 1);
    assert_eq!(r.document.layers[0].name, "Affinity preview");
    assert!(r.source_read_only && r.preview_only);
    assert!(r.warnings[0].contains("2×1"));
    assert!(r.warnings[0].contains("native Affinity document could not be read") && r.warnings[0].contains("no layers"), "{}", r.warnings[0]);
    let out = export(&r.document, "copy.png", &ExportOptions::default()).unwrap();
    let img = photocraft_codecs::decode(&out.bytes).unwrap();
    assert_eq!(img.data(), &[220, 40, 60, 255, 0, 0, 0, 0]);
    let native = export(&r.document, "copy.pcraft", &ExportOptions::default()).unwrap();
    let back = import("copy.pcraft", &native.bytes).unwrap();
    assert_eq!(back.document.size, r.document.size);
    assert!(!back.source_read_only && !back.preview_only);
    for ext in photocraft_io::affinity::EXTENSIONS {
        assert!(export(&r.document, ext, &ExportOptions::default()).unwrap_err().to_string().contains("Affinity export"));
    }
}

#[test]
fn magic_wins_over_the_name_and_renamed_legacy_files_are_rejected() {
    assert!(import("renamed.png", &file(&png())).unwrap().source_read_only);
    assert!(import("unknown", &file(&png())).unwrap().source_read_only);
    assert!(import("fake.af", &png()).is_err());
    // Affinity 1 and 2 (container versions 8 to 11) store the same preview record.
    let mut legacy = file(&png());
    legacy[4..6].copy_from_slice(&10u16.to_le_bytes());
    assert!(import("legacy.afphoto", &legacy).unwrap().preview_only);
    legacy[4..6].copy_from_slice(&13u16.to_le_bytes());
    assert!(import("future.af", &legacy).unwrap_err().to_string().contains("unknown container version"));
}

#[test]
fn preview_decoder_preserves_16_bit_gray_and_alpha() {
    let img = Image::from_u16(2, 1, ChannelLayout::GrayA, &[12345, 65535, 0, 0]).unwrap();
    let encoded = photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    let r = import("gray.af", &file(&encoded)).unwrap();
    assert_eq!(r.document.depth, photocraft_color::SampleType::U16);
    assert_eq!(r.document.mode, photocraft_color::ColorMode::Grayscale);
    let out = export(&r.document, "copy.png", &ExportOptions::default()).unwrap();
    let decoded = photocraft_codecs::decode(&out.bytes).unwrap();
    assert_eq!(decoded.sample_type(), SampleType::U16);
    assert_eq!(decoded.data(), img.data());
}

#[test]
fn previewless_current_files_return_an_export_fallback() {
    let mut bytes = file(&png());
    bytes[24..32].fill(0);
    let error = import("no-preview.af", &bytes).unwrap_err().to_string();
    assert!(error.contains("no embedded preview"), "{error}");
}

#[test]
fn corruption_and_every_truncation_fail_without_a_document() {
    let b = file(&png());
    for end in 0..b.len() {
        assert!(import("bad.af", &b[..end]).is_err(), "{end}");
    }
    let mut bad_crc = b.clone();
    // IHDR's CRC; the envelope reader rejects every PNG chunk's bad CRC.
    bad_crc[130] ^= 1;
    assert!(import("bad.af", &bad_crc).is_err());
}

#[test]
fn crc_failures_after_idat_cannot_be_tolerated_as_partial_pngs() {
    let mut bad_iend = file(&png());
    *bad_iend.last_mut().unwrap() ^= 1;
    assert!(import("bad-iend.af", &bad_iend).unwrap_err().to_string().contains("PNG chunk checksum"));

    let mut encoded = png();
    let tagged = b"tEXtnote\0preview";
    let mut chunk = ((tagged.len() - 4) as u32).to_be_bytes().to_vec();
    chunk.extend(tagged);
    chunk.extend(crc32fast::hash(tagged).to_be_bytes());
    let end = encoded.len() - 12;
    encoded.splice(end..end, chunk.clone());
    assert!(import("valid-ancillary.af", &file(&encoded)).is_ok());
    encoded[end + chunk.len() - 1] ^= 1;
    assert!(import("bad-ancillary.af", &file(&encoded)).unwrap_err().to_string().contains("PNG chunk checksum"));
}

#[test]
fn compressed_metadata_and_apng_fail_before_reaching_the_decoder() {
    for kind in [b"zTXt", b"iTXt", b"iCCP", b"acTL", b"fcTL", b"fdAT"] {
        let mut encoded = png();
        // A tiny declared payload stands in for arbitrarily large inflated metadata. It must
        // return the format's unsupported error, not be passed to a decompressor.
        let mut tagged = kind.to_vec();
        tagged.extend(b"preview\0\0\x78\x9c");
        let mut chunk = ((tagged.len() - 4) as u32).to_be_bytes().to_vec();
        chunk.extend(&tagged);
        chunk.extend(crc32fast::hash(&tagged).to_be_bytes());
        let end = encoded.len() - 12;
        encoded.splice(end..end, chunk);
        let error = import("unsupported.af", &file(&encoded)).unwrap_err().to_string();
        assert!(error.contains("unsupported Affinity file"), "{kind:?}: {error}");
        assert!(error.contains("PNG metadata") || error.contains("animated PNG"), "{error}");
    }
}

#[test]
fn many_valid_compressed_text_chunks_cannot_bypass_the_total_memory_budget() {
    let mut compressor = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    for _ in 0..1024 {
        compressor.write_all(&[b'x'; 1024]).unwrap();
    }
    let compressed = compressor.finish().unwrap();
    assert!(compressed.len() < 2048);
    let mut tagged = b"zTXtnote\0\0".to_vec();
    tagged.extend(compressed);
    let mut chunk = ((tagged.len() - 4) as u32).to_be_bytes().to_vec();
    chunk.extend(&tagged);
    chunk.extend(crc32fast::hash(&tagged).to_be_bytes());
    let mut encoded = png();
    let end = encoded.len() - 12;
    // A few MiB of encoded PNG would otherwise carry 1 GiB of aggregate text.
    encoded.splice(end..end, chunk.repeat(1024));
    let error = import("text-bomb.af", &file(&encoded)).unwrap_err().to_string();
    assert!(error.contains("compressed PNG metadata"), "{error}");
}

#[test]
fn invalid_critical_chunks_after_idat_fail_before_png_finish() {
    let original = png();
    for trailing in [chunk(b"IHDR", &original[16..29]), chunk(b"PLTE", &[0, 0, 0]), chunk(b"ABCD", &[])] {
        let mut encoded = original.clone();
        let end = encoded.len() - 12;
        encoded.splice(end..end, trailing);
        let error = import("bad-order.af", &file(&encoded)).unwrap_err().to_string();
        assert!(error.contains("Affinity file"), "{error}");
    }
    let mut encoded = original;
    let end = encoded.len() - 12;
    let mut interrupted = chunk(b"tEXt", b"note\0preview");
    interrupted.extend(chunk(b"IDAT", &[]));
    encoded.splice(end..end, interrupted);
    assert!(import("split-idat.af", &file(&encoded)).unwrap_err().to_string().contains("non-contiguous PNG image data"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn preview_and_decoder_survive_mutations(edits in prop::collection::vec((0usize..300, any::<u8>()), 0..16)) {
        let mut b = file(&png());
        for (i, v) in edits { if let Some(byte) = b.get_mut(i) { *byte = v; } }
        let _ = import("mutated.af", &b);
    }
}

mod native {
    use super::*;
    use photocraft_affinity::synth::{self, F, Method, tag};
    use photocraft_doc::LayerContent;

    fn rgba(r: f32, g: f32, b: f32, a: f32) -> F {
        F::Struct([r, g, b, a].iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    fn solid(id: u32, c: F) -> F {
        F::Def(
            id,
            vec![tag(b"FDsc")],
            vec![(tag(b"FDeF"), F::Def(id + 1, vec![tag(b"FilS")], vec![(tag(b"Colr"), F::Def(id + 2, vec![tag(b"RGBA")], vec![(tag(b"_col"), c)]))]))],
        )
    }

    fn node(x: f64, y: f64) -> Vec<u8> {
        let mut r = x.to_le_bytes().to_vec();
        r.extend(y.to_le_bytes());
        r.extend([1, 0]);
        r
    }

    /// A version-12 document at 144 ppi, 200×100: an artboard "Board" moved by (20, 10) holding
    /// a red ring (two subpaths, filled even-odd) with a 4 px blue stroke.
    pub(super) fn document(method: Method) -> Vec<u8> {
        let square = |s: f64, o: f64| vec![node(o, o), node(o + s, o), node(o + s, o + s), node(o, o + s), node(o, o)];
        let ring = F::Def(
            20,
            vec![tag(b"PCrv")],
            vec![
                (tag(b"Desc"), F::Str("Ring".into())),
                (
                    tag(b"Crvs"),
                    F::Obj(
                        tag(b"PCvD"),
                        vec![(
                            tag(b"Data"),
                            F::Pos(vec![
                                F::U8(0),
                                F::U32(2),
                                F::Bool(true),
                                F::Records(18, square(60.0, 0.0)),
                                F::Bool(true),
                                F::Records(18, square(20.0, 20.0)),
                            ]),
                        )],
                    ),
                ),
                (tag(b"BFFl"), F::Shared(vec![solid(21, rgba(1.0, 0.0, 0.0, 1.0))])),
                (tag(b"LIFl"), F::Shared(vec![solid(24, rgba(0.0, 0.0, 1.0, 1.0))])),
                (
                    tag(b"LILn"),
                    F::Shared(vec![F::Def(27, vec![tag(b"LDsc")], vec![(tag(b"LDeL"), F::Def(28, vec![tag(b"LSty")], vec![(tag(b"Wght"), F::F64(4.0))]))])]),
                ),
            ],
        );
        let board = F::Def(
            10,
            vec![tag(b"ShpN")],
            vec![
                (tag(b"Desc"), F::Str("Board".into())),
                (tag(b"ABEn"), F::Bool(true)),
                (tag(b"Shpe"), F::Def(11, vec![tag(b"ShNR")], vec![])),
                (tag(b"ShpB"), F::F64s(vec![0.0, 0.0, 120.0, 80.0])),
                (tag(b"Xfrm"), F::F64s(vec![1.0, 0.0, 20.0, 0.0, 1.0, 10.0])),
                (tag(b"BFFl"), F::Shared(vec![solid(12, rgba(1.0, 1.0, 1.0, 1.0))])),
                (tag(b"Chld"), F::Shared(vec![ring])),
            ],
        );
        let spread = F::Def(
            2,
            vec![tag(b"Sprd")],
            vec![(tag(b"SprB"), F::F64s(vec![0.0, 0.0, 200.0, 100.0])), (tag(b"SprT"), F::Bool(true)), (tag(b"Chld"), F::Shared(vec![board]))],
        );
        let doc = synth::stream(&[
            (tag(b"UVCn"), F::Obj(tag(b"UVCn"), vec![(tag(b"UPPI"), F::F64(144.0))])),
            (tag(b"DocR"), F::Def(1, vec![tag(b"DocN")], vec![(tag(b"Chld"), F::Shared(vec![spread]))])),
        ]);
        synth::container(&[("doc.dat", &doc, method)], Some(&png()))
    }

    #[test]
    fn artboards_shapes_and_paints_import_as_editable_layers() {
        for method in [Method::Stored, Method::Zlib, Method::Zstd] {
            let r = import("native.af", &document(method)).unwrap();
            assert!(r.source_read_only && !r.preview_only, "{:?}", r.warnings);
            let d = &r.document;
            assert_eq!((d.size.width, d.size.height, d.resolution_dpi), (200, 100, 144.0));
            let board = &d.layers[0];
            assert_eq!(board.name, "Board");
            let LayerContent::Group(g) = &board.content else { panic!("{:?}", board.content) };
            let ab = g.artboard.as_ref().unwrap();
            assert_eq!((ab.rect.x0, ab.rect.y0, ab.rect.x1, ab.rect.y1), (20, 10, 140, 90));
            let ring = &g.children[0];
            assert_eq!(ring.name, "Ring");
            let LayerContent::Shape(sh) = &ring.content else { panic!() };
            assert_eq!(sh.path.subpaths.len(), 2);
            assert_eq!(sh.path.subpaths[0].knots.len(), 4, "the closing duplicate knot is merged");
            assert_eq!(sh.path.subpaths[0].knots[0].anchor, photocraft_geom::Point::new(20.0, 10.0));
            assert_eq!(sh.stroke.as_ref().unwrap().width, 4.0);
            // The cache shows the ring: red on it, empty in its hole.
            let cache = sh.cache.as_ref().unwrap();
            let px = |x: i32, y: i32| cache.to_interleaved(photocraft_geom::Rect::new(x, y, x + 1, y + 1));
            assert_eq!(px(30, 20), [255, 0, 0, 255]);
            assert_eq!(px(50, 40)[3], 0, "even-odd hole");
            // Saved as a native PhotoCraft document and opened again.
            let saved = export(d, "copy.pcraft", &ExportOptions::default()).unwrap();
            let back = import("copy.pcraft", &saved.bytes).unwrap();
            assert_eq!(back.document.layers.len(), d.layers.len());
            assert!(export(d, "copy.psd", &ExportOptions::default()).is_ok());
            assert!(export(d, "copy.af", &ExportOptions::default()).is_err());
        }
    }

    #[test]
    fn a_damaged_native_document_falls_back_to_its_preview_with_the_reason() {
        let mut bytes = document(Method::Zlib);
        bytes[80] ^= 0xFF;
        let r = import("broken.af", &bytes).unwrap();
        assert!(r.preview_only);
        assert!(r.warnings[0].contains("could not be read (damaged Affinity file"), "{}", r.warnings[0]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn mutated_native_documents_never_panic(edits in prop::collection::vec((0usize..2048, any::<u8>()), 1..16), stored in any::<bool>()) {
            let mut b = document(if stored { Method::Stored } else { Method::Zstd });
            let n = b.len();
            for (i, v) in edits { b[i % n] = v; }
            let _ = import("mutated.af", &b);
        }
    }
}
