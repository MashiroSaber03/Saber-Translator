//! Layered TIFF (Photoshop layer data in tags 37724 / 34377): Document → TIFF → Document round
//! trips in every mode and depth, equivalence with the PSD path, the flat fall-backs, files
//! written by an independent implementation (psdtags, both byte orders), and malformed input.

mod common;

use common::*;
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, LayerContent};
use photocraft_io::*;
use photocraft_psd::tiff::{ByteOrder, ImageSourceData};

fn tiff(doc: &Document, layers: bool) -> ExportResult {
    export(doc, "x.tif", &ExportOptions { tiff_layers: layers, ..Default::default() }).expect("export")
}

/// The two Photoshop tags of a TIFF: (resources, layers).
fn tags(bytes: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let (r, l) = photocraft_codecs::tiff_photoshop_tags(bytes);
    (r.map(<[u8]>::to_vec), l.map(<[u8]>::to_vec))
}

fn roundtrip(doc: &Document) -> Document {
    let r = tiff(doc, true);
    let (resources, layers) = tags(&r.bytes);
    let layers = layers.expect("layer data");
    assert!(resources.is_some_and(|r| !r.is_empty()), "image resources");
    // The layer data matches the container's byte order and is byte-stable through the psd crate.
    let order = if photocraft_codecs::tiff_writes_little_endian() { ByteOrder::Little } else { ByteOrder::Big };
    let (isd, w) = ImageSourceData::from_bytes(&layers).expect("parse layer data");
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(isd.byte_order, order);
    assert_eq!(isd.to_bytes(order).unwrap().0, layers);
    // The TIFF is a plain image to any other reader: the composite at the document's size.
    let flat = photocraft_codecs::decode(&r.bytes).expect("flat decode");
    assert_eq!(flat.dimensions(), (doc.size.width, doc.size.height));
    // A TIFF holds one transparency channel: extra alpha channels (and a Quick Mask) are not
    // kept, and the export says so.
    let extra = doc.channels.len() + usize::from(doc.quick_mask.is_some());
    assert_eq!(r.warnings.iter().any(|w| w.contains("alpha channel(s) are not written")), extra > 0, "{:?}", r.warnings);
    import("x.tif", &r.bytes).expect("import").document
}

/// `doc` as a layered TIFF round trip returns it: without extra alpha channels or a Quick Mask.
fn expected(doc: &Document) -> Document {
    let mut d = doc.clone();
    d.channels.clear();
    d.quick_mask = None;
    d
}

macro_rules! rt_case {
    ($name:ident, $mode:expr, $depth:expr, $f:expr) => {
        #[test]
        fn $name() {
            let d = gen_doc($mode, $depth, $f);
            let back = roundtrip(&d);
            assert_docs_eq(&expected(&d), &back);
        }
    };
}

rt_case!(rt_rgb8_all, ColorMode::Rgb, SampleType::U8, Features::ALL);
rt_case!(rt_rgb16_all, ColorMode::Rgb, SampleType::U16, Features::ALL);
rt_case!(rt_rgb32_all, ColorMode::Rgb, SampleType::F32, Features::ALL);
rt_case!(rt_gray8_all, ColorMode::Grayscale, SampleType::U8, Features::ALL);
rt_case!(rt_gray16_all, ColorMode::Grayscale, SampleType::U16, Features::ALL);
rt_case!(rt_gray32_all, ColorMode::Grayscale, SampleType::F32, Features::ALL);
rt_case!(rt_cmyk8_all, ColorMode::Cmyk, SampleType::U8, Features::ALL);
rt_case!(rt_cmyk16_all, ColorMode::Cmyk, SampleType::U16, Features::ALL);
rt_case!(rt_rgb8_pixels, ColorMode::Rgb, SampleType::U8, Features::PIXELS);
rt_case!(rt_gray16_pixels, ColorMode::Grayscale, SampleType::U16, Features::PIXELS);
rt_case!(rt_cmyk8_pixels, ColorMode::Cmyk, SampleType::U8, Features::PIXELS);

#[test]
fn tiff_and_psd_open_as_the_same_document() {
    for (mode, depth) in [(ColorMode::Rgb, SampleType::U8), (ColorMode::Cmyk, SampleType::U16), (ColorMode::Grayscale, SampleType::F32)] {
        let d = gen_doc(mode, depth, Features::ALL);
        let psd = export(&d, "x.psd", &ExportOptions::default()).unwrap();
        let from_psd = import("x.psd", &psd.bytes).unwrap().document;
        let from_tif = roundtrip(&d);
        assert_docs_eq(&expected(&from_psd), &from_tif);
        // The composite the TIFF carries is what the document looks like.
        let tif = tiff(&d, true);
        let flat = import(
            "flat.tif",
            &photocraft_codecs::encode(&photocraft_codecs::decode(&tif.bytes).unwrap(), photocraft_codecs::Format::Tiff, &Default::default()).unwrap(),
        )
        .unwrap()
        .document;
        let a = photocraft_compose::flatten(&d).px;
        let b = photocraft_compose::flatten(&flat).px;
        assert!(max_diff(&a, &b) <= 2.0 / 255.0, "{mode:?} {depth:?}: composite differs by {}", max_diff(&a, &b));
    }
}

#[test]
fn every_compression_keeps_the_layers() {
    use photocraft_codecs::TiffCompression;
    let d = gen_doc(ColorMode::Rgb, SampleType::U16, Features::PIXELS);
    for c in [TiffCompression::None, TiffCompression::Lzw, TiffCompression::Deflate, TiffCompression::PackBits] {
        let mut opts = ExportOptions { tiff_layers: true, ..Default::default() };
        opts.encode.tiff_compression = c;
        let r = export(&d, "x.tiff", &opts).unwrap();
        let back = import("x.tiff", &r.bytes).unwrap().document;
        assert_docs_eq(&d, &back);
    }
}

/// The Artist tag's text, and the bytes every EXIF assertion looks for.
const ARTIST: &[u8] = b"fixture-artist";

/// A minimal little-endian EXIF (TIFF) block with a single Artist tag and no Orientation, so
/// `upright_exif` passes it through untouched.
fn exif() -> Vec<u8> {
    let mut artist = ARTIST.to_vec();
    artist.push(0); // ASCII NUL terminator
    let mut v = b"II*\0\x08\0\0\0".to_vec(); // TIFF header, first IFD at 8
    v.extend_from_slice(&1u16.to_le_bytes()); // one entry
    v.extend_from_slice(&0x013bu16.to_le_bytes()); // Artist
    v.extend_from_slice(&2u16.to_le_bytes()); // ASCII
    v.extend_from_slice(&(artist.len() as u32).to_le_bytes());
    v.extend_from_slice(&26u32.to_le_bytes()); // value offset: 8 + 2 + 12 + 4
    v.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    v.extend_from_slice(&artist);
    v
}

/// #1545: a layered TIFF carries EXIF as Photoshop resource 1058 and reads it back, so it must
/// not report a loss it never makes. A flat TIFF writes it as its own tags and keeps it too.
#[test]
fn layered_tiff_keeps_exif_without_claiming_a_loss() {
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    d.metadata.exif = Some(std::sync::Arc::new(exif()));

    let r = tiff(&d, true);
    assert!(!r.warnings.iter().any(|w| w.contains("EXIF")), "a layered TIFF keeps EXIF, so it must not claim to drop it: {:?}", r.warnings);
    // ...and the tags really are in the file, so the missing warning is a true absence of loss.
    let back = import("x.tif", &r.bytes).unwrap().document;
    let kept = back.metadata.exif.as_deref().expect("EXIF survives the layered export");
    assert!(kept.windows(ARTIST.len()).any(|w| w == ARTIST), "the Artist tag did not survive: {kept:?}");

    // The flat path writes the EXIF block's tags as the TIFF's own (here IFD0's Artist, which
    // reads back as the document's Artist text), so it reports no loss either.
    let flat = tiff(&d, false);
    assert!(!flat.warnings.iter().any(|w| w.contains("EXIF")), "a flat TIFF keeps EXIF: {:?}", flat.warnings);
    let back = import("x.tif", &flat.bytes).unwrap().document;
    let artist = std::str::from_utf8(ARTIST).unwrap().trim_end_matches('\0');
    assert!(back.metadata.text.iter().any(|(k, v)| k == "Artist" && v == artist), "{:?}", back.metadata.text);
}

#[test]
fn discard_layers_writes_a_flat_file() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let r = tiff(&d, false);
    assert_eq!(tags(&r.bytes), (None, None));
    assert!(r.warnings.iter().any(|w| w.contains("flattened")), "{:?}", r.warnings);
    let back = import("x.tif", &r.bytes).unwrap().document;
    assert_eq!(back.layers.len(), 1);
    assert!(!tiff_layers::would_write_layers(&back));
}

#[test]
fn a_plain_background_needs_no_layer_data() {
    let d = import("x.png", &export(&gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS), "x.png", &Default::default()).unwrap().bytes).unwrap().document;
    assert_eq!(d.layers.len(), 1);
    assert!(!tiff_layers::would_write_layers(&d));
    let r = tiff(&d, true);
    assert_eq!(tags(&r.bytes), (None, None));
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    let back = import("x.tif", &r.bytes).unwrap().document;
    assert_docs_eq(&d, &back);
}

#[test]
fn lab_documents_are_saved_flat_with_a_warning() {
    let d = gen_doc(ColorMode::Lab, SampleType::U8, Features::PIXELS);
    let r = tiff(&d, true);
    assert_eq!(tags(&r.bytes), (None, None));
    assert!(r.warnings.iter().any(|w| w.contains("Lab") && w.contains("layers are not kept")), "{:?}", r.warnings);
    assert!(import("x.tif", &r.bytes).is_ok());
}

#[test]
fn transparency_is_straight_in_the_composite() {
    // A translucent layer over nothing: the TIFF's composite keeps its colour under the alpha
    // instead of being matted against white like a PSD's merged image.
    let mut d = Document::new("t", photocraft_geom::Size::new(4, 4), ColorMode::Rgb, SampleType::U8);
    let fmt = d.pixel_format();
    let mut l = raster("Red", fmt, photocraft_geom::Rect::new(0, 0, 4, 4), 0, false);
    if let LayerContent::Raster(s) = &mut l.content {
        let px: Vec<f32> = (0..16).flat_map(|_| [1.0, 0.0, 0.0, 0.5]).collect();
        s.write_region(photocraft_geom::Rect::new(0, 0, 4, 4), &px);
    }
    d.layers.push(l);
    let r = tiff(&d, true);
    let flat = photocraft_codecs::decode(&r.bytes).unwrap();
    assert!(flat.layout().has_alpha());
    let px = flat.data();
    assert_eq!(&px[..4], &[255, 0, 0, 128]);
    let back = import("x.tif", &r.bytes).unwrap().document;
    assert_docs_eq(&d, &back);
}

#[test]
fn big_documents_use_the_psb_signature() {
    // Forcing PSB stands in for a > 30000 px document.
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let r = export(&d, "x.tif", &ExportOptions { force_psb: true, tiff_layers: true, ..Default::default() }).unwrap();
    let (_, layers) = tags(&r.bytes);
    assert!(layers.unwrap().starts_with(photocraft_psd::tiff::SIGNATURE_PSB));
    let back = import("x.tif", &r.bytes).unwrap().document;
    assert_docs_eq(&d, &back);
}

/// Files written by psdtags (`scripts/layered_tiff_fixtures.py`): two RGB layers, "Background"
/// (RLE) and an offset, translucent "Upper é" (ZIP, Multiply, opacity 200), in both byte orders
/// at 8 and 16 bits. Binary files stay out of this repository: run the script to write them
/// to `corpus/layered-tiff/` (gitignored); without them these checks are skipped.
fn fixture(name: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(format!("{}/../../corpus/layered-tiff/{name}", env!("CARGO_MANIFEST_DIR"))).ok();
    if bytes.is_none() {
        eprintln!("skipped {name}: run `python scripts/layered_tiff_fixtures.py` to write corpus/layered-tiff/");
    }
    bytes
}

#[test]
fn psdtags_fixtures_open_with_their_layers() {
    for (name, depth) in [
        ("layered-le-8bit.tif", SampleType::U8),
        ("layered-be-8bit.tif", SampleType::U8),
        ("layered-le-16bit.tif", SampleType::U16),
        ("layered-be-16bit.tif", SampleType::U16),
    ] {
        let Some(bytes) = fixture(name) else { continue };
        let r = import(name, &bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let d = &r.document;
        assert_eq!(d.size, photocraft_geom::Size::new(6, 4), "{name}");
        assert_eq!(d.mode, ColorMode::Rgb);
        assert_eq!(d.depth, depth, "{name}");
        let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["Background", "Upper \u{e9}"], "{name}");
        let upper = &d.layers[1];
        assert_eq!(upper.blend, photocraft_color::BlendMode::Multiply);
        assert!((upper.opacity - 200.0 / 255.0).abs() < 1e-6);
        let LayerContent::Raster(s) = &upper.content else { panic!("raster") };
        let bounds = s.content_bounds();
        assert_eq!((bounds.x0, bounds.y0, bounds.width(), bounds.height()), (1, 1, 4, 3), "{name}");
        // Pixel (x=2, y=0) of the upper layer: the generator's ramp, alpha 160/255.
        let px = s.pixel(3, 1);
        let expect = |v: u32| (v % 256) as f32 / 255.0;
        assert!((px[0] - expect(2 * 40 + 7)).abs() < 1e-3, "{name}: {px:?}");
        assert!((px[1] - expect(7 * 3)).abs() < 1e-3, "{name}: {px:?}");
        assert!((px[3] - 160.0 / 255.0).abs() < 1e-3, "{name}: {px:?}");
        let LayerContent::Raster(bg) = &d.layers[0].content else { panic!("raster") };
        let b = bg.pixel(5, 3);
        assert!((b[0] - expect(5 * 40 + 1)).abs() < 1e-3, "{name}: {b:?}");
        assert!((b[3] - 1.0).abs() < 1e-6);
        assert_eq!(d.resolution_dpi, 300.0);
        // tifffile writes a Software tag; nothing else is worth a warning.
        assert!(r.warnings.iter().all(|w| w.contains("text metadata")), "{name}: {:?}", r.warnings);
        // psdtags wrote the composite's alpha sample without the transparency flag, so by the
        // PSD rules it is an alpha channel (which a re-save cannot keep). Re-saving keeps
        // everything else; the file is then in the encoder's byte order.
        assert_eq!(d.channels.len(), 1, "{name}");
        let out = tiff(d, true);
        let back = import(name, &out.bytes).unwrap().document;
        assert_docs_eq(&expected(d), &back);
    }
}

/// Advanced Blending (`knko`, `infx`, `clbl`, `tsly`, `lmgm`, `vmgm`), fill opacity (`iOpa`),
/// channel restrictions (`brst`) and Blend If (the records' blending ranges) survive a layered
/// TIFF with its layer data in either byte order: the encoder's own, and the other one made by
/// transcoding the tag (the same length, spliced in place).
#[test]
fn advanced_blending_survives_layered_tiff_in_both_byte_orders() {
    use photocraft_doc::{AdvancedBlending, BlendRange, Knockout};
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut d = gen_doc(ColorMode::Rgb, depth, Features::PIXELS);
        let adv = AdvancedBlending {
            knockout: Knockout::Deep,
            blend_interior: true,
            blend_clipped: false,
            transparency_shapes: false,
            layer_mask_hides_effects: true,
            vector_mask_hides_effects: true,
        };
        d.layers[1].advanced = adv;
        d.layers[1].fill_opacity = 64.0 / 255.0;
        d.layers[1].excluded_channels = 0b010;
        d.layers[1].blend_if.set(0, [BlendRange { black: [10, 40], white: [200, 250] }, BlendRange { black: [0, 0], white: [128, 128] }]);
        d.layers[2].advanced.knockout = Knockout::Shallow;
        let r = tiff(&d, true);
        let (_, layers) = tags(&r.bytes);
        let layers = layers.expect("layer data");
        let (isd, w) = ImageSourceData::from_bytes(&layers).expect("parse");
        assert!(w.is_empty(), "{w:?}");
        let native = isd.byte_order;
        let other = if native == ByteOrder::Little { ByteOrder::Big } else { ByteOrder::Little };
        let (swapped, w) = isd.to_bytes(other).expect("transcode");
        assert!(w.is_empty(), "{depth:?}: blocks dropped while transcoding: {w:?}");
        assert_eq!(swapped.len(), layers.len());
        let at = r.bytes.windows(layers.len()).position(|x| x == &layers[..]).expect("tag data in the file");
        let mut flipped = r.bytes.clone();
        flipped[at..at + layers.len()].copy_from_slice(&swapped);
        for (order, bytes) in [(native, &r.bytes), (other, &flipped)] {
            let back = import("x.tif", bytes).unwrap_or_else(|e| panic!("{depth:?} {order:?}: {e}")).document;
            let l = &back.layers[1];
            assert_eq!(l.advanced, adv, "{depth:?} {order:?}");
            assert_eq!(l.fill_opacity, 64.0 / 255.0, "{depth:?} {order:?}");
            assert_eq!(l.excluded_channels, 0b010, "{depth:?} {order:?}");
            assert_eq!(l.blend_if, d.layers[1].blend_if, "{depth:?} {order:?}");
            assert_eq!(back.layers[2].advanced.knockout, Knockout::Shallow, "{depth:?} {order:?}");
            assert_docs_eq(&expected(&d), &back);
        }
    }
}

#[test]
fn damaged_layer_data_opens_flattened() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let r = tiff(&d, true);
    let (_, layers) = tags(&r.bytes);
    let layers = layers.unwrap();
    let at = r.bytes.windows(layers.len()).position(|w| w == &layers[..]).unwrap();
    // Garbage after the signature: no layers, a warning, the composite as one layer.
    let mut bad = r.bytes.clone();
    for b in &mut bad[at + 36..at + 36 + 24] {
        *b ^= 0xa5;
    }
    let r2 = import("x.tif", &bad).unwrap();
    assert_eq!(r2.document.layers.len(), 1);
    assert!(r2.warnings.iter().any(|w| w.contains("could not be read")), "{:?}", r2.warnings);
    // Every truncation and a sweep of bit flips through the whole file: never a panic.
    for cut in (0..r.bytes.len()).step_by(11) {
        let _ = import("x.tif", &r.bytes[..cut]);
    }
    for i in (0..r.bytes.len()).step_by(29) {
        let mut b = r.bytes.clone();
        b[i] ^= 0x5a;
        let _ = import("x.tif", &b);
    }
    for name in ["layered-le-8bit.tif", "layered-be-16bit.tif"] {
        let Some(bytes) = fixture(name) else { continue };
        for cut in (0..bytes.len()).step_by(7) {
            let _ = import(name, &bytes[..cut]);
        }
        for i in (0..bytes.len()).step_by(5) {
            let mut b = bytes.clone();
            b[i] ^= 0xff;
            let _ = import(name, &b);
        }
    }
}

#[test]
fn default_export_options_write_a_flat_tiff() {
    // Scripted and agent saves (CLI, batch, MCP) build on the defaults: layers only on request.
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    assert!(tiff_layers::would_write_layers(&d));
    let r = export(&d, "x.tif", &ExportOptions::default()).unwrap();
    assert_eq!(tags(&r.bytes), (None, None));
    assert_eq!(import("x.tif", &r.bytes).unwrap().document.layers.len(), 1);
}

/// #2940: layered TIFF data whose groups nest past the document's cap is rejected with the same
/// limit error as a PSD, instead of being converted.
#[test]
fn layered_tiff_nested_past_the_cap_is_rejected() {
    use photocraft_psd::{GroupSpec, LayerSpec, PixelData, PsdBuilder, PsdFile};
    let tiff_nested = |levels: usize| {
        let mut b = PsdBuilder::new(4, 4);
        for _ in 0..levels {
            b.begin_group(GroupSpec::new("g"));
        }
        b.push_layer(LayerSpec::new("deep", 0, 0, 1, 1, PixelData::Rgba8(vec![1, 2, 3, 4])));
        for _ in 0..levels {
            b.end_group().unwrap();
        }
        let file = PsdFile::from_bytes(&b.to_bytes().unwrap()).unwrap();
        let order = if photocraft_codecs::tiff_writes_little_endian() { ByteOrder::Little } else { ByteOrder::Big };
        let layers = ImageSourceData::from_psd(&file).to_bytes(order).unwrap().0;
        let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
        let mut img = photocraft_codecs::decode(&tiff(&d, false).bytes).unwrap();
        img.meta.photoshop_layers = Some(layers);
        photocraft_codecs::encode(&img, photocraft_codecs::Format::Tiff, &photocraft_codecs::EncodeOptions::default()).unwrap()
    };
    let err = import("deep.tif", &tiff_nested(101)).unwrap_err();
    assert!(err.to_string().contains("nested deeper than 100"), "{err}");
    let r = import("hundred.tif", &tiff_nested(100)).unwrap();
    assert_eq!(r.document.max_group_depth(), 100);
}
