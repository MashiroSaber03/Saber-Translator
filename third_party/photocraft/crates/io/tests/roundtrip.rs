//! Document → PSD → Document round trips.

mod common;

use common::*;
use photocraft_color::{ColorMode, SampleType};
use photocraft_io::*;
use photocraft_psd::PsdFile;

fn roundtrip(doc: &photocraft_doc::Document) -> photocraft_doc::Document {
    let r = export(doc, "x.psd", &ExportOptions::default()).expect("export");
    let file = PsdFile::from_bytes(&r.bytes).expect("parse");
    // Written files are byte-stable through the psd crate.
    assert_eq!(file.to_bytes().unwrap(), r.bytes);
    let imp = import("x.psd", &r.bytes).expect("import");
    imp.document
}

macro_rules! rt_case {
    ($name:ident, $mode:expr, $depth:expr, $f:expr) => {
        #[test]
        fn $name() {
            let d = gen_doc($mode, $depth, $f);
            let back = roundtrip(&d);
            assert_docs_eq(&d, &back);
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
rt_case!(rt_lab8_all, ColorMode::Lab, SampleType::U8, Features::ALL);
rt_case!(rt_lab16_all, ColorMode::Lab, SampleType::U16, Features::ALL);
rt_case!(rt_rgb8_pixels, ColorMode::Rgb, SampleType::U8, Features::PIXELS);
rt_case!(rt_gray16_pixels, ColorMode::Grayscale, SampleType::U16, Features::PIXELS);
rt_case!(rt_cmyk8_pixels, ColorMode::Cmyk, SampleType::U8, Features::PIXELS);

#[test]
fn psb_roundtrip() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::ALL);
    let r = export(&d, "x.psb", &ExportOptions::default()).unwrap();
    assert_eq!(r.bytes[5], 2);
    let back = import("x.psb", &r.bytes).unwrap().document;
    assert_docs_eq(&d, &back);
}

#[test]
fn force_psb_option() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let r = export(&d, "x.psd", &ExportOptions { force_psb: true, ..Default::default() }).unwrap();
    assert_eq!(r.bytes[5], 2);
}

#[test]
fn import_sets_name() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let r = export(&d, "psd", &ExportOptions::default()).unwrap();
    assert_eq!(import("hello.psd", &r.bytes).unwrap().document.name, "hello.psd");
}

#[test]
fn empty_document() {
    let d = photocraft_doc::Document::new("e", photocraft_geom::Size::new(4, 3), ColorMode::Rgb, SampleType::U8);
    let back = roundtrip(&d);
    // A flattened, fully transparent file comes back as one background layer.
    assert!(back.layers.len() <= 1);
    assert_eq!(back.size, d.size);
}

/// A header declaring a zero width or height is corrupt (the spec range starts at 1): opening it
/// fails instead of producing an empty document, and a zero-sized document is never written.
#[test]
fn zero_sized_psd_is_rejected() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let good = export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes;
    for range in [14..18, 18..22, 14..22] {
        let mut b = good.clone();
        b[range.clone()].fill(0);
        let err = import("z.psd", &b).expect_err("zero-sized header must not open");
        assert!(err.to_string().contains("at least 1x1"), "{range:?}: {err}");
    }
    let empty = photocraft_doc::Document::new("e", photocraft_geom::Size::new(0, 3), ColorMode::Rgb, SampleType::U8);
    assert!(export(&empty, "x.psd", &ExportOptions::default()).is_err());
}

#[test]
fn round_trip_reaches_a_fixed_point() {
    // The first export renders fill layers ourselves; after import they carry
    // those pixels as their fill cache, so from the second generation on the
    // bytes are identical.
    let d = gen_doc(ColorMode::Rgb, SampleType::U16, Features::ALL);
    let a = export(&d, "a.psd", &ExportOptions::default()).unwrap().bytes;
    let b = export(&import("a.psd", &a).unwrap().document, "a.psd", &ExportOptions::default()).unwrap().bytes;
    let c = export(&import("b.psd", &b).unwrap().document, "a.psd", &ExportOptions::default()).unwrap().bytes;
    assert_eq!(b, c, "export(import(x)) == x for x = export(import(export(d)))");
    // Layer records are already stable in the first generation.
    let (fa, fb) = (PsdFile::from_bytes(&a).unwrap(), PsdFile::from_bytes(&b).unwrap());
    assert_eq!(fa.layers(), fb.layers());
}

#[test]
fn text_shape_smart_raw_blocks_survive() {
    use photocraft_doc::*;
    use std::sync::Arc;
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let fmt = d.pixel_format();
    let cache = pattern(fmt, photocraft_geom::Rect::new(1, 1, 6, 4), 42, true);
    let placed = Arc::new(vec![1u8, 2, 3, 4]);
    let mut smart = Layer::new(
        "Smart",
        LayerContent::Smart(SmartObject {
            contents_id: SmartContentsId::fresh(),
            source: SmartSource::Linked { path: String::new() },
            transform: photocraft_geom::Affine::IDENTITY,
            smart_filters: vec![],
            cache: Some(cache.clone()),
            psd_raw: Some(placed.clone()),
            filters_enabled: true,
            filter_mask: None,
            warp: None,
            stack_mode: None,
            perspective: None,
        }),
    );
    smart.psd_blocks = vec![(*b"PlLd", Arc::new(vec![0; 4])), (*b"vmsk", Arc::new(vec![5; 8])), (*b"luni", Arc::new(vec![0; 8]))];
    // The typed vector mask (as import produces it) keeps the raw block while unchanged.
    smart.vector_mask = photocraft_io::vector_map::vector_mask_from_block(&[5; 8], d.size.width, d.size.height);
    d.layers.push(smart);
    let mut shape = Layer::new("Shape", LayerContent::Shape(ShapeLayer { fill: None, cache: Some(cache.clone()), psd_raw: None, ..Default::default() }));
    shape.psd_blocks = vec![(*b"vscg", Arc::new(vec![1, 2, 3, 4])), (*b"vmsk", Arc::new(vec![7; 8]))];
    if let LayerContent::Shape(s) = &mut shape.content {
        s.path = photocraft_io::vector_map::path_from_vmsk(&[7; 8], d.size.width, d.size.height).unwrap().0;
    }
    d.layers.push(shape);
    let back = roundtrip(&d);
    let n = back.layers.len();
    let (bs, bsh) = (&back.layers[n - 2], &back.layers[n - 1]);
    assert!(matches!(bs.content, LayerContent::Smart(_)));
    assert!(matches!(bsh.content, LayerContent::Shape(_)));
    // psd_raw overrides the PlLd entry; the stale luni is regenerated, not kept.
    assert!(bs.psd_blocks.contains(&(*b"PlLd", placed.clone())));
    assert!(bs.psd_blocks.iter().any(|b| &b.0 == b"vmsk"));
    assert!(!bs.psd_blocks.iter().any(|b| &b.0 == b"luni"));
    if let LayerContent::Smart(s) = &bs.content {
        assert_eq!(s.psd_raw.as_deref(), Some(&*placed));
    }
    if let LayerContent::Shape(s) = &bsh.content {
        assert_eq!(s.psd_raw.as_deref(), Some(&vec![7u8; 8]));
    }
    assert_eq!(bsh.psd_blocks, d.layers[n - 1].psd_blocks);
}

#[test]
fn raster_layer_blocks_preserved() {
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let blocks =
        vec![(*b"vmsk", std::sync::Arc::new(vec![0u8; 12])), (*b"clbl", std::sync::Arc::new(vec![0u8, 0, 0, 0])), (*b"Zzzz", std::sync::Arc::new(vec![1u8]))];
    d.layers[1].psd_blocks = blocks.clone();
    // `clbl` is written from the field (in place), so the field agrees with the preserved block.
    d.layers[1].advanced = photocraft_doc::AdvancedBlending { blend_clipped: false, ..Default::default() };
    d.layers[1].vector_mask = photocraft_io::vector_map::vector_mask_from_block(&[0; 12], d.size.width, d.size.height);
    let back = roundtrip(&d);
    // An odd-length layer block comes back with its pad byte inside the length (as Photoshop
    // lays layer blocks out, #200); everything else is verbatim.
    let mut expected = blocks.clone();
    expected[2].1 = std::sync::Arc::new(vec![1u8, 0]);
    assert_eq!(back.layers[1].psd_blocks, expected);
    assert_eq!(roundtrip(&back).layers[1].psd_blocks, expected, "stable after the first save");
    assert_eq!(back.layers[1].vector_mask, d.layers[1].vector_mask);
}

#[test]
fn psd_ids_are_kept_and_deduplicated() {
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    d.layers[0].psd_id = Some(500);
    d.layers[1].psd_id = Some(500); // duplicate: second gets a fresh id
    d.layers[2].psd_id = Some(3);
    let back = roundtrip(&d);
    assert_eq!(back.layers[0].psd_id, Some(500));
    assert_ne!(back.layers[1].psd_id, Some(500));
    assert_eq!(back.layers[2].psd_id, Some(3));
    let ids: Vec<u32> = back.walk().iter().filter_map(|(_, _, l)| l.psd_id).collect();
    let set: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(set.len(), ids.len(), "ids unique");
}

#[test]
fn edited_adjustment_regenerates_block_unedited_keeps_raw() {
    use photocraft_doc::*;
    use std::sync::Arc;
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    // Levels block with non-default extra records (record 5 changed).
    let base =
        Adjustment::Levels { master: adjust::LevelsChannel::default(), per_channel: Default::default(), space: Default::default(), black: Default::default() };
    let mut l = Layer::new("lv", LayerContent::Adjustment(base.clone()));
    let psd = document_to_psd(&{
        let mut t = d.clone();
        t.layers.push(l.clone());
        t
    });
    let rec = psd.layers().iter().find(|r| r.name() == "lv").unwrap();
    let mut raw = rec.block(b"levl").unwrap().data.clone();
    raw[2 + 5 * 10 + 1] = 7; // record 5 in_black = 7 (not modelled)
    l.psd_blocks = vec![(*b"levl", Arc::new(raw.clone()))];
    d.layers.push(l);
    let f = document_to_psd(&d);
    assert_eq!(f.layers().iter().find(|r| r.name() == "lv").unwrap().block(b"levl").unwrap().data, raw, "unchanged: raw kept");
    if let LayerContent::Adjustment(Adjustment::Levels { master, .. }) = &mut d.layers.last_mut().unwrap().content {
        master.in_black = 10.0 / 255.0;
    }
    let f = document_to_psd(&d);
    let data = &f.layers().iter().find(|r| r.name() == "lv").unwrap().block(b"levl").unwrap().data;
    assert_ne!(data, &raw, "edited: regenerated");
    assert_eq!(data[3], 10);
}

#[test]
fn text_layer_roundtrip_with_tysh() {
    use photocraft_doc::*;
    use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value, VersionedDescriptor};
    // Minimal TySh: version, transform, text version, text descriptor, warp.
    let mut tysh = 1u16.to_be_bytes().to_vec();
    for v in [1.0f64, 0.0, 0.0, 1.0, 3.0, 4.0] {
        tysh.extend_from_slice(&v.to_be_bytes());
    }
    tysh.extend_from_slice(&50u16.to_be_bytes());
    let text = Descriptor::new("TxLr").with("Txt ", Value::Text(UnicodeString::new_nul("Hello")));
    tysh.extend(VersionedDescriptor::new(text).to_bytes());
    tysh.extend_from_slice(&1u16.to_be_bytes());
    tysh.extend(VersionedDescriptor::new(Descriptor::new("warp")).to_bytes());
    tysh.extend_from_slice(&[0; 16]);
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let fmt = d.pixel_format();
    d.layers.push(Layer::new(
        "Hello",
        LayerContent::Text(TextLayer {
            text: "Hello".into(),
            font_family: String::new(),
            size_pt: 0.0,
            color: photocraft_color::Color::BLACK,
            transform: photocraft_geom::Affine { m: [1.0, 0.0, 0.0, 1.0, 3.0, 4.0] },
            cache: Some(pattern(fmt, photocraft_geom::Rect::new(3, 4, 9, 8), 5, true)),
            psd_raw: Some(std::sync::Arc::new(tysh)),
            ..Default::default()
        }),
    ));
    let back = roundtrip(&d);
    let t = back.layers.last().unwrap();
    match &t.content {
        LayerContent::Text(tl) => {
            assert_eq!(tl.text, "Hello");
            assert_eq!(tl.transform.m, [1.0, 0.0, 0.0, 1.0, 3.0, 4.0]);
        }
        other => panic!("{}", other.kind_name()),
    }
    assert_layers_eq(d.layers.last().unwrap(), t, "");
}

#[test]
fn effects_raw_roundtrip() {
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let mut lfx = vec![0, 0, 0, 0];
    lfx.extend(photocraft_psd::testgen::sample_descriptor().to_bytes());
    // Typed items and the raw block must agree; the raw bytes are then kept.
    let (master, items) = photocraft_io::effects_map::parse_lfx2(&lfx).unwrap();
    d.layers[1].effects.enabled = master;
    d.layers[1].effects.items = items;
    d.layers[1].effects.psd_raw = Some(std::sync::Arc::new(lfx));
    let back = roundtrip(&d);
    assert_docs_eq(&d, &back);
}

#[test]
fn gradient_and_pattern_fills_roundtrip() {
    use photocraft_doc::*;
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    d.layers.push(Layer::new(
        "grad",
        LayerContent::Fill(Fill::gradient(
            vec![(0.0, photocraft_color::Color::rgb(1.0, 0.0, 0.0)), (1.0, photocraft_color::Color::rgb(0.0, 0.0, 1.0))],
            30.0,
            1.0,
            GradientStyle::Diamond,
            true,
        )),
    ));
    d.layers.push(Layer::new(
        "pat",
        LayerContent::Fill(Fill::Pattern { name: "Dots".into(), scale: 0.5, id: String::new(), angle: 0.0, link: true, phase: (0.0, 0.0) }),
    ));
    let back = roundtrip(&d);
    assert_docs_eq(&d, &back);
}

#[test]
fn psd_raw_resources_and_global_blocks_preserved() {
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    d.metadata.psd_resources.push((4001, "named".into(), std::sync::Arc::new(vec![1])));
    d.metadata.psd_global_blocks.push((*b"8BIM", *b"Patt", std::sync::Arc::new(vec![7; 6])));
    d.metadata.psd_global_blocks.push((*b"8B64", *b"lnk2", std::sync::Arc::new(vec![1, 2])));
    let back = roundtrip(&d);
    // A PSD holds no `8B64` blocks (Photoshop refuses the file): a kept PSB block is written as `8BIM`.
    let mut expected = d.metadata.clone();
    expected.psd_global_blocks[1].0 = *b"8BIM";
    assert_eq!(back.metadata, expected);
}

#[test]
fn document_to_psd_direct_api() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::ALL);
    let f = document_to_psd(&d);
    assert_eq!(f.header.width, 24);
    let (d2, _warnings) = psd_to_document(&f);
    assert_docs_eq(&d, &d2);
}

#[test]
fn big_document_goes_psb_with_warning() {
    let d = photocraft_doc::Document::new("big", photocraft_geom::Size::new(30_001, 1), ColorMode::Grayscale, SampleType::U8);
    let (f, w) = document_to_psd_with(&d, &PsdExportOptions::default());
    assert_eq!(f.header.version, photocraft_psd::Version::Psb);
    assert!(w.iter().any(|w| w.contains("PSB")));
}

#[test]
fn effects_roundtrip_typed_and_raw() {
    use photocraft_doc::*;
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    d.global_light = GlobalLight { angle: 45.0, altitude: 25.0 };
    d.layers[1].effects.items = vec![
        Effect::default_drop_shadow(),
        Effect::Stroke(StrokeFx {
            common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
            size: 3.0,
            position: StrokePosition::Center,
            paint: FxPaint::Color(photocraft_color::Color::rgb(0.0, 1.0, 0.0)),
        }),
        Effect::Stroke(StrokeFx {
            common: FxCommon::new(photocraft_color::BlendMode::Normal, 0.5),
            size: 6.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(photocraft_color::Color::rgb(0.0, 0.0, 1.0)),
        }),
    ];
    let back = roundtrip(&d);
    assert_eq!(back.global_light, d.global_light);
    let fx = &back.layers[1].effects;
    assert_eq!(fx.items.len(), 3);
    let raw = fx.psd_raw.clone().expect("lfx2 kept");
    // Unchanged → the same lfx2 bytes are written again.
    let f = document_to_psd(&back);
    let rec = f.layers().iter().find(|r| r.name() == back.layers[1].name).unwrap();
    assert_eq!(rec.block(b"lfx2").unwrap().data, *raw);
    // Edited → regenerated.
    let mut edited = back.clone();
    if let Effect::DropShadow(s) = &mut edited.layers[1].effects.items[0] {
        s.distance = 20.0;
    }
    let f = document_to_psd(&edited);
    let rec = f.layers().iter().find(|r| r.name() == back.layers[1].name).unwrap();
    let data = &rec.block(b"lfx2").unwrap().data;
    assert_ne!(data, &*raw);
    let (_, items) = photocraft_io::effects_map::parse_lfx2(data).unwrap();
    assert!(matches!(&items[0], Effect::DropShadow(s) if s.distance == 20.0));
}

#[test]
fn group_effects_use_lfxs() {
    use photocraft_doc::*;
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let n = d.layers.iter().position(|l| l.is_group()).unwrap();
    d.layers[n]
        .effects
        .items
        .push(Effect::ColorOverlay { common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0), color: photocraft_color::Color::rgb(1.0, 0.0, 0.0) });
    let f = document_to_psd(&d);
    let rec = f.layers().iter().find(|r| r.name() == d.layers[n].name && r.section_type().is_folder()).unwrap();
    assert!(rec.block(b"lfxs").is_some() && rec.block(b"lfx2").is_none());
    let (back, _) = psd_to_document(&f);
    assert_eq!(back.layers[n].effects.items.len(), 1);
}

#[test]
fn clearing_effects_drops_stale_blocks() {
    use photocraft_doc::*;
    let mut d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    d.layers[1].effects.items.push(Effect::default_drop_shadow());
    let back = roundtrip(&d);
    let mut cleared = back.clone();
    cleared.layers[1].effects = Effects { enabled: true, ..Default::default() };
    let f = document_to_psd(&cleared);
    let rec = f.layers().iter().find(|r| r.name() == back.layers[1].name).unwrap();
    assert!(rec.block(b"lfx2").is_none());
}

#[test]
fn link_groups_and_effects_reference_round_trip() {
    use photocraft_doc::{Document, Layer, LayerContent, Size};
    let mut d = Document::new("l", Size::new(8, 8), ColorMode::Rgb, SampleType::U8);
    let fmt = d.pixel_format();
    let mut a = Layer::raster("a", fmt);
    a.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[1.0, 0.0, 0.0, 1.0]);
    a.link_group = Some(7);
    a.effects.reference = Some((-201.0, 47.5));
    let mut b = Layer::raster("b", fmt);
    b.link_group = Some(7);
    let mut inner = Layer::raster("inner", fmt);
    inner.link_group = Some(3);
    let g = Layer::group("g", vec![inner]);
    d.layers = vec![a, b, g];
    let back = roundtrip(&d);
    assert_eq!(back.layers[0].link_group, Some(7));
    assert_eq!(back.layers[1].link_group, Some(7));
    assert_eq!(back.layers[2].link_group, None);
    let LayerContent::Group(gb) = &back.layers[2].content else { panic!() };
    assert_eq!(gb.children[0].link_group, Some(3));
    assert_eq!(back.layers[0].effects.reference, Some((-201.0, 47.5)));
    assert_eq!(back.layers[1].effects.reference, None);
    // Unchanged documents write the same resource bytes again.
    let r1 = export(&back, "x.psd", &ExportOptions::default()).unwrap().bytes;
    let f1 = PsdFile::from_bytes(&r1).unwrap();
    let res = |f: &PsdFile| f.resources.iter().find(|r| r.id == 1026).map(|r| r.data.clone());
    let f0 = PsdFile::from_bytes(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    assert_eq!(res(&f1), res(&f0));
    // One u16 per record: a, b, divider, inner, group (bottom first).
    assert_eq!(res(&f0).unwrap(), vec![0, 7, 0, 7, 0, 0, 0, 3, 0, 0]);
    // Large engine ids are renumbered by first appearance.
    let mut big = d.clone();
    big.layers[0].link_group = Some(1 << 40);
    big.layers[1].link_group = Some(1 << 40);
    let fb = PsdFile::from_bytes(&export(&big, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    assert_eq!(res(&fb).unwrap(), vec![0, 1, 0, 1, 0, 0, 0, 2, 0, 0]);
    // Unlinked documents write no 1026.
    let mut none = d.clone();
    none.layers[0].link_group = None;
    none.layers[1].link_group = None;
    if let LayerContent::Group(g) = &mut none.layers[2].content {
        g.children[0].link_group = None;
    }
    let fnone = PsdFile::from_bytes(&export(&none, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    assert!(res(&fnone).is_none());
}

#[test]
fn channel_restrictions_round_trip_as_brst() {
    use photocraft_doc::{Document, Layer, Size};
    let mut d = Document::new("c", Size::new(8, 8), ColorMode::Rgb, SampleType::U8);
    let mut a = Layer::raster("a", d.pixel_format());
    a.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[1.0, 0.0, 0.0, 1.0]);
    a.excluded_channels = 0b110;
    let b = Layer::raster("b", d.pixel_format());
    d.layers = vec![a, b];
    let f = document_to_psd(&d);
    let rec = f.layers().iter().find(|r| r.name() == "a").unwrap();
    assert_eq!(rec.block(b"brst").unwrap().data, vec![0, 0, 0, 1, 0, 0, 0, 2]);
    assert!(f.layers().iter().find(|r| r.name() == "b").unwrap().block(b"brst").is_none());
    let back = roundtrip(&d);
    assert_eq!(back.layers[0].excluded_channels, 0b110);
    assert_eq!(back.layers[1].excluded_channels, 0);
    // Cleared restrictions drop the preserved block.
    let mut cleared = back.clone();
    cleared.layers[0].excluded_channels = 0;
    let f = document_to_psd(&cleared);
    assert!(f.layers().iter().find(|r| r.name() == "a").unwrap().block(b"brst").is_none());
}

/// Advanced Blending maps to `knko` / `infx` / `clbl` / `tsly` / `lmgm` / `vmgm` (one byte +
/// three of padding), at every depth: non-default values are written, default layers gain no
/// blocks, and imported blocks are rewritten in place.
#[test]
fn advanced_blending_round_trips_as_tagged_blocks() {
    use photocraft_doc::{AdvancedBlending, Document, Knockout, Layer, Size};
    for depth in SampleType::ALL {
        let mut d = Document::new("a", Size::new(8, 8), ColorMode::Rgb, depth);
        let mut a = Layer::raster("a", d.pixel_format());
        a.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[1.0, 0.0, 0.0, 1.0]);
        let adv = AdvancedBlending {
            knockout: Knockout::Deep,
            blend_interior: true,
            blend_clipped: false,
            transparency_shapes: false,
            layer_mask_hides_effects: true,
            vector_mask_hides_effects: true,
        };
        a.advanced = adv;
        let mut g = Layer::group("g", vec![Layer::raster("c", d.pixel_format())]);
        g.advanced.knockout = Knockout::Shallow;
        g.fill_opacity = 128.0 / 255.0;
        let b = Layer::raster("b", d.pixel_format());
        d.layers = vec![a, b, g];
        let f = document_to_psd(&d);
        let rec = |name: &str| f.layers().iter().find(|r| r.name() == name).unwrap().clone();
        let ra = rec("a");
        for (k, v) in [(b"knko", 2u8), (b"infx", 1), (b"clbl", 0), (b"tsly", 0), (b"lmgm", 1), (b"vmgm", 1)] {
            assert_eq!(ra.block(k).unwrap().data, vec![v, 0, 0, 0], "{depth:?} {}", String::from_utf8_lossy(k));
        }
        for k in [b"knko", b"infx", b"clbl", b"tsly", b"lmgm", b"vmgm"] {
            assert!(rec("b").block(k).is_none(), "default layer gained {}", String::from_utf8_lossy(k));
        }
        assert_eq!(rec("g").block(b"knko").unwrap().data, vec![1, 0, 0, 0]);
        let back = roundtrip(&d);
        assert_eq!(back.layers[0].advanced, adv, "{depth:?}");
        assert!(back.layers[1].advanced.is_default());
        assert_eq!(back.layers[2].advanced.knockout, Knockout::Shallow);
        assert_eq!(back.layers[2].fill_opacity, 128.0 / 255.0);
        // Reset to the defaults: the imported blocks stay (in place) with default values.
        let mut reset = back.clone();
        reset.layers[0].advanced = AdvancedBlending::default();
        let f = document_to_psd(&reset);
        let ra = f.layers().iter().find(|r| r.name() == "a").unwrap();
        assert_eq!(ra.block(b"knko").unwrap().data, vec![0, 0, 0, 0]);
        assert_eq!(ra.block(b"clbl").unwrap().data, vec![1, 0, 0, 0]);
        assert!(roundtrip(&reset).layers[0].advanced.is_default());
    }
}

#[test]
fn blend_if_round_trips_as_blending_ranges() {
    use photocraft_doc::{BlendIf, BlendRange, Document, Layer, Size};
    let mut d = Document::new("b", Size::new(8, 8), ColorMode::Rgb, SampleType::U8);
    let mut a = Layer::raster("a", d.pixel_format());
    a.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[1.0, 1.0, 1.0, 1.0]);
    let mut bi = BlendIf::default();
    // Gray › This Layer: hide the whites, fading from 200 to 230.
    bi.set(0, [BlendRange { black: [0, 0], white: [200, 230] }, BlendRange::FULL]);
    // Green › Underlying Layer: show only over values from 64 up.
    bi.set(2, [BlendRange::FULL, BlendRange { black: [64, 64], white: [255, 255] }]);
    a.blend_if = bi.clone();
    let mut g = Layer::group("g", vec![Layer::raster("c", d.pixel_format())]);
    g.blend_if.set(1, [BlendRange { black: [1, 2], white: [3, 4] }, BlendRange::FULL]);
    let b = Layer::raster("b", d.pixel_format());
    d.layers = vec![a, b, g];
    let f = document_to_psd(&d);
    let rec = f.layers().iter().find(|r| r.name() == "a").unwrap();
    let full = [0u8, 0, 255, 255, 0, 0, 255, 255];
    let mut want = Vec::new();
    want.extend_from_slice(&[0, 0, 200, 230, 0, 0, 255, 255]);
    want.extend_from_slice(&full);
    want.extend_from_slice(&[0, 0, 255, 255, 64, 64, 255, 255]);
    want.extend_from_slice(&full);
    assert_eq!(rec.blending_ranges.data, want);
    // Layers without Blend If keep full ranges for gray + R, G, B.
    let rb = f.layers().iter().find(|r| r.name() == "b").unwrap();
    assert_eq!(rb.blending_ranges, photocraft_psd::BlendingRanges::full(3));
    let back = roundtrip(&d);
    assert_eq!(back.layers[0].blend_if, bi);
    assert!(back.layers[1].blend_if.is_default());
    assert_eq!(back.layers[2].blend_if.get(1)[0], BlendRange { black: [1, 2], white: [3, 4] });
}

#[test]
fn psd_16bit_layers_with_compatibility_section_open_visible() {
    use photocraft_doc::{Document, Layer, LayerContent, Size};
    use photocraft_geom::Rect;
    use photocraft_raster::Surface;

    // A 16-bit document with multiple distinct user layers.
    let mut doc = Document::new("Test16", Size::new(50, 50), ColorMode::Rgb, SampleType::U16);
    let fmt = doc.pixel_format();

    let bg_data = vec![255u8; 50 * 50 * 8];
    let bg_surf = Surface::from_interleaved(fmt, Rect::new(0, 0, 50, 50), &bg_data);
    let mut bg = Layer::new("Background", LayerContent::Raster(bg_surf));
    bg.visible = true;
    doc.layers.push(bg);

    let fg_data = vec![128u8; 20 * 20 * 8];
    let fg_surf = Surface::from_interleaved(fmt, Rect::new(10, 10, 30, 30), &fg_data);
    let mut fg = Layer::new("Foreground", LayerContent::Raster(fg_surf));
    fg.visible = true;
    doc.layers.push(fg);

    let exported = export(&doc, "test.psd", &ExportOptions::default()).expect("export");
    let bytes = exported.bytes;

    let cmd_len = u32::from_be_bytes(bytes[26..30].try_into().unwrap()) as usize;
    let res_pos = 30 + cmd_len;
    let res_len = u32::from_be_bytes(bytes[res_pos..res_pos + 4].try_into().unwrap()) as usize;
    let lm_pos = res_pos + 4 + res_len;
    let lm_len = u32::from_be_bytes(bytes[lm_pos..lm_pos + 4].try_into().unwrap()) as usize;
    let li_pos = lm_pos + 4;

    // Simulate Photoshop's 16-bit format: an empty/compatibility layer info section (count = 0)
    // before the global Lr16 block (#4928).
    let mut modified = bytes;
    let new_lm_len = (lm_len + 4) as u32;
    modified[lm_pos..lm_pos + 4].copy_from_slice(&new_lm_len.to_be_bytes());
    modified[li_pos..li_pos + 4].copy_from_slice(&4u32.to_be_bytes());
    modified.splice(li_pos + 4..li_pos + 4, [0u8, 0, 0, 0]);

    let imported = import("test.psd", &modified).expect("import").document;

    // Must import the 16-bit layers from Lr16 ("Background" and "Foreground").
    assert_eq!(imported.layers.len(), 2);
    assert_eq!(imported.layers[0].name, "Background");
    assert_eq!(imported.layers[1].name, "Foreground");
    assert!(imported.layers[0].visible);
    assert!(imported.layers[1].visible);
    assert!(imported.layers[0].surface().unwrap().tile_count() > 0);
    assert!(imported.layers[1].surface().unwrap().tile_count() > 0);
}
