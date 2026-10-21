//! PSD (testgen / builder) → Document → PSD structural equality.

use photocraft_doc::LayerContent;
use photocraft_io::*;
use photocraft_psd::testgen;
use photocraft_psd::{ColorMode, Compression, LayerNode, PsdFile, SectionType, Version};

/// Sample of a layer channel in document coordinates (0 outside the rect).
fn sample(f: &PsdFile, li: usize, id: i16, x: i32, y: i32, cache: &mut std::collections::HashMap<(usize, i16), Vec<u8>>) -> Vec<u8> {
    let rec = &f.layers()[li];
    let r = rec.channel_rect(id);
    let bps = usize::from(f.header.depth / 8);
    if x < r.left || x >= r.right || y < r.top || y >= r.bottom || rec.channel(id).is_none() {
        let default = if id == -2 { rec.layer_mask().map_or(0, |m| m.default_color) } else { 0 };
        return if default == 0 || bps != 1 { vec![0; bps] } else { vec![default] };
    }
    let plane = cache.entry((li, id)).or_insert_with(|| rec.decode_channel(id, f.header.depth, f.header.version).unwrap_or_default());
    let w = (r.right - r.left) as usize;
    let i = ((y - r.top) as usize * w + (x - r.left) as usize) * bps;
    plane.get(i..i + bps).map(<[u8]>::to_vec).unwrap_or(vec![0; bps])
}

fn shape(f: &PsdFile, nodes: &[LayerNode]) -> String {
    nodes
        .iter()
        .map(|n| {
            let r = &f.layers()[n.index()];
            let st = r.section_type();
            let kids = if n.is_group() { format!("[{}]", shape(f, n.children())) } else { String::new() };
            format!("{}{}{}", r.name(), if st == SectionType::ClosedFolder { "(c)" } else { "" }, kids)
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn leaves(nodes: &[LayerNode], out: &mut Vec<usize>) {
    for n in nodes {
        out.push(n.index());
        leaves(n.children(), out);
    }
}

fn assert_structurally_equal(a: &PsdFile, b: &PsdFile) {
    assert_eq!(a.header.width, b.header.width);
    assert_eq!(a.header.height, b.header.height);
    assert_eq!(a.header.depth, b.header.depth);
    assert_eq!(a.header.color_mode, b.header.color_mode);
    let (ta, tb) = (a.layer_tree(), b.layer_tree());
    assert_eq!(shape(a, &ta), shape(b, &tb), "tree shape");
    let (mut la, mut lb) = (Vec::new(), Vec::new());
    leaves(&ta, &mut la);
    leaves(&tb, &mut lb);
    let cc = testgen::mode_channels(a.header.color_mode) as i16;
    let (mut ca, mut cb) = (Default::default(), Default::default());
    for (&ia, &ib) in la.iter().zip(&lb) {
        let (ra, rb) = (&a.layers()[ia], &b.layers()[ib]);
        let n = ra.name();
        assert_eq!(n, rb.name());
        let blend = if matches!(ra.blend_mode, photocraft_psd::BlendMode::Unknown(_)) { photocraft_psd::BlendMode::Normal } else { ra.blend_mode };
        if ra.section_type().is_folder() {
            assert_eq!(ra.section_divider().and_then(|s| s.blend_mode), rb.section_divider().and_then(|s| s.blend_mode), "{n}");
        } else {
            assert_eq!(blend, rb.blend_mode, "{n}: blend");
        }
        assert_eq!(ra.opacity, rb.opacity, "{n}: opacity");
        assert_eq!(ra.fill_opacity(), rb.fill_opacity(), "{n}: fill");
        assert_eq!(ra.is_visible(), rb.is_visible(), "{n}: visible");
        assert_eq!(ra.clipping, rb.clipping, "{n}: clipping");
        assert_eq!(ra.block(b"lfx2").map(|x| &x.data), rb.block(b"lfx2").map(|x| &x.data), "{n}: lfx2");
        if ra.section_type().is_folder() {
            continue;
        }
        // Pixels: compare every channel over the union of both rects.
        let u = |r: photocraft_psd::Rect, s: photocraft_psd::Rect| (r.left.min(s.left), r.top.min(s.top), r.right.max(s.right), r.bottom.max(s.bottom));
        let mut ids: Vec<i16> = (-1..cc).collect();
        if ra.layer_mask().is_some() && (ra.channel(-2).is_some() || ra.channel(-3).is_some()) {
            ids.push(-2);
        }
        for id in ids {
            // The exporter writes the real mask (-3) as the user mask (-2).
            let src_id = if id == -2 && ra.channel(-3).is_some() && ra.layer_mask().is_some_and(|m| m.real.is_some()) { -3 } else { id };
            let (x0, y0, x1, y1) = u(ra.channel_rect(src_id), rb.channel_rect(id));
            for y in y0..y1 {
                for x in x0..x1 {
                    let pa = sample(a, ia, src_id, x, y, &mut ca);
                    let pb = sample(b, ib, id, x, y, &mut cb);
                    assert_eq!(pa, pb, "{n}: channel {id} at ({x},{y})");
                }
            }
        }
    }
}

fn via_doc(f: &PsdFile) -> PsdFile {
    let bytes = f.to_bytes().unwrap();
    let imp = import("t.psd", &bytes).unwrap();
    let out = export(&imp.document, "t.psd", &ExportOptions::default()).unwrap();
    PsdFile::from_bytes(&out.bytes).unwrap()
}

macro_rules! tg {
    ($name:ident, $v:expr, $mode:expr, $depth:expr, $c:expr) => {
        #[test]
        fn $name() {
            let f = testgen::layered($v, $mode, $depth, $c);
            let g = via_doc(&f);
            assert_structurally_equal(&f, &g);
        }
    };
}

tg!(testgen_rgb8_rle, Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
tg!(testgen_rgb8_raw, Version::Psd, ColorMode::Rgb, 8, Compression::Raw);
tg!(testgen_rgb16_zip, Version::Psd, ColorMode::Rgb, 16, Compression::Zip);
tg!(testgen_rgb32_zipp, Version::Psd, ColorMode::Rgb, 32, Compression::ZipPrediction);
tg!(testgen_gray8_zipp, Version::Psd, ColorMode::Grayscale, 8, Compression::ZipPrediction);
tg!(testgen_gray16_rle, Version::Psd, ColorMode::Grayscale, 16, Compression::Rle);
tg!(testgen_gray32_raw, Version::Psd, ColorMode::Grayscale, 32, Compression::Raw);
tg!(testgen_cmyk8_rle, Version::Psd, ColorMode::Cmyk, 8, Compression::Rle);
tg!(testgen_cmyk16_zip, Version::Psd, ColorMode::Cmyk, 16, Compression::Zip);
tg!(testgen_lab8_rle, Version::Psd, ColorMode::Lab, 8, Compression::Rle);
tg!(testgen_psb_rgb8_rle, Version::Psb, ColorMode::Rgb, 8, Compression::Rle);
tg!(testgen_psb_cmyk16_zipp, Version::Psb, ColorMode::Cmyk, 16, Compression::ZipPrediction);

#[test]
fn testgen_small_all_compressions() {
    for v in [Version::Psd, Version::Psb] {
        for c in Compression::ALL {
            let f = testgen::small(v, c);
            assert_structurally_equal(&f, &via_doc(&f));
        }
    }
}

#[test]
fn testgen_import_details() {
    let f = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
    let (d, warnings) = psd_to_document(&f);
    // Unknown blend mode warned about.
    assert!(warnings.iter().any(|w| w.contains("unknown blend mode")), "{warnings:?}");
    // Groups: Outer (open) containing Inner (closed).
    let outer = d.layers.iter().find(|l| l.name == "Outer").expect("outer");
    let LayerContent::Group(g) = &outer.content else { panic!() };
    assert!(g.expanded);
    assert_eq!(outer.blend, photocraft_color::BlendMode::PassThrough);
    let inner = g.children.iter().find(|l| l.name == "Inner").unwrap();
    let LayerContent::Group(gi) = &inner.content else { panic!() };
    assert!(!gi.expanded);
    // Negative offsets preserved.
    let neg = d.layers.iter().find(|l| l.name.starts_with("Neg offset")).unwrap();
    let b = neg.surface().unwrap().content_bounds();
    assert!(b.x0 < 0 && b.y0 < 0);
    assert!(neg.mask.is_some());
    assert!(d.icc_profile.is_some());
    assert!((d.resolution_dpi - 72.0).abs() < 1e-3);
    assert!(d.metadata.xmp.is_some());
    // Named saved-path resource (2000) becomes a document path, raw bytes kept for export.
    assert!(d.paths.iter().any(|p| p.name == "Path 1" && p.psd_raw.as_deref() == Some(&vec![1u8, 2, 3])));
    // Global blocks preserved.
    assert!(d.metadata.psd_global_blocks.iter().any(|b| &b.1 == b"Patt"));
    // lfx2 kept on effects.
    assert!(d.walk().iter().any(|(_, _, l)| l.effects.psd_raw.is_some()));
}

#[test]
fn fallback_modes_import_flattened() {
    for mode in [ColorMode::Indexed, ColorMode::Bitmap, ColorMode::Duotone] {
        let depth = testgen::mode_depths(mode)[0];
        let f = testgen::merged_only(Version::Psd, mode, depth, Compression::Rle, 9, 5);
        let (d, w) = psd_to_document(&f);
        assert_eq!(d.layers.len(), 1, "{mode:?}");
        assert!(!w.is_empty(), "{mode:?} should warn");
        assert_eq!(d.depth, photocraft_color::SampleType::U8);
        // Re-export produces a valid PSD.
        let out = export(&d, "x.psd", &ExportOptions::default()).unwrap();
        assert!(PsdFile::from_bytes(&out.bytes).is_ok());
    }
}

#[test]
fn multichannel_imports_ink_channels() {
    for depth in testgen::mode_depths(ColorMode::Multichannel) {
        let f = testgen::merged_only(Version::Psd, ColorMode::Multichannel, *depth, Compression::Rle, 9, 5);
        let (d, _) = psd_to_document(&f);
        assert_eq!(d.mode, photocraft_color::ColorMode::Multichannel);
        assert!(d.layers.is_empty());
        assert_eq!(d.channels.len(), usize::from(f.header.channels));
        assert!(d.channels.iter().all(|c| c.spot.is_some()));
        // Stored dark = ink: the channel value is 1 − the stored sample.
        let merged = f.decode_merged().unwrap();
        let first = if *depth == 8 { f32::from(merged[0]) / 255.0 } else { f32::from(u16::from_be_bytes([merged[0], merged[1]])) / 65535.0 };
        assert!((d.channels[0].surface.pixel(0, 0)[0] - (1.0 - first)).abs() < 1e-3);
        // Re-export writes the same planes.
        let out = export(&d, "x.psd", &ExportOptions::default()).unwrap();
        let back = PsdFile::from_bytes(&out.bytes).unwrap();
        assert_eq!(back.header.color_mode, ColorMode::Multichannel);
        assert_eq!(back.decode_merged().unwrap(), merged, "{depth}");
    }
}

#[test]
fn indexed_colors_match_palette() {
    let f = testgen::merged_only(Version::Psd, ColorMode::Indexed, 8, Compression::Raw, 4, 2);
    let (d, _) = psd_to_document(&f);
    let s = d.layers[0].surface().unwrap();
    let rgba = f.composite_rgba8().unwrap();
    for y in 0..2 {
        for x in 0..4 {
            let p = s.pixel(x, y);
            let i = ((y * 4 + x) * 4) as usize;
            for (c, v) in p.iter().take(3).enumerate() {
                assert_eq!((v * 255.0).round() as u8, rgba.data[i + c]);
            }
        }
    }
}

#[test]
fn flattened_psd_becomes_background() {
    let f = testgen::merged_only(Version::Psd, ColorMode::Rgb, 16, Compression::ZipPrediction, 7, 3);
    let (d, _) = psd_to_document(&f);
    assert_eq!(d.layers.len(), 1);
    assert_eq!(d.layers[0].name, "Background");
    assert_eq!(d.depth, photocraft_color::SampleType::U16);
    let s = d.layers[0].surface().unwrap();
    let merged = f.decode_merged().unwrap();
    let v = u16::from_be_bytes([merged[0], merged[1]]);
    assert_eq!((s.pixel(0, 0)[0] * 65535.0).round() as u16, v);
}

#[test]
fn builder_psd_imports() {
    use photocraft_psd::{GroupSpec, LayerSpec, MaskSpec, PixelData, PsdBuilder, Rect};
    let mut b = PsdBuilder::new(4, 4);
    let mut s = LayerSpec::new("a", 0, 0, 2, 2, PixelData::Rgba8(vec![255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 9, 9, 9, 9]));
    s.mask = Some(MaskSpec { rect: Rect::from_xywh(0, 0, 1, 1), data: vec![100], default_color: 255, disabled: true });
    s.fill_opacity = Some(77);
    b.push_layer(s);
    b.begin_group(GroupSpec { open: false, ..GroupSpec::new("G") });
    b.push_layer(LayerSpec::new("b", 1, 1, 1, 1, PixelData::Rgba8(vec![1, 2, 3, 4])));
    b.end_group().unwrap();
    let bytes = b.to_bytes().unwrap();
    let d = import("b.psd", &bytes).unwrap().document;
    assert_eq!(d.layers.len(), 2);
    let a = &d.layers[0];
    assert_eq!(a.fill_opacity, 77.0 / 255.0);
    let m = a.mask.as_ref().unwrap();
    assert!(!m.enabled);
    assert_eq!(m.surface.pixel(0, 0), vec![100.0 / 255.0]);
    assert_eq!(m.surface.pixel(3, 3), vec![1.0]);
    let s = a.surface().unwrap();
    assert_eq!(s.pixel(1, 0), vec![0.0, 1.0, 0.0, 128.0 / 255.0]);
    assert!(matches!(&d.layers[1].content, LayerContent::Group(g) if !g.expanded && g.children.len() == 1));
}

#[test]
fn group_nesting_past_the_cap_is_rejected() {
    use photocraft_psd::{GroupSpec, LayerSpec, PixelData, PsdBuilder};
    let mut b = PsdBuilder::new(4, 4);
    for _ in 0..101 {
        b.begin_group(GroupSpec::new("g"));
    }
    b.push_layer(LayerSpec::new("deep", 0, 0, 1, 1, PixelData::Rgba8(vec![1, 2, 3, 4])));
    for _ in 0..101 {
        b.end_group().unwrap();
    }
    let bytes = b.to_bytes().unwrap();
    let err = import("deep.psd", &bytes).unwrap_err();
    assert!(err.to_string().contains("nested deeper than 100"), "{err}");
    // The never-fail converter still returns a document — capped, with a warning
    // about the part that was not imported (it used to recurse to the file's depth).
    let (d, warnings) = psd_to_document(&photocraft_psd::PsdFile::from_bytes(&bytes).unwrap());
    assert_eq!(d.max_group_depth(), 100);
    assert!(warnings.iter().any(|w| w.contains("deeper than 100")), "{warnings:?}");
}

#[test]
fn group_nesting_at_the_cap_imports() {
    use photocraft_psd::{GroupSpec, LayerSpec, PixelData, PsdBuilder};
    let mut b = PsdBuilder::new(4, 4);
    for _ in 0..100 {
        b.begin_group(GroupSpec::new("g"));
    }
    b.push_layer(LayerSpec::new("deep", 0, 0, 1, 1, PixelData::Rgba8(vec![1, 2, 3, 4])));
    for _ in 0..100 {
        b.end_group().unwrap();
    }
    let r = import("hundred.psd", &b.to_bytes().unwrap()).unwrap();
    assert_eq!(r.document.max_group_depth(), 100);
    assert!(!r.warnings.iter().any(|w| w.contains("deeper")), "{:?}", r.warnings);
}

#[test]
fn layered_import_skips_the_unused_merged_composite() {
    use photocraft_psd::{LayerSpec, PixelData, PsdBuilder};
    // A layered file whose merged composite carries nothing behind the colour channels never
    // decodes it - that composite is roughly half a Photoshop save's bytes. Here its data is
    // garbage: the layered import does not touch it, while a file with an extra channel does
    // decode (and warn), which is what makes the skip observable.
    let mut b = PsdBuilder::new(4, 4);
    b.push_layer(LayerSpec::new("a", 0, 0, 2, 2, PixelData::Rgba8(vec![7u8; 16])));
    b.composite(PixelData::Rgba8(vec![128; 64]));
    let mut f = photocraft_psd::PsdFile::from_bytes(&b.to_bytes().unwrap()).unwrap();
    f.image_data.data.clear();
    f.layer_info.as_mut().unwrap().merged_alpha = true;
    let (d, warnings) = psd_to_document(&f);
    assert_eq!(d.layers.len(), 1);
    assert!(!warnings.iter().any(|w| w.contains("merged image")), "an unused composite is not decoded: {warnings:?}");
    // A real extra channel behind the colour ones is part of the composite: it is decoded,
    // and the garbage above surfaces as a warning.
    f.layer_info.as_mut().unwrap().merged_alpha = false;
    let (_, warnings) = psd_to_document(&f);
    assert!(warnings.iter().any(|w| w.contains("merged image")), "{warnings:?}");
}

#[test]
fn locks_and_labels_from_psd() {
    use photocraft_psd::TaggedBlock;
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layers_mut()[0].blocks.push(TaggedBlock::protection(0b101));
    f.layers_mut()[0].blocks.push(TaggedBlock::sheet_color(3));
    let (d, _) = psd_to_document(&f);
    let l = &d.layers[0];
    assert!(l.locks.transparency && l.locks.position && !l.locks.pixels);
    assert_eq!(l.label, photocraft_doc::LabelColor::Yellow);
}

#[test]
fn artboard_lock_round_trips_byte_for_byte() {
    // Reference files write a Background layer's locks as lspf = 0x0D (transparency + position +
    // the bit-3 lock). PhotoCraft read the artboard lock at bit 4, so 0x0D came back as 0x05
    // and the lock bytes drifted on every re-save.
    use photocraft_psd::TaggedBlock;
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layers_mut()[0].blocks.push(TaggedBlock::protection(0x0D));
    let g = via_doc(&f);
    let lspf = g.layers().iter().find_map(|l| l.block(b"lspf")).expect("exported layer should carry an lspf block");
    assert_eq!(lspf.data, 0x0Du32.to_be_bytes());
}

#[test]
fn adjustment_and_fill_layers_from_psd() {
    use photocraft_psd::{LayerRecord, TaggedBlock};
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let mk = |name: &str, key: &[u8; 4], data: Vec<u8>| LayerRecord {
        name: name.as_bytes().to_vec(),
        blocks: vec![TaggedBlock::unicode_name(name), TaggedBlock::new(*key, data)],
        ..Default::default()
    };
    let soco = {
        use photocraft_psd::descriptor::*;
        let c = Descriptor::new("RGBC").with("Rd  ", Value::Double(255.0)).with("Grn ", Value::Double(0.0)).with("Bl  ", Value::Double(0.0));
        VersionedDescriptor::new(Descriptor::new("null").with("Clr ", Value::Descriptor(c))).to_bytes()
    };
    f.layers_mut().push(mk("inv", b"nvrt", vec![]));
    f.layers_mut().push(mk("post", b"post", vec![0, 4, 0, 0]));
    f.layers_mut().push(mk("sel", b"selc", vec![1, 2, 3]));
    f.layers_mut().push(mk("red", b"SoCo", soco));
    let (d, _) = psd_to_document(&f);
    let n = d.layers.len();
    use photocraft_doc::{Adjustment, Fill};
    assert!(matches!(d.layers[n - 4].content, LayerContent::Adjustment(Adjustment::Invert)));
    assert!(matches!(d.layers[n - 3].content, LayerContent::Adjustment(Adjustment::Posterize { levels: 4 })));
    assert!(
        matches!(&d.layers[n - 2].content, LayerContent::Adjustment(Adjustment::Unsupported { psd_key, raw }) if psd_key == "selc" && raw == &vec![1, 2, 3])
    );
    assert!(matches!(&d.layers[n - 1].content, LayerContent::Fill(Fill::Solid(c)) if c.c[0] == 1.0));
}

#[test]
fn unmodelled_blocks_are_preserved() {
    use photocraft_psd::TaggedBlock;
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layers_mut()[0].blocks.push(TaggedBlock::new(*b"vmsk", vec![0; 8]));
    let (d, _) = psd_to_document(&f);
    assert!(d.layers[0].psd_blocks.iter().any(|b| &b.0 == b"vmsk"));
    assert!(!d.layers[0].psd_blocks.iter().any(|b| &b.0 == b"luni" || &b.0 == b"lyid"));
    assert_eq!(d.layers[0].psd_id, f.layers()[0].layer_id());
}

#[test]
fn fill_layer_keeps_photoshop_pixels() {
    use photocraft_psd::{ChannelData, LayerRecord, TaggedBlock};
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let soco = {
        use photocraft_psd::descriptor::*;
        let c = Descriptor::new("RGBC").with("Rd  ", Value::Double(255.0)).with("Grn ", Value::Double(0.0)).with("Bl  ", Value::Double(0.0));
        VersionedDescriptor::new(Descriptor::new("null").with("Clr ", Value::Descriptor(c))).to_bytes()
    };
    // Photoshop-rendered pixels deliberately differ from the parametric fill.
    let plane = |v: u8| ChannelData { id: 0, compression: Some(Compression::Raw), data: vec![v; 4] };
    let mut chans = vec![plane(255), plane(10), plane(20), plane(30)];
    for (i, c) in chans.iter_mut().enumerate() {
        c.id = i as i16 - 1;
    }
    f.layers_mut().push(LayerRecord {
        rect: photocraft_psd::Rect::from_xywh(0, 0, 2, 2),
        name: b"fill".to_vec(),
        channels: chans,
        blocks: vec![TaggedBlock::unicode_name("fill"), TaggedBlock::new(*b"SoCo", soco)],
        ..Default::default()
    });
    let (d, _) = psd_to_document(&f);
    let l = d.layers.last().unwrap();
    assert!(matches!(l.content, LayerContent::Fill(_)));
    let fc = l.fill_cache.as_ref().expect("cache");
    assert_eq!(fc.surface.pixel(1, 1), vec![10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0]);
    // The compositor uses the cache while the fill is unchanged.
    let flat = photocraft_compose::flatten(&d);
    let p = flat.get(1, 1);
    assert!((p[0] - 10.0 / 255.0).abs() < 1e-3, "{p:?}");
    // Editing the fill invalidates the cache.
    let mut d2 = d.clone();
    let n = d2.layers.len();
    d2.layers[n - 1].content = LayerContent::Fill(photocraft_doc::Fill::Solid(photocraft_color::Color::rgb(0.0, 0.0, 1.0)));
    let p = photocraft_compose::flatten(&d2).get(1, 1);
    assert!(p[2] > 0.99 && p[0] < 0.01, "{p:?}");
}

#[test]
fn text_and_smart_detected() {
    use photocraft_psd::{LayerRecord, TaggedBlock};
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    for (name, key) in [("t", b"TySh"), ("s", b"PlLd"), ("sh", b"vscg")] {
        f.layers_mut().push(LayerRecord {
            name: name.as_bytes().to_vec(),
            blocks: vec![TaggedBlock::unicode_name(name), TaggedBlock::new(*key, vec![0; 8])],
            ..Default::default()
        });
    }
    let (d, _) = psd_to_document(&f);
    let n = d.layers.len();
    assert_eq!(d.layers[n - 3].content.kind_name(), "Type");
    assert_eq!(d.layers[n - 2].content.kind_name(), "Smart Object");
    assert_eq!(d.layers[n - 1].content.kind_name(), "Shape");
}

#[test]
fn guides_and_alpha_names_roundtrip() {
    let mut d = photocraft_doc::Document::new("g", photocraft_geom::Size::new(8, 8), photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8);
    d.guides.horizontal = vec![1.0, 2.5];
    d.guides.vertical = vec![7.03125];
    for name in ["Spot \u{e9}", "Alpha 2"] {
        d.channels.push(photocraft_doc::AlphaChannel::new(
            name,
            photocraft_raster::Surface::new(photocraft_color::PixelFormat::new(
                photocraft_color::ColorMode::Grayscale,
                photocraft_color::SampleType::U8,
                false,
            )),
        ));
    }
    let f = document_to_psd(&d);
    // RGB + merged transparency (the empty document is transparent) + 2 alpha channels.
    assert_eq!(f.header.channels, 6);
    let (d2, _) = psd_to_document(&f);
    assert_eq!(d2.guides, d.guides);
    let names: Vec<_> = d2.channels.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["Spot \u{e9}", "Alpha 2"]);
}

#[test]
fn resolution_info_cm_display_unit_stays_pixels_per_inch() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let mut ri = photocraft_psd::ResolutionInfo::from_dpi(300.0);
    ri.h_res_unit = 2;
    ri.v_res_unit = 2;
    f.resources[0] = photocraft_psd::ImageResource::new(1005, ri.to_bytes());
    let (d, _) = psd_to_document(&f);
    assert!((d.resolution_dpi - 300.0).abs() < 1e-3, "{}", d.resolution_dpi);
    let out = document_to_psd(&d);
    let written = out.resolution().unwrap();
    assert!((written.h_res() - 300.0).abs() < 1e-3, "{}", written.h_res());
    assert!((written.v_res() - 300.0).abs() < 1e-3, "{}", written.v_res());
}

#[test]
fn stale_index_resources_dropped() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    for id in [1024u16, 1026, 1036, 1057, 1069, 1072] {
        f.resources.push(photocraft_psd::ImageResource::new(id, vec![0, 0]));
    }
    let (d, _) = psd_to_document(&f);
    assert!(d.metadata.psd_resources.iter().all(|r| ![1024u16, 1026, 1036, 1057, 1069, 1072].contains(&r.0)));
}

#[test]
fn merged_composite_unmattes_photoshop_white() {
    // A 1x1 RGB+alpha flattened file with color matted against white.
    let mut f = testgen::merged_only(Version::Psd, ColorMode::Rgb, 8, Compression::Raw, 1, 1);
    f.header.channels = 4;
    // straight red at alpha 0.5, matted: r = 1, g = b = 0.5
    f.image_data = photocraft_psd::ImageData { compression: Compression::Raw, data: vec![255, 128, 128, 128] };
    let m = merged_composite(&f).unwrap();
    assert!((m[0][0] - 1.0).abs() < 0.01 && m[0][1] < 0.01 && m[0][2] < 0.01, "{:?}", m[0]);
    let (d, _) = psd_to_document(&f);
    let p = d.layers[0].surface().unwrap().pixel(0, 0);
    assert!(p[1] < 0.01 && (p[3] - 128.0 / 255.0).abs() < 1e-6, "{p:?}");

    // HDR and negative straight samples must survive the actual F32 PSD import boundary.
    let mut hdr = testgen::merged_only(Version::Psd, ColorMode::Rgb, 32, Compression::Raw, 1, 1);
    hdr.header.channels = 4;
    hdr.image_data =
        photocraft_psd::ImageData { compression: Compression::Raw, data: [1.5_f32, 0.25, 0.5, 0.5].into_iter().flat_map(f32::to_be_bytes).collect() };
    assert_eq!(merged_composite(&hdr).unwrap(), vec![[2.0, -0.5, 0.0, 0.5]]);
    let mut deep = testgen::merged_only(Version::Psd, ColorMode::Rgb, 16, Compression::Raw, 1, 1);
    deep.header.channels = 4;
    deep.image_data.data = [65535_u16, 65535, 65535, 65500].into_iter().flat_map(u16::to_be_bytes).collect();
    assert_eq!(merged_composite(&deep).unwrap()[0][3], 65500.0 / 65535.0);
    let imported = import("hdr.psd", &hdr.to_bytes().unwrap()).unwrap().document;
    assert_eq!(imported.depth, photocraft_color::SampleType::F32);
    assert_eq!(imported.layers[0].surface().unwrap().pixel(0, 0), vec![2.0, -0.5, 0.0, 0.5]);
}

#[test]
fn vector_rendered_mask_not_doubled_on_shapes() {
    use photocraft_psd::{LayerRecord, TaggedBlock};
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let mut mask = photocraft_psd::LayerMask::new(photocraft_psd::Rect::default(), 0, 8);
    mask.flags = 8;
    f.layers_mut().push(LayerRecord {
        name: b"shape".to_vec(),
        mask: photocraft_psd::MaskData::Mask(mask),
        channels: vec![photocraft_psd::ChannelData { id: -2, compression: Some(Compression::Raw), data: vec![] }],
        blocks: vec![TaggedBlock::unicode_name("shape"), TaggedBlock::new(*b"SoCo", vec![]), TaggedBlock::new(*b"vmsk", vec![0; 8])],
        ..Default::default()
    });
    let (d, _) = psd_to_document(&f);
    let l = d.layers.last().unwrap();
    assert_eq!(l.content.kind_name(), "Shape");
    assert!(l.mask.is_none());

    let record = f.layers_mut().last_mut().unwrap();
    let photocraft_psd::MaskData::Mask(mask) = &mut record.mask else { panic!("fixture has a mask") };
    mask.real = Some(photocraft_psd::RealMask { flags: 0, background: 255, rect: photocraft_psd::Rect { top: 0, left: 0, bottom: 1, right: 1 } });
    record.channels.push(photocraft_psd::ChannelData { id: -3, compression: Some(Compression::Raw), data: vec![64] });
    let (imported, _) = psd_to_document(&f);
    assert_eq!(imported.layers.last().unwrap().content.kind_name(), "Shape");
    let mask = imported.layers.last().unwrap().mask.as_ref().expect("selected real mask must survive synthetic cleanup");
    assert!((mask.surface.sample_channel(0, 0, 0) - 64.0 / 255.0).abs() < 1e-6);
    let exported = export(&imported, "real-mask.psd", &Default::default()).unwrap();
    let reopened = import("real-mask.psd", &exported.bytes).unwrap().document;
    assert_eq!(reopened.layers.last().unwrap().mask.as_ref().unwrap().surface.pixel(0, 0), mask.surface.pixel(0, 0));
}

#[test]
fn every_blend_mode_maps_both_ways() {
    for m in std::iter::once(photocraft_color::BlendMode::PassThrough).chain(photocraft_color::BlendMode::LAYER_MODES) {
        let k = photocraft_psd::BlendMode::from_key(m.psd_key());
        assert!(!matches!(k, photocraft_psd::BlendMode::Unknown(_)), "{m:?}");
        assert_eq!(photocraft_color::BlendMode::from_psd_key(k.key()), Some(m));
    }
}

#[test]
fn layer_bounds_without_channel_data_allocate_nothing() {
    // A record can declare a 300000x300000 rectangle in a few bytes. With no channel data to
    // back it, import used to interleave a ~360 GB buffer and abort on the allocation (#755).
    use photocraft_psd::LayerRecord;
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    f.layers_mut().push(LayerRecord { rect: photocraft_psd::Rect::from_xywh(0, 0, 300_000, 300_000), name: b"huge".to_vec(), ..Default::default() });
    let (d, warnings) = psd_to_document(&f);
    let l = d.layers.last().unwrap();
    assert!(matches!(&l.content, LayerContent::Raster(s) if s.tile_count() == 0));
    assert!(warnings.iter().any(|w| w.contains("\"huge\"")), "{warnings:?}");
}
