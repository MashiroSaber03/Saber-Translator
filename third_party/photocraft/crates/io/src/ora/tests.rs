use super::*;
use photocraft_doc::{Adjustment, Color, Fill, LayerMask};

const CANVAS: Size = Size { width: 70, height: 50 };

fn px(fmt: PixelFormat, rgba: [f32; 4]) -> Vec<u8> {
    let mut out = vec![0; fmt.bytes_per_pixel()];
    photocraft_raster::encode_pixel(&fmt, &rgba, &mut out);
    out
}

/// A raster layer holding a `w`x`h` block at (`x`, `y`) of a colour ramp in `fmt`.
fn block(name: &str, fmt: PixelFormat, x: i32, y: i32, w: u32, h: u32, seed: u32) -> Layer {
    let mut s = Surface::new(fmt);
    let mut data = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let v = |k: u32| ((i * 7 + j * 13 + k * 29 + seed * 31) % 256) as f32 / 255.0;
            data.extend(px(fmt, [v(0), v(1), v(2), 0.25 + 0.75 * v(3)]));
        }
    }
    s.write_interleaved(Rect::from_xywh(x, y, w, h), &data);
    Layer::new(name, LayerContent::Raster(s))
}

fn doc(depth: SampleType) -> Document {
    let fmt = PixelFormat::new(ColorMode::Rgb, depth, true);
    let mut d = Document::new("t.ora", CANVAS, ColorMode::Rgb, depth);
    let mut base = block("Base", fmt, 0, 0, 70, 50, 1);
    base.locks.transparency = true;
    let mut off = block("Off canvas", fmt, -12, 30, 40, 30, 2);
    off.blend = BlendMode::Multiply;
    off.opacity = 0.5;
    let mut hidden = block("Hidden", fmt, 10, 10, 5, 5, 3);
    hidden.visible = false;
    let mut inner = block("Inner screen", fmt, 20, 5, 30, 20, 4);
    inner.blend = BlendMode::Screen;
    let mut group = Layer::group("Isolated", vec![inner, block("Inner plain", fmt, 5, 5, 8, 8, 5)]);
    group.blend = BlendMode::Overlay;
    group.opacity = 0.75;
    let mut through_child = block("Through child", fmt, 40, 20, 25, 25, 6);
    through_child.blend = BlendMode::LinearLight;
    through_child.locks.all = true;
    let through = Layer::group("Pass <&> \"through\"", vec![through_child]);
    d.layers = vec![base, off, hidden, group, through];
    d
}

fn round_trip(d: &Document) -> (Document, Vec<String>, Vec<String>) {
    let out = export(d).unwrap();
    let back = crate::import("back.ora", &out.bytes).unwrap();
    (back.document, out.warnings, back.warnings)
}

/// Layer trees equal in what OpenRaster stores.
fn assert_same(a: &[Layer], b: &[Layer]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.visible, b.visible, "{}", a.name);
        assert_eq!(a.blend, b.blend, "{}", a.name);
        assert!((a.opacity - b.opacity).abs() < 1e-6, "{}", a.name);
        assert_eq!(a.locks.all, b.locks.all, "{}", a.name);
        assert_eq!(a.locks.transparency, b.locks.transparency, "{}", a.name);
        match (&a.content, &b.content) {
            (LayerContent::Raster(x), LayerContent::Raster(y)) => {
                let r = x.content_bounds();
                assert_eq!(r, y.content_bounds(), "{}", a.name);
                assert_eq!(x.to_interleaved(r), y.to_interleaved(r), "{}", a.name);
            }
            (LayerContent::Group(x), LayerContent::Group(y)) => assert_same(&x.children, &y.children),
            _ => panic!("layer kinds differ for {}", a.name),
        }
    }
}

#[test]
fn layers_groups_and_properties_round_trip_at_8_and_16_bit() {
    for depth in [SampleType::U8, SampleType::U16] {
        let d = doc(depth);
        let (back, export_warnings, import_warnings) = round_trip(&d);
        assert!(export_warnings.is_empty(), "{export_warnings:?}");
        assert!(import_warnings.is_empty(), "{import_warnings:?}");
        assert_eq!(back.size, d.size);
        assert_eq!(back.depth, depth);
        assert_same(&d.layers, &back.layers);
        assert_eq!(photocraft_compose::flatten(&d).px, photocraft_compose::flatten(&back).px);
    }
}

#[test]
fn float_documents_save_as_16_bit() {
    let mut d = doc(SampleType::U16);
    d.depth = SampleType::F32;
    let fmt = PixelFormat::RGBA32F;
    d.layers = vec![block("Float", fmt, 0, 0, 10, 10, 9)];
    let (back, warnings, _) = round_trip(&d);
    assert!(warnings.iter().any(|w| w.contains("16-bit")), "{warnings:?}");
    assert_eq!(back.depth, SampleType::U16);
    let (a, b) = (photocraft_compose::flatten(&d), photocraft_compose::flatten(&back));
    let err = a.px.iter().zip(&b.px).flat_map(|(p, q)| (0..4).map(move |c| (p[c] - q[c]).abs())).fold(0.0f32, f32::max);
    assert!(err < 1.0 / 65535.0 + 1e-6, "{err}");
}

#[test]
fn every_blend_mode_is_written_or_reported() {
    for mode in BlendMode::layer_modes() {
        let mut d = Document::new("m.ora", CANVAS, ColorMode::Rgb, SampleType::U8);
        let mut l = block("L", PixelFormat::RGBA8, 0, 0, 4, 4, 0);
        l.blend = mode;
        d.layers.push(l);
        let (back, warnings, import_warnings) = round_trip(&d);
        let got = back.layers.first().map(|l| l.blend);
        if op_name(mode).is_some() {
            assert_eq!(got, Some(mode), "{mode:?}");
            assert!(warnings.is_empty(), "{mode:?}: {warnings:?}");
        } else {
            assert_eq!(got, Some(BlendMode::Normal), "{mode:?}");
            assert!(warnings.iter().any(|w| w.contains("no OpenRaster equivalent")), "{mode:?}");
        }
        // Xor round-trips through nothing; the approximations only warn on import.
        let _ = import_warnings;
    }
}

#[test]
fn masks_are_applied_to_pixels_and_reported() {
    let mut d = Document::new("m.ora", CANVAS, ColorMode::Rgb, SampleType::U8);
    let mut l = block("Masked", PixelFormat::RGBA8, 0, 0, 20, 20, 0);
    let mut mask = LayerMask::hide_all();
    mask.surface.fill_rect(Rect::from_xywh(0, 0, 10, 20), &[1.0]);
    l.mask = Some(mask);
    d.layers.push(l);
    let (back, warnings, _) = round_trip(&d);
    assert!(warnings.iter().any(|w| w.contains("layer masks were applied")), "{warnings:?}");
    let s = back.layers.first().and_then(Layer::surface).unwrap();
    assert_eq!(s.content_bounds(), Rect::from_xywh(0, 0, 10, 20));
    assert_eq!(photocraft_compose::flatten(&d).px, photocraft_compose::flatten(&back).px);
}

#[test]
fn other_layer_kinds_are_rendered_or_left_out() {
    let mut d = Document::new("k.ora", CANVAS, ColorMode::Rgb, SampleType::U8);
    d.layers.push(block("Base", PixelFormat::RGBA8, 0, 0, 70, 50, 0));
    d.layers.push(Layer::new("Solid", LayerContent::Fill(Fill::Solid(Color::rgb(0.2, 0.4, 0.6)))));
    d.layers.push(Layer::new("Invert", LayerContent::Adjustment(Adjustment::Invert)));
    let (back, warnings, _) = round_trip(&d);
    assert_eq!(back.layers.len(), 2);
    assert!(warnings.iter().any(|w| w.contains("fill layers are saved to OpenRaster as pixels")), "{warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("adjustment layers cannot be saved")), "{warnings:?}");
    let solid = back.layers.get(1).and_then(Layer::surface).unwrap();
    assert_eq!(solid.content_bounds(), back.bounds());
}

#[test]
fn grayscale_saves_as_rgb_and_cmyk_is_refused() {
    let mut g = Document::new("g.ora", CANVAS, ColorMode::Grayscale, SampleType::U8);
    g.layers.push(block("Gray", PixelFormat::GRAYA8, 0, 0, 10, 10, 0));
    let (back, warnings, _) = round_trip(&g);
    assert!(warnings.iter().any(|w| w.contains("grayscale")), "{warnings:?}");
    assert_eq!(back.mode, ColorMode::Rgb);
    let c = Document::new("c.ora", CANVAS, ColorMode::Cmyk, SampleType::U8);
    assert!(matches!(export(&c), Err(IoError::Unsupported(_))));
}

#[test]
fn archive_layout_follows_the_specification() {
    let out = export(&doc(SampleType::U8)).unwrap();
    // A stored `mimetype` first, readable at a fixed offset.
    assert_eq!(out.bytes.get(30..38), Some(&b"mimetype"[..]));
    assert_eq!(out.bytes.get(38..54), Some(MIMETYPE));
    assert!(is_ora(&out.bytes));
    let zip = ZipReader::new(&out.bytes).unwrap();
    for entry in ["stack.xml", "mergedimage.png", "Thumbnails/thumbnail.png"] {
        assert!(zip.find(entry).is_some(), "{entry}");
    }
    let thumb = codecs::decode(&zip.read_by_name("Thumbnails/thumbnail.png", 1 << 20).unwrap()).unwrap();
    assert!(thumb.dimensions().0 <= 256 && thumb.dimensions().1 <= 256);
    let merged = codecs::decode(&zip.read_by_name("mergedimage.png", 1 << 20).unwrap()).unwrap();
    assert_eq!(merged.dimensions(), (70, 50));
    let xml = String::from_utf8(zip.read_by_name("stack.xml", 1 << 20).unwrap()).unwrap();
    assert!(xml.contains("name=\"Pass &lt;&amp;&gt; &quot;through&quot;\""), "{xml}");
    assert!(xml.contains("isolation=\"auto\""), "{xml}");
    assert!(xml.contains("composite-op=\"krita:linear light\""), "{xml}");
}

/// An ORA archive from a hand-written stack.xml and named PNG entries.
fn archive(stack: &str, images: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut z = ZipWriter::new();
    z.add("mimetype", MIMETYPE).unwrap();
    z.add("stack.xml", stack.as_bytes()).unwrap();
    for (name, data) in images {
        z.add(name, data).unwrap();
    }
    z.finish().unwrap()
}

fn solid_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let data = rgba.iter().copied().cycle().take((w * h * 4) as usize).collect();
    codecs::encode(&Image::from_raw(w, h, ChannelLayout::Rgba, CSample::U8, data).unwrap(), Format::Png, &Default::default()).unwrap()
}

fn gray16_png(w: u32, h: u32) -> Vec<u8> {
    let data = (0..w * h).flat_map(|i| ((i * 1000) as u16).to_ne_bytes()).collect();
    codecs::encode(&Image::from_raw(w, h, ChannelLayout::Gray, CSample::U16, data).unwrap(), Format::Png, &Default::default()).unwrap()
}

#[test]
fn reads_other_applications_stacks() {
    // The shapes Krita, MyPaint and GIMP write: a root stack with attributes, `auto`
    // isolation, svg:plus, krita: ids, unknown elements, fractional positions.
    let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<image version="0.0.3" w="8" h="6" xres="150" yres="150">
 <stack opacity="1" name="root" isolation="isolate" composite-op="svg:src-over">
  <layer name="Plus" src="data/a.png" x="2.4" y="1" composite-op="svg:plus" opacity="0.5" selected="true"/>
  <stack name="Folder" isolation="auto" composite-op="krita:darker color">
   <layer name="Inside" src="data/b.png" composite-op="krita:hard mix" visibility="hidden" alpha-preserve="true"/>
  </stack>
  <stack name="Through"><layer src="data/b.png"/></stack>
  <text name="Caption">hello</text>
  <layer name="Gray16" src="data/g.png" composite-op="svg:soft-light" edit-locked="true"/>
  <layer name="Strange" src="data/a.png" composite-op="svg:dst-in"/>
  <layer name="Lost" src="data/missing.png"/>
 </stack>
</image>"#;
    let bytes =
        archive(xml, &[("data/a.png", solid_png(3, 2, [255, 0, 0, 255])), ("data/b.png", solid_png(8, 6, [0, 0, 255, 128])), ("data/g.png", gray16_png(4, 4))]);
    // Detected by content, whatever the name says.
    let r = crate::import("no extension", &bytes).unwrap();
    let d = &r.document;
    assert_eq!((d.size.width, d.size.height, d.depth), (8, 6, SampleType::U16));
    assert_eq!(d.resolution_dpi, 150.0);
    let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Lost", "Strange", "Gray16", "Through", "Folder", "Plus"]);
    let get = |n: &str| d.layers.iter().find(|l| l.name == n).unwrap();
    let plus = get("Plus");
    assert_eq!((plus.blend, plus.opacity), (BlendMode::LinearDodge, 0.5));
    assert_eq!(plus.surface().unwrap().content_bounds(), Rect::from_xywh(2, 1, 3, 2));
    let folder = get("Folder");
    assert_eq!(folder.blend, BlendMode::DarkerColor, "a blend mode isolates an auto group");
    let inside = folder.children().unwrap().first().unwrap();
    assert!(!inside.visible && inside.locks.transparency);
    assert_eq!(inside.blend, BlendMode::HardMix);
    assert_eq!(get("Through").blend, BlendMode::PassThrough);
    let gray = get("Gray16");
    assert!(gray.locks.all);
    assert_eq!(gray.blend, BlendMode::SoftLight);
    assert_eq!(gray.surface().unwrap().format(), PixelFormat::RGBA16);
    assert_eq!(get("Strange").blend, BlendMode::Normal);
    assert!(get("Lost").surface().unwrap().content_bounds().is_empty());
    let all = r.warnings.join("\n");
    for needle in ["svg:dst-in", "<text>", "missing from the archive", "krita:hard mix", "svg:soft-light"] {
        assert!(all.contains(needle), "{needle} not in {all}");
    }
}

#[test]
fn preserves_surrounding_whitespace_in_layer_and_group_names() {
    let xml = r#"<image w="2" h="2"><stack>
      <stack name=" Folder "><layer name=" Pixel " src="a.png"/></stack>
      <layer name="  " src="a.png"/>
    </stack></image>"#;
    let bytes = archive(xml, &[("a.png", solid_png(2, 2, [255, 0, 0, 255]))]);
    let result = crate::import("names.ora", &bytes).unwrap();
    let layers = &result.document.layers;

    let folder = layers.iter().find(|layer| layer.name == " Folder ").expect("group name is preserved");
    assert_eq!(folder.children().unwrap()[0].name, " Pixel ");
    assert!(layers.iter().any(|layer| layer.name == "Layer 3"), "blank names keep the fallback");
}

#[test]
fn malformed_archives_fail_cleanly() {
    let png = solid_png(2, 2, [1, 2, 3, 4]);
    let ok = |s: &str| archive(s, &[("a.png", png.clone())]);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("not a zip", b"definitely not a zip archive at all".to_vec()),
        ("no stack", {
            let mut z = ZipWriter::new();
            z.add("mimetype", MIMETYPE).unwrap();
            z.finish().unwrap()
        }),
        ("wrong mimetype", {
            let mut z = ZipWriter::new();
            z.add("mimetype", b"image/png").unwrap();
            z.add("stack.xml", b"<image w='1' h='1'><stack/></image>").unwrap();
            z.finish().unwrap()
        }),
        ("bad xml", ok("<image w='2' h='2'><stack>")),
        ("wrong root", ok("<picture w='2' h='2'><stack/></picture>")),
        ("no root stack", ok("<image w='2' h='2'/>")),
        ("zero width", ok("<image w='0' h='2'><stack/></image>")),
        ("fractional width", ok("<image w='2.5' h='2'><stack/></image>")),
        ("huge canvas", ok("<image w='300000' h='300000'><stack/></image>")),
        ("bad opacity", ok("<image w='2' h='2'><stack><layer src='a.png' opacity='lots'/></stack></image>")),
        ("nan opacity", ok("<image w='2' h='2'><stack><layer src='a.png' opacity='NaN'/></stack></image>")),
        ("far offset", ok("<image w='2' h='2'><stack><layer src='a.png' x='99999999999'/></stack></image>")),
        ("not an image", archive("<image w='2' h='2'><stack><layer src='a.png'/></stack></image>", &[("a.png", b"garbage".to_vec())])),
        ("not utf-8", archive("", &[]).into_iter().chain([]).collect()),
    ];
    for (what, bytes) in cases {
        assert!(crate::import("x.ora", &bytes).is_err(), "{what}");
    }
    let deep = format!("<image w='2' h='2'><stack>{}{}</stack></image>", "<stack>".repeat(200), "</stack>".repeat(200));
    assert!(matches!(crate::import("x.ora", &ok(&deep)), Err(IoError::Unsupported(_))));
}

#[test]
fn layer_and_pixel_budgets_are_enforced() {
    let many = format!("<image w='2' h='2'><stack>{}</stack></image>", "<stack/>".repeat(MAX_NODES + 1));
    assert!(crate::import("x.ora", &archive(&many, &[])).is_err());
    // A tiny PNG claiming a vast size is refused by the decoder limits, not allocated.
    let big = solid_png(1, 1, [0; 4]);
    let mut forged = big.clone();
    if let Some(ihdr) = forged.get_mut(16..24) {
        ihdr.copy_from_slice(&[0, 0, 0x75, 0x30, 0, 0, 0x75, 0x30]);
    }
    let xml = "<image w='2' h='2'><stack><layer src='a.png'/></stack></image>";
    assert!(crate::import("x.ora", &archive(xml, &[("a.png", forged)])).is_err());
}

#[test]
fn truncated_and_corrupted_files_never_panic() {
    let bytes = export(&doc(SampleType::U8)).unwrap().bytes;
    let step = (bytes.len() / 300).max(1);
    for cut in (0..bytes.len()).step_by(step) {
        let _ = crate::import("x.ora", bytes.get(..cut).unwrap_or_default());
    }
    let mut state = 0x9e37_79b9_u32;
    for _ in 0..300 {
        let mut b = bytes.clone();
        for _ in 0..4 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let i = state as usize % b.len();
            if let Some(byte) = b.get_mut(i) {
                *byte ^= (state >> 24) as u8 | 1;
            }
        }
        let _ = crate::import("x.ora", &b);
    }
}

#[test]
fn zip_mimetype_reads_only_a_leading_stored_entry() {
    assert_eq!(zip_mimetype(&archive("<image/>", &[])), Some(MIMETYPE));
    assert_eq!(zip_mimetype(b"PK\x03\x04"), None);
    assert_eq!(zip_mimetype(&[]), None);
    let mut z = ZipWriter::new();
    z.add("stack.xml", b"x").unwrap();
    z.add("mimetype", MIMETYPE).unwrap();
    assert_eq!(zip_mimetype(&z.finish().unwrap()), None);
}

#[test]
fn export_names_route_through_the_crate_entry_point() {
    let d = doc(SampleType::U8);
    let out = crate::export(&d, "picture.ORA", &Default::default()).unwrap();
    assert!(is_ora(&out.bytes));
    let back = crate::import("picture.ora", &out.bytes).unwrap();
    assert!(!back.source_read_only);
}

#[test]
fn normal_group_fill_opacity_survives_openraster_export() {
    let mut d = Document::new("group.ora", Size { width: 8, height: 8 }, ColorMode::Rgb, SampleType::U8);
    let mut surface = Surface::new(PixelFormat::RGBA8);
    surface.write_interleaved(Rect::from_xywh(0, 0, 8, 8), &[255, 0, 0, 255].repeat(64));
    let mut group = Layer::group("Folder", vec![Layer::new("Red", LayerContent::Raster(surface))]);
    group.blend = BlendMode::Normal;
    group.opacity = 1.0;
    group.fill_opacity = 0.5;
    d.layers.push(group);

    let (back, warnings, _) = round_trip(&d);
    assert!(warnings.iter().any(|w| w.contains("fill opacity was combined")), "{warnings:?}");
    let imported_group = back.layers.first().unwrap();
    assert!((imported_group.opacity - 0.5).abs() < 1e-6);
    assert_eq!(imported_group.fill_opacity, 1.0);
    let before = photocraft_compose::flatten(&d);
    let after = photocraft_compose::flatten(&back);
    for (a, b) in before.px.iter().zip(&after.px) {
        assert!((a[3] - b[3]).abs() < 1.0 / 255.0 + 1e-6, "alpha: {a:?} vs {b:?}");
    }
}

#[test]
fn group_opacity_and_fill_are_combined_without_affecting_raster_opacity() {
    let mut d = Document::new("opacity.ora", Size { width: 8, height: 8 }, ColorMode::Rgb, SampleType::U8);
    let mut group = Layer::group("Folder", vec![block("Child", PixelFormat::RGBA8, 0, 0, 8, 8, 0)]);
    group.blend = BlendMode::Normal;
    group.opacity = 0.5;
    group.fill_opacity = 0.5;
    d.layers.push(group);
    let (back, _, _) = round_trip(&d);
    assert!((back.layers[0].opacity - 0.25).abs() < 1e-6);
    assert_eq!(back.layers[0].children().unwrap()[0].opacity, 1.0);
}

/// One document per thing OpenRaster can't hold, each also checked against the exporter.
fn lossy_documents() -> Vec<(&'static str, Document)> {
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
    let with = |f: &dyn Fn(&mut Document)| {
        let mut d = doc(SampleType::U8);
        f(&mut d);
        d
    };
    vec![
        ("adjustment", with(&|d| d.layers.push(Layer::new("Curves", LayerContent::Adjustment(Adjustment::Invert))))),
        ("fill", with(&|d| d.layers.push(Layer::new("Fill", LayerContent::Fill(Fill::Solid(Color::rgba(1.0, 0.0, 0.0, 1.0))))))),
        ("mask", {
            let mut l = block("Masked", fmt, 0, 0, 20, 20, 9);
            l.mask = Some(LayerMask::reveal_all());
            with(&move |d| d.layers.push(l.clone()))
        }),
        ("clipped", with(&|d| d.layers[1].clipped = true)),
        ("fill opacity", with(&|d| d.layers[0].fill_opacity = 0.5)),
        ("float", doc(SampleType::F32)),
        ("profile", with(&|d| d.icc_profile = Some(photocraft_cms::Builtin::DisplayP3.profile().to_bytes()))),
        ("metadata", with(&|d| d.metadata.xmp = Some("<x/>".into()))),
    ]
}

/// The exporter's warnings and [`losses`] are one list, so they can't drift apart.
#[test]
fn losses_are_exactly_what_the_exporter_warns_about() {
    for (what, d) in lossy_documents() {
        let losses = losses(&d);
        assert!(!losses.is_empty(), "{what}: OpenRaster can't hold it");
        assert_eq!(losses, export(&d).unwrap().warnings, "{what}");
    }
}

/// A document of plain pixel layers and groups loses nothing, so its OpenRaster file can stay its
/// own; an sRGB profile isn't a loss, as readers assume sRGB.
#[test]
fn plain_documents_lose_nothing() {
    for depth in [SampleType::U8, SampleType::U16] {
        let mut d = doc(depth);
        assert_eq!(losses(&d), Vec::<String>::new(), "{depth:?}");
        d.icc_profile = Some(photocraft_cms::Builtin::Srgb.profile().to_bytes());
        assert_eq!(losses(&d), Vec::<String>::new(), "{depth:?} tagged sRGB");
        assert_eq!(export(&d).unwrap().warnings, Vec::<String>::new());
    }
    let mut cmyk = doc(SampleType::U8);
    cmyk.mode = ColorMode::Cmyk;
    assert_eq!(losses(&cmyk).len(), 1, "a document OpenRaster can't hold at all");
}
