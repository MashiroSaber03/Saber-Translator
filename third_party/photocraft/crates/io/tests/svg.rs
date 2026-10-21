//! SVG import: shapes, groups, transforms, curves, text, gradients, rasterised fallbacks, the
//! size limit, sniffing and malformed input.

use photocraft_color::blend::BlendMode;
use photocraft_doc::vector::{FillRule, LineCap, LineJoin};
use photocraft_doc::{Fill, LayerContent};
use photocraft_io::*;

const NS: &str = "xmlns=\"http://www.w3.org/2000/svg\"";

fn open(svg: &str) -> ImportResult {
    import("drawing.svg", svg.as_bytes()).unwrap_or_else(|e| panic!("{e}\n{svg}"))
}

fn px(doc: &photocraft_doc::Document, x: i32, y: i32) -> [f32; 4] {
    photocraft_compose::flatten(doc).get(x, y)
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.03)
}

#[test]
fn basic_shapes_become_shape_layers_with_their_fills_and_strokes() {
    let r = open(&format!(
        "<svg {NS} width=\"100\" height=\"60\">\
           <rect id=\"box\" x=\"10\" y=\"10\" width=\"30\" height=\"20\" fill=\"#ff0000\"/>\
           <circle cx=\"70\" cy=\"30\" r=\"15\" fill=\"rgb(0,0,255)\" stroke=\"#000\" stroke-width=\"4\" stroke-linecap=\"round\" stroke-linejoin=\"bevel\"/>\
         </svg>"
    ));
    let d = &r.document;
    assert_eq!((d.size.width, d.size.height), (100, 60));
    assert_eq!(d.layers.len(), 2, "{:?}", d.layers.iter().map(|l| &l.name).collect::<Vec<_>>());
    assert_eq!(d.layers[0].name, "box");
    assert_eq!(d.layers[1].name, "Shape 1", "unnamed shapes are numbered");
    let LayerContent::Shape(sh) = &d.layers[0].content else { panic!("not a shape") };
    assert_eq!(sh.fill, Some(Fill::Solid(photocraft_color::Color::rgba(1.0, 0.0, 0.0, 1.0))));
    assert!(sh.stroke.is_none());
    assert_eq!(sh.path.subpaths.len(), 1);
    assert!(sh.path.subpaths[0].closed);
    let LayerContent::Shape(c) = &d.layers[1].content else { panic!("not a shape") };
    let st = c.stroke.as_ref().expect("stroke");
    assert!((st.width - 4.0).abs() < 1e-3);
    assert_eq!((st.cap, st.join), (LineCap::Round, LineJoin::Bevel));
    assert_eq!(st.paint, Fill::Solid(photocraft_color::Color::rgba(0.0, 0.0, 0.0, 1.0)));
    // The composite: red inside the box, blue inside the circle, black on its edge, nothing else.
    assert!(close(px(d, 20, 20), [1.0, 0.0, 0.0, 1.0]), "{:?}", px(d, 20, 20));
    assert!(close(px(d, 70, 30), [0.0, 0.0, 1.0, 1.0]), "{:?}", px(d, 70, 30));
    assert!(close(px(d, 55, 30), [0.0, 0.0, 0.0, 1.0]), "{:?}", px(d, 55, 30));
    assert!(px(d, 3, 3)[3] < 0.01 && px(d, 95, 55)[3] < 0.01);
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
}

#[test]
fn groups_keep_opacity_blend_and_nesting_while_plain_groups_vanish() {
    let r = open(&format!(
        "<svg {NS} width=\"40\" height=\"40\">\
           <g transform=\"translate(5,5)\">\
             <g id=\"art\" opacity=\"0.5\" style=\"mix-blend-mode:multiply\">\
               <rect width=\"10\" height=\"10\" fill=\"#00ff00\"/>\
               <g id=\"inner\"><rect x=\"10\" width=\"10\" height=\"10\" fill=\"#0000ff\"/></g>\
             </g>\
           </g>\
         </svg>"
    ));
    let d = &r.document;
    assert_eq!(d.layers.len(), 1, "the transform-only group is not a layer");
    let art = &d.layers[0];
    assert_eq!(art.name, "art");
    assert!((art.opacity - 0.5).abs() < 1e-6);
    assert_eq!(art.blend, BlendMode::Multiply);
    let LayerContent::Group(g) = &art.content else { panic!("not a group") };
    assert_eq!(g.children.len(), 2);
    assert!(matches!(g.children[0].content, LayerContent::Shape(_)));
    assert_eq!(g.children[1].name, "inner");
    // The translate was baked into the paths: the green square now starts at (5, 5).
    assert!(px(d, 3, 3)[3] < 0.01);
    let p = px(d, 7, 7);
    assert!(p[1] > 0.9 && (p[3] - 0.5).abs() < 0.03, "{p:?}");
}

#[test]
fn svg_group_rasterization_threshold_is_configurable_and_clamped() {
    let source = format!(
        "<svg {NS} width=\"20\" height=\"20\"><g id=\"outer\"><rect width=\"2\" height=\"2\" fill=\"blue\"/><g id=\"inner\"><rect width=\"10\" height=\"10\" fill=\"red\"/><rect x=\"12\" width=\"5\" height=\"5\" fill=\"green\"/></g></g></svg>"
    );
    let shallow = import_with_svg_group_depth("nested.svg", source.as_bytes(), 1).unwrap();
    let LayerContent::Group(outer) = &shallow.document.layers[0].content else { panic!("outer group should remain editable") };
    assert!(outer.children.iter().any(|child| matches!(child.content, LayerContent::Raster(_))), "the subtree past the setting should be rasterized");
    assert!(shallow.warnings.iter().any(|warning| warning.contains("nested too deeply")));

    let unlimited = import_with_svg_group_depth("nested.svg", source.as_bytes(), 0).unwrap();
    let LayerContent::Group(outer) = &unlimited.document.layers[0].content else { panic!("outer group should remain editable") };
    assert!(outer.children.iter().any(|child| matches!(child.content, LayerContent::Group(_))), "zero disables the preference threshold");

    let clamped = import_with_svg_group_depth("nested.svg", source.as_bytes(), usize::MAX).unwrap();
    assert!(matches!(clamped.document.layers[0].content, LayerContent::Group(_)), "out-of-range settings clamp to the document's hard group limit");
}

#[test]
fn transforms_curves_and_fill_rules_are_in_the_knots() {
    let r = open(&format!(
        "<svg {NS} width=\"60\" height=\"60\">\
           <g transform=\"translate(10,20) scale(2)\"><rect width=\"5\" height=\"5\" fill=\"black\"/></g>\
           <path d=\"M30 10 Q 40 0 50 10 L 50 20 Z\" fill=\"black\"/>\
           <path d=\"M0 40 h20 v20 h-20 z M5 45 h10 v10 h-10 z\" fill=\"black\" fill-rule=\"evenodd\"/>\
         </svg>"
    ));
    let d = &r.document;
    let LayerContent::Shape(rect) = &d.layers[0].content else { panic!() };
    let xs: Vec<f64> = rect.path.subpaths[0].knots.iter().map(|k| k.anchor.x).collect();
    let ys: Vec<f64> = rect.path.subpaths[0].knots.iter().map(|k| k.anchor.y).collect();
    assert_eq!((xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max)), (10.0, 20.0));
    assert_eq!((ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max)), (20.0, 30.0));
    let LayerContent::Shape(curve) = &d.layers[1].content else { panic!() };
    let k = &curve.path.subpaths[0].knots;
    assert_eq!(k.len(), 3);
    // The quadratic became a cubic: the first knot's leaving handle moved off its anchor.
    assert!(k[0].out_ctrl != k[0].anchor && k[1].in_ctrl != k[1].anchor);
    assert!(k[1].out_ctrl == k[1].anchor, "the straight segment keeps retracted handles");
    let LayerContent::Shape(ring) = &d.layers[2].content else { panic!() };
    assert_eq!(ring.path.fill_rule, FillRule::EvenOdd);
    assert_eq!(ring.path.subpaths.len(), 2);
    assert!(px(d, 2, 42)[3] > 0.99 && px(d, 10, 50)[3] < 0.01, "even-odd leaves the hole empty");
}

#[test]
fn gradients_are_approximated_with_a_warning() {
    let r = open(&format!(
        "<svg {NS} width=\"100\" height=\"20\">\
           <defs><linearGradient id=\"g\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">\
             <stop offset=\"0\" stop-color=\"#ff0000\"/><stop offset=\"1\" stop-color=\"#0000ff\" stop-opacity=\"0.5\"/>\
           </linearGradient></defs>\
           <rect width=\"100\" height=\"20\" fill=\"url(#g)\"/>\
         </svg>"
    ));
    let LayerContent::Shape(sh) = &r.document.layers[0].content else { panic!() };
    let Some(Fill::Gradient { stops, angle, .. }) = &sh.fill else { panic!("{:?}", sh.fill) };
    assert_eq!(stops.len(), 2);
    assert!((stops[1].1.alpha - 0.5).abs() < 1e-6);
    assert!(angle.abs() < 1e-3, "left to right is 0°, got {angle}");
    assert!(r.warnings.iter().any(|w| w.contains("gradient")), "{:?}", r.warnings);
    // Red on the left, bluish on the right.
    let (l, rr) = (px(&r.document, 2, 10), px(&r.document, 97, 10));
    assert!(l[0] > 0.8 && l[2] < 0.2, "{l:?}");
    assert!(rr[2] > rr[0], "{rr:?}");
}

#[test]
fn clip_paths_masks_filters_patterns_and_images_are_rasterised() {
    // A clip of one path becomes a vector mask (#4266); one of two shapes is still rasterised.
    let r = open(&format!(
        "<svg {NS} width=\"60\" height=\"30\">\
           <defs>\
             <clipPath id=\"c\"><rect width=\"10\" height=\"30\"/><rect x=\"26\" width=\"2\" height=\"30\"/></clipPath>\
             <pattern id=\"p\" width=\"4\" height=\"4\" patternUnits=\"userSpaceOnUse\"><rect width=\"2\" height=\"4\" fill=\"black\"/></pattern>\
           </defs>\
           <g id=\"clipped\" clip-path=\"url(#c)\"><rect width=\"30\" height=\"30\" fill=\"#ff0000\"/></g>\
           <rect id=\"striped\" x=\"30\" width=\"30\" height=\"30\" fill=\"url(#p)\"/>\
         </svg>"
    ));
    let d = &r.document;
    assert_eq!(d.layers.len(), 2);
    assert!(matches!(d.layers[0].content, LayerContent::Raster(_)), "clipped group is pixels");
    assert!(matches!(d.layers[1].content, LayerContent::Raster(_)), "pattern paint is pixels");
    assert!(close(px(d, 5, 15), [1.0, 0.0, 0.0, 1.0]), "{:?}", px(d, 5, 15));
    assert!(px(d, 20, 15)[3] < 0.01, "clipped away");
    // 4 px tiles with a 2 px bar from the pattern origin: x = 32 is a bar, x = 34 a gap.
    assert!(px(d, 32, 15)[3] > 0.99 && px(d, 34, 15)[3] < 0.01, "stripes: {:?} {:?}", px(d, 32, 15), px(d, 34, 15));
    assert!(r.warnings.iter().any(|w| w.contains("clip-path")) && r.warnings.iter().any(|w| w.contains("pattern")), "{:?}", r.warnings);
}

/// A 2×2 PNG (red, green / blue, white) as a data URL, for `<image>`.
fn png_data_url() -> String {
    let img = photocraft_codecs::Image::from_u8(2, 2, photocraft_codecs::ChannelLayout::Rgb, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]).unwrap();
    let bytes = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::from("data:image/png;base64,");
    for chunk in bytes.chunks(3) {
        let n = chunk.len();
        let v = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= n {
                out.push(T[((v >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[test]
fn embedded_raster_images_become_pixel_layers() {
    let r = open(&format!(
        "<svg {NS} width=\"20\" height=\"20\"><image id=\"photo\" x=\"0\" y=\"0\" width=\"20\" height=\"20\" href=\"{}\" style=\"image-rendering:pixelated\"/></svg>",
        png_data_url()
    ));
    let d = &r.document;
    assert_eq!(d.layers.len(), 1);
    assert_eq!(d.layers[0].name, "photo");
    assert!(matches!(d.layers[0].content, LayerContent::Raster(_)));
    assert!(close(px(d, 3, 3), [1.0, 0.0, 0.0, 1.0]), "{:?}", px(d, 3, 3));
    assert!(close(px(d, 16, 16), [1.0, 1.0, 1.0, 1.0]), "{:?}", px(d, 16, 16));
}

#[test]
fn text_becomes_outline_shapes_or_warns_without_a_font() {
    let r = open(&format!(
        "<svg {NS} width=\"120\" height=\"60\"><text x=\"5\" y=\"45\" font-size=\"48\" font-family=\"Arial, DejaVu Sans, Helvetica, sans-serif\" fill=\"black\">II</text></svg>"
    ));
    let d = &r.document;
    if d.layers.is_empty() {
        assert!(r.warnings.iter().any(|w| w.contains("font")), "{:?}", r.warnings);
        return;
    }
    assert_eq!(d.layers[0].name, "II", "the text names its group");
    let LayerContent::Group(g) = &d.layers[0].content else { panic!("{:?}", d.layers[0].content.kind_name()) };
    assert!(g.children.iter().all(|l| matches!(l.content, LayerContent::Shape(_))));
    let dark = (0..120).filter(|&x| px(d, x, 30)[3] > 0.5).count();
    assert!(dark > 4, "glyph strokes cross the middle row: {dark}");
}

#[test]
fn size_comes_from_the_viewbox_and_huge_drawings_shrink_to_fit() {
    let r = open(&format!("<svg {NS} viewBox=\"0 0 300 150\"><rect width=\"300\" height=\"150\"/></svg>"));
    assert_eq!((r.document.size.width, r.document.size.height), (300, 150));
    let r = open(&format!("<svg {NS} viewBox=\"0 0 100000 50000\"><rect x=\"50000\" width=\"50000\" height=\"50000\" fill=\"black\"/></svg>"));
    let d = &r.document;
    assert_eq!((d.size.width, d.size.height), (svg::MAX_SIDE, svg::MAX_SIDE / 2));
    assert!(r.warnings.iter().any(|w| w.contains("scaled")), "{:?}", r.warnings);
    // The right half is still the right half.
    let (w, h) = (d.size.width as i32, d.size.height as i32);
    assert!(px(d, w / 4, h / 2)[3] < 0.01 && px(d, 3 * w / 4, h / 2)[3] > 0.99);
}

fn nested(depth: usize) -> String {
    format!("<svg {NS} width=\"8\" height=\"8\">{}<rect width=\"8\" height=\"8\"/>{}</svg>", "<g>".repeat(depth), "</g>".repeat(depth))
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(bytes).unwrap();
    e.finish().unwrap()
}

/// 8 000 nested groups (70 KB) overflowed the XML parser's recursion on a 2 MB thread stack, an
/// abort nothing can catch. It is an error now, found before the parser runs; the thread here
/// has a small stack so a regression aborts the test instead of passing by luck.
#[test]
fn deeply_nested_markup_is_an_error_not_a_stack_overflow() {
    let deep = nested(8_000);
    let gz = gzip(deep.as_bytes());
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(move || {
            for (name, bytes) in [("deep.svg", deep.as_bytes()), ("deep.svgz", gz.as_slice())] {
                let e = import(name, bytes).err().unwrap_or_else(|| panic!("{name} opened"));
                assert!(matches!(&e, IoError::Svg(m) if m.contains("nested") && m.contains(&svg::MAX_XML_DEPTH.to_string())), "{name}: {e}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
    // Just inside the limit still opens, the empty groups folded away.
    let r = open(&nested(svg::MAX_XML_DEPTH - 1));
    assert_eq!(r.document.layers.len(), 1);
}

/// The depth check reads the markup, not the parser's tree: it must not count `<` in comments,
/// CDATA, processing instructions or the DOCTYPE, nor end a tag at a `>` inside quotes.
#[test]
fn the_depth_check_skips_comments_cdata_doctype_and_quoted_markup() {
    let svg = format!(
        "<?xml version=\"1.0\"?><!-- {g} --><!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\
         <svg {NS} width=\"8\" height=\"8\"><desc><![CDATA[ {g} ]]></desc><?pi {g} ?>\
         <g id='a>b' data-x=\"c/>d\"><rect width=\"8\" height=\"8\" fill=\"#ff0000\"/></g><!-- {g} --></svg>",
        g = "<g>".repeat(svg::MAX_XML_DEPTH + 10)
    );
    let r = open(&svg);
    assert_eq!(r.document.layers.len(), 1);
    assert!(close(px(&r.document, 4, 4), [1.0, 0.0, 0.0, 1.0]));
}

#[test]
fn svgz_is_inflated_with_a_size_cap() {
    let plain = format!("<svg {NS} width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\" fill=\"#0000ff\"/></svg>");
    let r = import("drawing.svgz", &gzip(plain.as_bytes())).unwrap();
    assert_eq!(r.document.layers.len(), 1);
    assert!(close(px(&r.document, 4, 4), [0.0, 0.0, 1.0, 1.0]));
    // A bomb: a few hundred KB of gzip holding more than the cap of whitespace.
    let mut bomb = format!("<svg {NS} width=\"8\" height=\"8\">").into_bytes();
    bomb.resize(usize::try_from(svg::MAX_INFLATED_BYTES).unwrap() + 1024, b' ');
    bomb.extend_from_slice(b"</svg>");
    let gz = gzip(&bomb);
    assert!(gz.len() < 2 << 20, "{} bytes compressed", gz.len());
    let e = import("bomb.svgz", &gz).err().unwrap_or_else(|| panic!("the bomb opened"));
    assert!(matches!(&e, IoError::Svg(m) if m.contains("inflates past")), "{e}");
    // Corrupt gzip is an error too.
    let e = import("bad.svgz", &[0x1f, 0x8b, 0x08, 0x00, 0xff, 0xff]).err().unwrap_or_else(|| panic!("corrupt gzip opened"));
    assert!(matches!(e, IoError::Svg(_)), "{e}");
}

#[test]
fn sniffed_without_an_extension_and_refused_when_malformed() {
    let svg = format!("<?xml version=\"1.0\"?>\n<svg {NS} width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\"/></svg>");
    assert!(svg::is_svg(svg.as_bytes()));
    let r = import("mystery.bin", svg.as_bytes()).unwrap();
    assert_eq!(r.document.layers.len(), 1);
    for bad in ["<svg", "<svg xmlns=\"http://www.w3.org/2000/svg\"><rect", "not xml at all", ""] {
        let e = import("x.svg", bad.as_bytes()).err().unwrap_or_else(|| panic!("{bad:?} opened"));
        assert!(matches!(e, IoError::Svg(_)), "{bad:?}: {e}");
    }
    let empty = open(&format!("<svg {NS} width=\"8\" height=\"8\"/>"));
    assert!(empty.document.layers.is_empty());
    assert!(empty.warnings.iter().any(|w| w.contains("nothing")), "{:?}", empty.warnings);
}

#[test]
fn rasterize_renders_the_whole_drawing_at_a_scale() {
    let svg = format!("<svg {NS} width=\"20\" height=\"10\"><rect x=\"0\" y=\"4\" width=\"20\" height=\"0.5\" fill=\"black\"/></svg>");
    let b = svg::rasterize(svg.as_bytes(), 10.0).unwrap();
    assert_eq!((b.rect.width(), b.rect.height()), (200, 100));
    // The half-unit line is a crisp 5 px band at 10×.
    assert!(b.get(100, 39)[3] < 0.01 && b.get(100, 42)[3] > 0.99 && b.get(100, 45)[3] < 0.01, "{:?}", (b.get(100, 39), b.get(100, 42), b.get(100, 45)));
    let one = svg::rasterize(svg.as_bytes(), 1.0).unwrap();
    assert_eq!((one.rect.width(), one.rect.height()), (20, 10));
    let huge = format!("<svg {NS} width=\"10000\" height=\"10000\"><rect width=\"10\" height=\"10\"/></svg>");
    assert!(svg::rasterize(huge.as_bytes(), 64.0).is_err(), "a render past the pixel cap is an error, not an allocation");
}

#[test]
fn an_svg_round_trips_through_pcraft_as_shapes() {
    let r = open(&format!(
        "<svg {NS} width=\"50\" height=\"50\"><rect id=\"a\" width=\"25\" height=\"25\" fill=\"#123456\"/><circle cx=\"37\" cy=\"37\" r=\"10\" fill=\"#abcdef\"/></svg>"
    ));
    let saved = export(&r.document, "x.pcraft", &ExportOptions::default()).unwrap();
    let back = import("x.pcraft", &saved.bytes).unwrap().document;
    assert_eq!(back.layers.len(), 2);
    assert!(back.layers.iter().all(|l| matches!(l.content, LayerContent::Shape(_))));
    assert!(close(px(&back, 10, 10), px(&r.document, 10, 10)));
}

#[test]
fn image_hrefs_to_local_files_are_never_read() {
    // A real PNG on disk, referenced by absolute path and as a file URL: opening an untrusted
    // drawing must not pull it in (only `data:` URLs resolve).
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("svg-local-image");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("secret.png");
    let img = photocraft_codecs::Image::from_u8(2, 2, photocraft_codecs::ChannelLayout::Rgb, vec![255; 12]).unwrap();
    std::fs::write(&file, photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap()).unwrap();
    let path = file.to_string_lossy().into_owned();
    for href in [path.clone(), format!("file://{path}"), "secret.png".to_string()] {
        let r = open(&format!("<svg {NS} width=\"20\" height=\"20\"><image x=\"0\" y=\"0\" width=\"20\" height=\"20\" href=\"{href}\"/></svg>"));
        let d = &r.document;
        assert!(d.layers.iter().all(|l| !matches!(l.content, LayerContent::Raster(_))), "{href}: {:?}", d.layers.len());
        assert!(px(d, 10, 10)[3] < 0.01, "{href}: {:?}", px(d, 10, 10));
    }
    // The same picture as a data URL still loads.
    assert!(matches!(
        open(&format!("<svg {NS} width=\"20\" height=\"20\"><image width=\"20\" height=\"20\" href=\"{}\"/></svg>", png_data_url())).document.layers[0].content,
        LayerContent::Raster(_)
    ));
}
