use super::tiles::{Space, lzf_decompress};
use super::*;
use photocraft_codecs::{ChannelLayout, Format, Image, SampleType as CSample};
use photocraft_color::{ColorMode, SampleType};
use photocraft_format::zip::ZipWriter;
use photocraft_geom::Rect;

const T: usize = 4; // small tiles keep the fixtures short

/// LZF with literal runs only (valid liblzf output).
fn lzf_literal(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for run in data.chunks(32) {
        out.push((run.len() - 1) as u8);
        out.extend_from_slice(run);
    }
    out
}

/// A tiled pixel file from (x, y, interleaved Krita pixels) tiles; odd tiles LZF, even raw.
fn tile_file(pixel_size: usize, tiles: &[(i32, i32, Vec<u8>)]) -> Vec<u8> {
    let mut out = format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE {pixel_size}\nDATA {}\n", tiles.len()).into_bytes();
    for (i, (x, y, px)) in tiles.iter().enumerate() {
        let n = T * T;
        let mut planar = vec![0; n * pixel_size];
        for p in 0..n {
            for k in 0..pixel_size {
                planar[k * n + p] = px[p * pixel_size + k];
            }
        }
        let body: Vec<u8> = if i % 2 == 1 { [vec![1], lzf_literal(&planar)].concat() } else { [vec![0], planar].concat() };
        out.extend(format!("{x},{y},LZF,{}\n", body.len()).into_bytes());
        out.extend(body);
    }
    out
}

fn archive(maindoc: &str, files: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut z = ZipWriter::new();
    z.add("mimetype", MIMETYPE).unwrap();
    z.add("maindoc.xml", maindoc.as_bytes()).unwrap();
    for (name, data) in files {
        z.add(name, data).unwrap();
    }
    z.finish().unwrap()
}

fn maindoc(cs: &str, layers: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE DOC PUBLIC '-//KDE//DTD krita 2.0//EN' 'http://www.calligra.org/DTD/krita-2.0.dtd'>\n\
         <DOC xmlns=\"http://www.calligra.org/DTD/krita\" syntaxVersion=\"2.0\"><IMAGE name=\"img\" width=\"12\" height=\"8\" colorspacename=\"{cs}\" x-res=\"300\" y-res=\"300\"><layers>{layers}</layers></IMAGE></DOC>"
    )
}

/// One tile of BGRA8 pixels: pixel p = (b, g, r, a) from `f(p)`.
fn bgra(f: impl Fn(usize) -> [u8; 4]) -> Vec<u8> {
    (0..T * T).flat_map(f).collect()
}

fn layer<'a>(d: &'a Document, name: &str) -> &'a Layer {
    fn find<'a>(ls: &'a [Layer], name: &str) -> Option<&'a Layer> {
        ls.iter().find_map(|l| if l.name == name { Some(l) } else { l.children().and_then(|c| find(c, name)) })
    }
    find(&d.layers, name).unwrap_or_else(|| panic!("no layer {name}"))
}

#[test]
fn reads_layers_groups_masks_and_properties() {
    let doc = maindoc(
        "RGBA",
        r#"<layer nodetype="grouplayer" name="Through" passthrough="1" compositeop="multiply" visible="1" opacity="255" filename="layer1" x="2" y="1" collapsed="1">
             <layers><layer nodetype="paintlayer" name="Inner" colorspacename="RGBA" filename="layer2" compositeop="screen" opacity="128" x="1" y="0" locked="1"/></layers>
           </layer>
           <layer nodetype="paintlayer" name="Masked" colorspacename="RGBA" filename="layer3" compositeop="diff" visible="0" x="0" y="0">
             <masks><mask nodetype="transparencymask" name="m" filename="mask4" x="0" y="0"/><mask nodetype="selectionmask" filename="sel"/></masks>
           </layer>
           <layer nodetype="shapelayer" name="Vector" filename="layer5"/>
           <layer nodetype="paintlayer" name="Lab layer" colorspacename="LABA" filename="layer6"/>"#,
    );
    let red = bgra(|p| [0, 0, 255, (p * 16) as u8]);
    let files = [
        ("img/layers/layer2", tile_file(4, &[(0, 0, red.clone()), (4, 0, bgra(|_| [10, 20, 30, 255]))])),
        ("img/layers/layer2.defaultpixel", vec![0; 4]),
        ("img/layers/layer3", tile_file(4, &[(-4, 4, bgra(|_| [1, 2, 3, 200]))])),
        ("img/layers/mask4.pixelselection", tile_file(1, &[(-4, 4, (0..T * T).map(|p| (p * 10) as u8).collect())])),
        ("img/layers/mask4.pixelselection.defaultpixel", vec![0]),
        ("img/annotations/icc", b"not really a profile".to_vec()),
    ];
    let r = crate::import("art.kra", &archive(&doc, &files)).unwrap();
    let d = &r.document;
    assert!(r.source_read_only && !r.preview_only);
    assert_eq!((d.size.width, d.size.height, d.mode, d.depth), (12, 8, ColorMode::Rgb, SampleType::U8));
    assert_eq!(d.resolution_dpi, 300.0);
    assert!(d.icc_profile.is_some());
    // Bottom-to-top: the unsupported nodes are left out.
    let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Masked", "Through"]);
    let through = layer(d, "Through");
    assert_eq!(through.blend, BlendMode::PassThrough);
    assert!(matches!(&through.content, LayerContent::Group(g) if !g.expanded));
    let inner = layer(d, "Inner");
    assert_eq!((inner.blend, inner.locks.all), (BlendMode::Screen, true));
    assert!((inner.opacity - 128.0 / 255.0).abs() < 1e-6);
    // The layer's offset (1, 0) moves its tiles; the group's (2, 1) does not, as in Krita.
    let s = inner.surface().unwrap();
    assert_eq!(s.content_bounds(), Rect::from_xywh(1, 0, 8, 4));
    let px = s.to_interleaved(Rect::from_xywh(1, 0, 2, 1));
    assert_eq!(px, [255, 0, 0, 0, 255, 0, 0, 16]);
    assert_eq!(s.to_interleaved(Rect::from_xywh(5, 0, 1, 1)), [30, 20, 10, 255]);
    let masked = layer(d, "Masked");
    assert!(!masked.visible);
    assert_eq!(masked.blend, BlendMode::Difference);
    let mask = masked.mask.as_ref().unwrap();
    assert_eq!(mask.surface.to_interleaved(Rect::from_xywh(-4, 4, 2, 1)), [0, 10]);
    assert_eq!(mask.surface.to_interleaved(Rect::from_xywh(5, 5, 1, 1)), [0], "outside the tiles the default (hide) applies");
    let all = r.warnings.join("\n");
    for needle in ["vector layers", "LABA", "pass-through group's blend mode"] {
        assert!(all.contains(needle), "{needle} not in {all}");
    }
}

#[test]
fn sixteen_bit_and_gray_spaces_convert_exactly() {
    let px16: Vec<u8> = (0..T * T).flat_map(|p| [1000u16 + p as u16, 2000, 3000, 65535].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).collect();
    let doc = maindoc(
        "RGBA16",
        r#"<layer nodetype="paintlayer" name="L" colorspacename="RGBA16" filename="l"/><layer nodetype="paintlayer" name="G" colorspacename="GRAYA" filename="g"/>"#,
    );
    let gray: Vec<u8> = (0..T * T).flat_map(|p| [p as u8 * 10, 255]).collect();
    let r =
        crate::import("a.kra", &archive(&doc, &[("img/layers/l", tile_file(8, &[(0, 0, px16)])), ("img/layers/g", tile_file(2, &[(0, 0, gray)]))])).unwrap();
    let d = &r.document;
    assert_eq!(d.depth, SampleType::U16);
    let s = layer(d, "L").surface().unwrap();
    assert_eq!(s.format(), PixelFormat::RGBA16);
    let got: Vec<u16> = s.to_interleaved(Rect::from_xywh(1, 0, 1, 1)).as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes(*b)).collect();
    // Krita's B, G, R, A becomes R, G, B, A.
    assert_eq!(got, [3000, 2000, 1001, 65535]);
    // A gray layer in an RGB document is converted.
    let g = layer(d, "G").surface().unwrap();
    assert_eq!(g.format(), PixelFormat::RGBA16);
    let v: Vec<u16> = g.to_interleaved(Rect::from_xywh(2, 0, 1, 1)).as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes(*b)).collect();
    assert_eq!(v, [20 * 257, 20 * 257, 20 * 257, 65535]);

    let doc = maindoc("GRAYAU16", r#"<layer nodetype="paintlayer" name="G16" colorspacename="GRAYAU16" filename="g"/>"#);
    let g16: Vec<u8> = (0..T * T).flat_map(|p| [(p as u16 * 4000).to_le_bytes(), 40000u16.to_le_bytes()].concat()).collect();
    let r = crate::import("g.kra", &archive(&doc, &[("img/layers/g", tile_file(4, &[(0, 0, g16)]))])).unwrap();
    assert_eq!((r.document.mode, r.document.depth), (ColorMode::Grayscale, SampleType::U16));
    let s = layer(&r.document, "G16").surface().unwrap();
    let v: Vec<u16> = s.to_interleaved(Rect::from_xywh(3, 0, 1, 1)).as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes(*b)).collect();
    assert_eq!(v, [12000, 40000]);
}

#[test]
fn a_nonzero_default_pixel_fills_untouched_areas() {
    let doc = maindoc("RGBA", r#"<layer nodetype="paintlayer" name="L" colorspacename="RGBA" filename="l"/>"#);
    let files = [("img/layers/l", tile_file(4, &[])), ("img/layers/l.defaultpixel", vec![255, 0, 0, 255])];
    let r = crate::import("a.kra", &archive(&doc, &files)).unwrap();
    let s = layer(&r.document, "L").surface().unwrap();
    assert_eq!(s.to_interleaved(Rect::from_xywh(100, 100, 1, 1)), [0, 0, 255, 255]);
}

fn merged_png() -> Vec<u8> {
    let img = Image::from_raw(12, 8, ChannelLayout::Rgba, CSample::U8, vec![90; 12 * 8 * 4]).unwrap();
    photocraft_codecs::encode(&img, Format::Png, &Default::default()).unwrap()
}

#[test]
fn unsupported_colour_spaces_open_as_the_merged_image() {
    let doc = maindoc("CMYKA", "");
    let r = crate::import("c.kra", &archive(&doc, &[("mergedimage.png", merged_png())])).unwrap();
    assert!(r.preview_only && r.source_read_only);
    assert_eq!(r.document.layers.len(), 1);
    assert!(r.warnings.first().is_some_and(|w| w.contains("CMYKA") && w.contains("merged image")), "{:?}", r.warnings);
    assert!(crate::import("c.kra", &archive(&doc, &[])).is_err(), "no merged image to fall back to");
}

#[test]
fn export_is_refused() {
    let d = Document::new("x", Size::new(2, 2), ColorMode::Rgb, SampleType::U8);
    for name in ["x.kra", "x.KRZ"] {
        assert!(matches!(crate::export(&d, name, &Default::default()), Err(IoError::Unsupported(_))), "{name}");
    }
}

#[test]
fn lzf_decodes_literals_and_overlapping_references() {
    // "abc" then a 6-byte back-reference 3 back: abcabcabc.
    assert_eq!(lzf_decompress(&[2, b'a', b'b', b'c', 0x80, 2], 9).unwrap(), b"abcabcabc");
    // A long back-reference (len 7 + extra) over a single byte: run-length.
    assert_eq!(lzf_decompress(&[0, b'x', 0xe0, 3, 0], 13).unwrap(), vec![b'x'; 13]);
    for bad in [&[0x20, 5][..], &[5, 1, 2], &[0xe0], &[2, 1, 2, 3, 0x20]] {
        assert!(lzf_decompress(bad, 64).is_err(), "{bad:?}");
    }
    assert!(lzf_decompress(&[2, 1, 2, 3], 2).is_err(), "overrun");
    assert!(lzf_decompress(&[2, 1, 2, 3], 4).is_err(), "short");
}

#[test]
fn malformed_files_fail_cleanly() {
    let one = |cs: &str, data: Vec<u8>| {
        archive(&maindoc(cs, r#"<layer nodetype="paintlayer" name="L" colorspacename="RGBA" filename="l"/>"#), &[("img/layers/l", data)])
    };
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("not a zip", b"nope".to_vec()),
        ("no maindoc", {
            let mut z = ZipWriter::new();
            z.add("mimetype", MIMETYPE).unwrap();
            z.finish().unwrap()
        }),
        ("bad xml", archive("<DOC><IMAGE", &[])),
        ("no image", archive("<DOC/>", &[])),
        ("zero size", archive(&maindoc("RGBA", "").replace("width=\"12\"", "width=\"0\""), &[])),
        ("huge", archive(&maindoc("RGBA", "").replace("width=\"12\"", "width=\"300000\"").replace("height=\"8\"", "height=\"300000\""), &[])),
        ("version 1", one("RGBA", b"VERSION 1\n".to_vec())),
        ("no header", one("RGBA", b"garbage".to_vec())),
        ("big tiles", one("RGBA", b"VERSION 2\nTILEWIDTH 99999\nTILEHEIGHT 64\nPIXELSIZE 4\nDATA 0\n".to_vec())),
        ("pixel size", one("RGBA", tile_file(2, &[]))),
        ("too many tiles", one("RGBA", format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE 4\nDATA 999999\n").into_bytes())),
        ("bad tile line", one("RGBA", format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE 4\nDATA 1\n0,0,ZIP,4\n").into_bytes())),
        ("truncated tile", one("RGBA", format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE 4\nDATA 1\n0,0,LZF,400\n\0").into_bytes())),
        ("raw size", one("RGBA", format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE 4\nDATA 1\n0,0,LZF,3\n\0ab").into_bytes())),
        ("flag", one("RGBA", format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE 4\nDATA 1\n0,0,LZF,1\n\x07").into_bytes())),
        ("far tile", one("RGBA", format!("VERSION 2\nTILEWIDTH {T}\nTILEHEIGHT {T}\nPIXELSIZE 4\nDATA 1\n2000000000,0,LZF,1\n\x01").into_bytes())),
        ("bad opacity", archive(&maindoc("RGBA", r#"<layer nodetype="paintlayer" name="L" colorspacename="RGBA" opacity="x"/>"#), &[])),
    ];
    for (what, bytes) in cases {
        assert!(crate::import("x.kra", &bytes).is_err(), "{what}");
    }
    let deep = maindoc("RGBA", &format!("{}{}", r#"<layer nodetype="grouplayer"><layers>"#.repeat(150), "</layers></layer>".repeat(150)));
    assert!(matches!(crate::import("x.kra", &archive(&deep, &[])), Err(IoError::Unsupported(_))));
}

#[test]
fn truncated_and_corrupted_files_never_panic() {
    let doc = maindoc(
        "RGBA",
        r#"<layer nodetype="paintlayer" name="L" colorspacename="RGBA" filename="l"><masks><mask nodetype="transparencymask" filename="m"/></masks></layer>"#,
    );
    let bytes = archive(
        &doc,
        &[
            ("img/layers/l", tile_file(4, &[(0, 0, bgra(|p| [p as u8; 4])), (4, 4, bgra(|_| [9; 4]))])),
            ("img/layers/m.pixelselection", tile_file(1, &[(0, 0, vec![7; T * T])])),
            ("mergedimage.png", merged_png()),
        ],
    );
    assert!(crate::import("ok.kra", &bytes).is_ok());
    for cut in (0..bytes.len()).step_by((bytes.len() / 300).max(1)) {
        let _ = crate::import("x.kra", bytes.get(..cut).unwrap_or_default());
    }
    let mut state = 0x1234_5678_u32;
    for _ in 0..400 {
        let mut b = bytes.clone();
        for _ in 0..3 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let i = state as usize % b.len();
            b[i] ^= (state >> 24) as u8 | 1;
        }
        let _ = crate::import("x.kra", &b);
    }
}

#[test]
fn space_names() {
    assert_eq!(Space::from_name("RGBA"), Some(Space::Rgba8));
    assert_eq!(Space::from_name("RGBAF32"), None);
    assert!(is_kra(&archive("<DOC/>", &[])));
}

#[test]
fn a_png_with_a_misleading_kra_extension_remains_a_png() {
    for name in ["renamed.kra", "renamed.krz"] {
        let r = crate::import(name, &merged_png()).unwrap();
        assert_eq!(r.document.size.width, 12);
        assert!(!r.source_read_only, "a real PNG must not be treated as a Krita source");
    }
}

#[test]
fn the_mimetype_wins_over_the_extension() {
    let bytes = archive(&maindoc("RGBA", ""), &[]);
    assert!(is_kra(&bytes));
    assert!(crate::import("download", &bytes).is_ok_and(|r| r.source_read_only));
}
