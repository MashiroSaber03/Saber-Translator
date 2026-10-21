//! Vector data ⇄ PSD: synthetic round trips of paths, shape layers and vector masks, plus a
//! corpus check (`corpus/psd`, feature `corpus`) that compares our rasterization
//! of every plain shape layer with the pixels Photoshop stored for it.

#[cfg(feature = "corpus")]
use std::path::{Path as FsPath, PathBuf};
#[cfg(feature = "corpus")]
use std::sync::Arc;

use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{
    Document, Fill, Knot, Layer, LayerContent, LineCap, LineJoin, LiveShape, NamedPath, Path, PathOp, ShapeLayer, ShapeStroke, Size, StrokeAlign, Subpath,
    VectorMask,
};
use photocraft_geom::{Point, Rect};
use photocraft_io::*;
#[cfg(feature = "corpus")]
use photocraft_psd::PsdFile;

fn roundtrip(d: &Document) -> Document {
    let r = export(d, "x.psd", &ExportOptions::default()).unwrap();
    import("x.psd", &r.bytes).unwrap().document
}

/// Coordinates on a 64-px grid are exact in PSD's 8.24 fixed point for a 256-px canvas.
fn path() -> Path {
    Path::new(vec![
        Subpath {
            closed: true,
            op: PathOp::Combine,
            knots: vec![
                Knot::corner(16.0, 16.0),
                Knot::smooth(Point::new(128.0, 32.0), Point::new(96.0, 16.0), Point::new(160.0, 48.0)),
                Knot::corner(64.0, 192.0),
            ],
        },
        Subpath::polygon(&[(32.0, 32.0), (64.0, 32.0), (64.0, 64.0)]).with_op(PathOp::Subtract),
        Subpath::polyline(&[(200.0, 10.0), (250.0, 60.0)]).with_op(PathOp::Exclude),
    ])
}

fn doc() -> Document {
    Document::with_background("v", Size::new(256, 256), ColorMode::Rgb, SampleType::U8, Color::WHITE)
}

fn shape_layer(sh: ShapeLayer) -> Layer {
    let mut sh = sh;
    sh.cache = Some(photocraft_vector::render_shape(&sh, photocraft_color::PixelFormat::RGBA8, Rect::new(0, 0, 256, 256)));
    Layer::new("Shape", LayerContent::Shape(sh))
}

#[test]
fn saved_paths_work_path_and_clipping_roundtrip() {
    let mut d = doc();
    d.paths.push(NamedPath { name: "Outline".into(), path: path(), psd_raw: None });
    let mut inv = path();
    inv.inverted = true;
    d.paths.push(NamedPath { name: "Inverted".into(), path: inv, psd_raw: None });
    d.work_path = Some(Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (128.0, 0.0), (128.0, 128.0)])]));
    d.clipping_path = Some(photocraft_doc::ClippingPath { name: "Outline".into(), flatness: 0.0 });
    let b = roundtrip(&d);
    assert_eq!(b.paths.len(), 2);
    assert_eq!(b.paths[0].name, "Outline");
    assert_eq!(b.paths[0].path, d.paths[0].path);
    assert_eq!(b.paths[1].path, d.paths[1].path);
    assert_eq!(b.work_path, d.work_path);
    assert_eq!(b.clipping_path.as_ref().map(|c| c.name.as_str()), Some("Outline"));
    // Unedited paths are written back byte-identical; edited ones are regenerated.
    let raw0 = b.paths[0].psd_raw.clone().unwrap();
    let mut b2 = b.clone();
    b2.paths[1].path.subpaths.pop();
    let c = roundtrip(&b2);
    assert_eq!(c.paths[0].psd_raw.as_deref(), Some(&*raw0));
    assert_eq!(c.paths[1].path, b2.paths[1].path);
    // Removing the clipping path drops the preserved resource.
    let mut b3 = c.clone();
    b3.clipping_path = None;
    assert!(roundtrip(&b3).clipping_path.is_none());
}

#[test]
fn shape_layer_typed_roundtrip() {
    let mut d = doc();
    let sh = ShapeLayer {
        path: path(),
        fill: Some(Fill::Solid(Color::rgb(1.0, 0.0, 0.0))),
        stroke: Some(ShapeStroke {
            width: 4.0,
            paint: Fill::Solid(Color::rgb(0.0, 0.0, 1.0)),
            opacity: 0.5,
            align: StrokeAlign::Inside,
            cap: LineCap::Round,
            join: LineJoin::Round,
            miter_limit: 100.0,
            dashes: vec![2.0, 1.0],
            dash_offset: 0.0,
        }),
        live: None,
        ..Default::default()
    };
    d.layers.push(shape_layer(sh.clone()));
    let mut ell = ShapeLayer {
        path: photocraft_vector::shapes::ellipse(64.0, 64.0, 128.0, 64.0),
        fill: None,
        stroke: Some(ShapeStroke::default()),
        live: Some(LiveShape::Ellipse { rect: [64.0, 64.0, 128.0, 64.0] }),
        ..Default::default()
    };
    ell.path.subpaths[0].op = PathOp::Combine;
    d.layers.push(shape_layer(ell.clone()));
    let b = roundtrip(&d);
    let LayerContent::Shape(s) = &b.layers[1].content else { panic!("not a shape") };
    assert_eq!(s.path, sh.path);
    assert_eq!(s.fill, sh.fill);
    let st = s.stroke.as_ref().unwrap();
    let want = sh.stroke.as_ref().unwrap();
    assert_eq!((st.width, st.align, st.cap, st.join, &st.dashes), (want.width, want.align, want.cap, want.join, &want.dashes));
    assert!((st.opacity - 0.5).abs() < 1e-6);
    assert!(s.cache.is_some());
    let LayerContent::Shape(e) = &b.layers[2].content else { panic!() };
    assert_eq!(e.fill, None);
    assert_eq!(e.live, ell.live);
    // Kappa handles are quantized to PSD's 8.24 fixed point (1/65536 of 256 px).
    let pts = |p: &Path| p.subpaths.iter().flat_map(|s| s.knots.iter().flat_map(|k| [k.anchor, k.in_ctrl, k.out_ctrl])).collect::<Vec<_>>();
    for (a, b) in pts(&e.path).iter().zip(pts(&ell.path)) {
        assert!((a.x - b.x).abs() < 1e-4 && (a.y - b.y).abs() < 1e-4);
    }
    // A second round trip keeps every vector block byte-identical.
    let c = roundtrip(&b);
    for i in 1..3 {
        let keys = |l: &Layer| {
            l.psd_blocks.iter().filter(|(k, _)| matches!(k, b"vmsk" | b"vsms" | b"vstk" | b"vogk" | b"vscg" | b"SoCo")).cloned().collect::<Vec<_>>()
        };
        assert_eq!(keys(&c.layers[i]), keys(&b.layers[i]), "layer {i}");
    }
}

#[test]
fn custom_stroke_styles_roundtrip_through_psd_vstk() {
    for align in [StrokeAlign::Inside, StrokeAlign::Center, StrokeAlign::Outside] {
        for (cap, join) in [(LineCap::Butt, LineJoin::Miter), (LineCap::Round, LineJoin::Round), (LineCap::Square, LineJoin::Bevel)] {
            for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
                let mut d = doc();
                d.depth = depth;
                let want = ShapeStroke {
                    width: 5.0,
                    paint: Fill::Solid(Color::rgb(0.0, 0.0, 1.0)),
                    opacity: 0.5,
                    align,
                    cap,
                    join,
                    miter_limit: 2.0,
                    dashes: vec![0.0, 2.0, 4.0, 1.0],
                    dash_offset: -0.5,
                };
                let sh = ShapeLayer {
                    path: Path::new(vec![Subpath::polygon(&[(32.0, 32.0), (160.0, 32.0), (96.0, 160.0)])]),
                    stroke: Some(want.clone()),
                    ..Default::default()
                };
                d.layers.push(shape_layer(sh.clone()));
                let back = roundtrip(&d);
                let LayerContent::Shape(result) = &back.layers[1].content else { panic!("not a shape") };
                assert_eq!(result.stroke.as_ref().unwrap(), &want);
                let clip = d.bounds();
                let before = photocraft_vector::CompiledShape::new(&sh, 0.01, clip).render_rgba(clip);
                let after = photocraft_vector::CompiledShape::new(result, 0.01, clip).render_rgba(clip);
                assert_eq!(before, after);
            }
        }
    }
}

#[test]
fn vector_mask_roundtrip_and_removal() {
    let mut d = doc();
    let mut l = Layer::raster("pix", d.pixel_format());
    l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 256, 256), &[0.0, 0.0, 0.0, 1.0]);
    let mut vm = VectorMask::new(Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (128.0, 0.0), (128.0, 256.0), (0.0, 256.0)])]));
    vm.linked = false;
    l.vector_mask = Some(vm.clone());
    d.layers.push(l);
    let b = roundtrip(&d);
    assert_eq!(b.layers[1].vector_mask.as_ref(), Some(&vm));
    assert!(b.layers[1].psd_blocks.iter().any(|(k, _)| k == b"vmsk"));
    // The composite honours it (left half black, right half white).
    let f = photocraft_compose::flatten(&b);
    assert!(f.get(10, 10)[0] < 0.01 && f.get(200, 10)[0] > 0.99);
    // Removing the vector mask removes the block.
    let mut c = b.clone();
    c.layers[1].vector_mask = None;
    let e = roundtrip(&c);
    assert!(e.layers[1].vector_mask.is_none());
    assert!(!e.layers[1].psd_blocks.iter().any(|(k, _)| k == b"vmsk" || k == b"vsms"));
    // Groups carry vector masks too.
    let mut g = doc();
    let mut grp = Layer::group("G", vec![Layer::raster("in", g.pixel_format())]);
    grp.vector_mask = Some(vm);
    g.layers.push(grp);
    assert!(roundtrip(&g).layers[1].vector_mask.is_some());
}

#[cfg(feature = "corpus")]
fn collect(dir: &FsPath, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb")) {
            out.push(p);
        }
    }
}

/// Every plain shape layer in the corpus (solid fill, no stroke, no effects, visible fill):
/// our fill coverage vs the alpha of Photoshop's pixels. Reported; asserted loosely.
#[cfg(feature = "corpus")]
#[test]
fn corpus_shape_coverage_matches_photoshop() {
    let root = FsPath::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd");
    assert!(root.is_dir(), "{} is missing: run `cargo xtask corpus --all`", root.display());
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let (mut n, mut good) = (0, 0);
    for p in &files {
        let name = p.strip_prefix(&root).unwrap_or(p).display().to_string();
        let bytes = std::fs::read(p).unwrap_or_default();
        let Ok(file) = PsdFile::from_bytes(&bytes) else { continue };
        let (d, _) = psd_to_document(&file);
        for (_, _, l) in d.walk() {
            let LayerContent::Shape(sh) = &l.content else { continue };
            let Some(cache) = &sh.cache else { continue };
            if sh.stroke.is_some() || !matches!(sh.fill, Some(Fill::Solid(_))) || !l.effects.items.is_empty() || sh.path.is_empty() {
                continue;
            }
            let r =
                cache.content_bounds().union(&photocraft_vector::fill_rasterizer(&sh.path, 0.01).pixel_bounds().unwrap_or(Rect::EMPTY)).intersect(&d.bounds());
            if r.is_empty() {
                continue;
            }
            let ours = photocraft_vector::path_coverage(&sh.path, r);
            let mut theirs = vec![[0.0f32; 4]; ours.len()];
            cache.read_rgba_into(r, &mut theirs);
            let (mut inter, mut uni, mut max) = (0.0f64, 0.0f64, 0.0f32);
            for (a, b) in ours.iter().zip(&theirs) {
                inter += f64::from(a.min(b[3]));
                uni += f64::from(a.max(b[3]));
                max = max.max((a - b[3]).abs());
            }
            let iou = inter / uni.max(1e-9);
            n += 1;
            if iou > 0.97 {
                good += 1;
            }
            eprintln!("{name:<55} {:<20} IoU {iou:.4} max {max:.3} subpaths {}", l.name.chars().take(20).collect::<String>(), sh.path.subpaths.len());
        }
    }
    eprintln!("vector corpus: {good}/{n} plain shape layers with IoU > 0.97");
    assert!(n == 0 || good * 10 >= n * 9, "{good}/{n}");
}

/// (layer name, block key, data).
#[cfg(feature = "corpus")]
type NamedBlock = (String, [u8; 4], Arc<Vec<u8>>);

/// Unedited corpus shapes and vector masks export with byte-identical vector blocks.
#[cfg(feature = "corpus")]
#[test]
fn corpus_vector_blocks_survive_roundtrip() {
    let root = FsPath::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd");
    assert!(root.is_dir(), "{} is missing: run `cargo xtask corpus --all`", root.display());
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let keys: [&[u8; 4]; 6] = [b"vmsk", b"vsms", b"vogk", b"vstk", b"vscg", b"SoCo"];
    let mut checked = 0;
    for p in &files {
        let bytes = std::fs::read(p).unwrap_or_default();
        let Ok(file) = PsdFile::from_bytes(&bytes) else { continue };
        let (d, _) = psd_to_document(&file);
        let Ok(out) = export(&d, "x.psd", &ExportOptions::default()) else { continue };
        let back = PsdFile::from_bytes(&out.bytes).unwrap();
        let blocks = |f: &PsdFile| -> Vec<NamedBlock> {
            f.layers()
                .iter()
                .flat_map(|l| {
                    let n = l.name();
                    l.blocks.iter().filter(|b| keys.contains(&&b.key)).map(move |b| (n.clone(), b.key, Arc::new(b.data.clone())))
                })
                .collect()
        };
        let (a, b) = (blocks(&file), blocks(&back));
        checked += a.len();
        let summary = |v: &[NamedBlock]| v.iter().map(|(n, k, d)| format!("{n}/{}:{}", String::from_utf8_lossy(k), d.len())).collect::<Vec<_>>();
        for x in &a {
            assert!(
                b.contains(x),
                "{}: {}/{} changed or missing; before {:?} after {:?}",
                p.display(),
                x.0,
                String::from_utf8_lossy(&x.1),
                summary(&a),
                summary(&b)
            );
        }
        // Saved paths too.
        for r in file.resources.iter().filter(|r| (2000..=2997).contains(&r.id)) {
            assert!(back.resources.iter().any(|q| q.id == r.id && q.data == r.data), "{}: path resource {}", p.display(), r.id);
        }
    }
    eprintln!("vector corpus: {checked} vector blocks byte-identical after round trip");
}
