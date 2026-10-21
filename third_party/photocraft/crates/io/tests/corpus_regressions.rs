//! Synthetic reproductions of failures found by the psd-tools corpus
//! (`cargo xtask corpus --psd-tools`; see `tests/corpus.rs`). The corpus
//! files themselves are never committed: each test rebuilds the relevant
//! structure from our own documents.

use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, Fill, Layer, LayerContent, LayerMask, Path, ShapeLayer, Size, Subpath, VectorMask};
use photocraft_geom::Rect;
use photocraft_io::*;
use photocraft_psd::layer::CHANNEL_REAL_USER_MASK;
use photocraft_psd::{ChannelData, Compression, MaskData, PsdFile, RealMask, Version};

fn doc(depth: SampleType) -> Document {
    Document::with_background("t", Size::new(32, 32), ColorMode::Rgb, depth, Color::WHITE)
}

fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
    Path::new(vec![Subpath::polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])])
}

fn reimport(file: &PsdFile) -> ImportResult {
    import("x.psd", &file.to_bytes().unwrap()).unwrap()
}

/// psd-tools 32bit.psd, 300dpi.psd, transparentbg.psd, vector-mask2.psd: shape layers whose
/// record stores no pixels (empty rect; every 32-bit shape layer) imported as invisible. They
/// are now rendered from their path and fill.
#[test]
fn shape_layer_without_stored_pixels_renders_from_its_path() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut d = doc(depth);
        let sh = ShapeLayer { path: square(8.0, 8.0, 24.0, 24.0), fill: Some(Fill::Solid(Color::BLACK)), ..Default::default() };
        let mut sh2 = sh.clone();
        sh2.cache = Some(photocraft_vector::render_shape(&sh, d.pixel_format(), d.bounds()));
        d.layers.push(Layer::new("Shape 1", LayerContent::Shape(sh2)));
        let mut file = PsdFile::from_bytes(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
        // Drop the shape record's pixels, as Photoshop does for 32-bit documents.
        let bits = file.header.depth;
        let rec = file.layers_mut().last_mut().unwrap();
        assert!(rec.block(b"vmsk").is_some() || rec.block(b"vsms").is_some());
        rec.rect = Default::default();
        for ch in &mut rec.channels {
            *ch = ChannelData::encode(ch.id, Compression::Raw, &[], 0, 0, bits, Version::Psd).unwrap();
        }
        let imp = reimport(&file);
        let f = photocraft_compose::flatten(&imp.document);
        assert!(f.get(16, 16)[0] < 0.01, "{depth:?}: shape interior renders black: {:?}", f.get(16, 16));
        assert!(f.get(2, 2)[0] > 0.99, "{depth:?}: outside stays white");
    }
}

/// psd-tools mask-density-vectormask.psd, mask_parameters.psd: vector mask density and feather
/// were dropped on export (only the user mask's parameters were written).
#[test]
fn vector_mask_density_and_feather_survive_export() {
    for with_user_mask in [false, true] {
        let mut d = doc(SampleType::U8);
        let mut l = Layer::raster("pix", d.pixel_format());
        l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 32, 32), &[0.0, 0.0, 0.0, 1.0]);
        let mut vm = VectorMask::new(square(0.0, 0.0, 16.0, 32.0));
        vm.density = 128.0 / 255.0;
        vm.feather = 1.5;
        l.vector_mask = Some(vm);
        if with_user_mask {
            let mut m = LayerMask::reveal_all();
            m.density = 204.0 / 255.0;
            m.feather = 2.5;
            l.mask = Some(m);
        }
        d.layers.push(l);
        let r = export(&d, "x.psd", &ExportOptions::default()).unwrap();
        assert!(!r.warnings.iter().any(|w| w.contains("density")), "{:?}", r.warnings);
        let b = import("x.psd", &r.bytes).unwrap().document;
        let vm = b.layers[1].vector_mask.as_ref().unwrap();
        assert_eq!((vm.density, vm.feather), (128.0 / 255.0, 1.5), "user mask: {with_user_mask}");
        assert_eq!(b.layers[1].mask.is_some(), with_user_mask);
        if let Some(m) = &b.layers[1].mask {
            assert_eq!((m.density, m.feather), (204.0 / 255.0, 2.5));
        }
    }
}

/// psd-tools layer_mask_data.psd, mask-density-layervectormask.psd: a layer with both a user
/// mask and a vector mask stores the "real" user mask (channel -3) and mask parameters; the
/// real-mask fields precede the parameters, so the -3 channel failed to decode.
#[test]
fn real_user_mask_with_parameters_is_decoded() {
    let mut d = doc(SampleType::U8);
    let mut l = Layer::raster("pix", d.pixel_format());
    l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 32, 32), &[0.0, 0.0, 0.0, 1.0]);
    let mut m = LayerMask::reveal_all();
    m.density = 0.5;
    l.mask = Some(m);
    let mut vm = VectorMask::new(square(0.0, 0.0, 32.0, 32.0));
    vm.density = 0.5;
    l.vector_mask = Some(vm);
    d.layers.push(l);
    let mut file = PsdFile::from_bytes(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    // Add a real user mask hiding the left half (0) inside its rect, revealing elsewhere.
    let rec = file.layers_mut().last_mut().unwrap();
    let MaskData::Mask(pm) = &mut rec.mask else { panic!("mask record") };
    assert!(pm.parameters.is_some());
    let real_rect = photocraft_psd::Rect { top: 0, left: 0, bottom: 32, right: 16 };
    pm.real = Some(RealMask { flags: 0, background: 255, rect: real_rect });
    pm.trailing.clear();
    rec.channels.push(ChannelData::encode(CHANNEL_REAL_USER_MASK, Compression::Rle, &[0u8; 16 * 32], 16, 32, 8, Version::Psd).unwrap());
    let imp = reimport(&file);
    assert!(imp.warnings.iter().all(|w| !w.contains("could not be decoded")), "{:?}", imp.warnings);
    let mask = imp.document.layers[1].mask.as_ref().unwrap();
    // Hidden at density 0.5: 1 - 0.5 · (1 - 0).
    assert!((mask.value(4, 4) - 0.5).abs() < 0.01, "real mask hides the left half: {}", mask.value(4, 4));
    assert_eq!(mask.value(24, 4), 1.0, "and reveals outside its rect");
    assert_eq!(mask.density, 128.0 / 255.0);
    assert_eq!(imp.document.layers[1].vector_mask.as_ref().map(|v| v.density), Some(128.0 / 255.0));
}

/// psd-tools layer_mask_data.psd, mask-parameters-no-real-channel.psd: a shape layer whose
/// vector mask has a density (or feather) shows its fill beyond the path at 1 - density; the
/// stored pixels hold only the shape. It imports as a fill layer with a soft vector mask.
#[test]
fn shape_with_vector_mask_density_imports_as_fill_with_soft_mask() {
    let mut d = doc(SampleType::U8);
    let mut l = Layer::new("Ellipse", LayerContent::Fill(Fill::Solid(Color::BLACK)));
    let mut vm = VectorMask::new(square(8.0, 8.0, 24.0, 24.0));
    vm.density = 0.8;
    l.vector_mask = Some(vm);
    d.layers.push(l);
    let mut file = PsdFile::from_bytes(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    // Photoshop stores the shape alone: replace the record's pixels with the path's coverage.
    let rec = file.layers_mut().last_mut().unwrap();
    assert!(rec.block(b"SoCo").is_some() && rec.block(b"vmsk").is_some());
    rec.rect = photocraft_psd::Rect { top: 8, left: 8, bottom: 24, right: 24 };
    for ch in &mut rec.channels {
        if ch.id >= -1 {
            let v = if ch.id == -1 { 255 } else { 0 };
            *ch = ChannelData::encode(ch.id, Compression::Raw, &[v; 16 * 16], 16, 16, 8, Version::Psd).unwrap();
        }
    }
    let imp = reimport(&file);
    let l = &imp.document.layers[1];
    assert!(matches!(l.content, LayerContent::Fill(_)), "{}", l.content.kind_name());
    assert_eq!(l.vector_mask.as_ref().map(|v| v.density), Some(204.0 / 255.0));
    let f = photocraft_compose::flatten(&imp.document);
    assert!(f.get(16, 16)[0] < 0.01, "inside the path: black");
    assert!((f.get(2, 2)[0] - 0.8).abs() < 0.01, "outside: 20% of the fill shows: {:?}", f.get(2, 2));
}

/// psd-tools adjustment-fillers.psd: a solid-colour shape whose vector path is empty with the
/// initial-fill record set (it covers the canvas) and no stored pixels rendered nothing.
#[test]
fn shape_without_pixels_and_full_initial_fill_covers_the_canvas() {
    let mut d = doc(SampleType::U8);
    let mut path = Path::new(Vec::new());
    path.inverted = true;
    let sh = ShapeLayer { path, fill: Some(Fill::Solid(Color::BLACK)), ..Default::default() };
    let mut sh2 = sh.clone();
    sh2.cache = Some(photocraft_vector::render_shape(&sh, d.pixel_format(), d.bounds()));
    d.layers.push(Layer::new("Color Fill 1", LayerContent::Shape(sh2)));
    let mut file = PsdFile::from_bytes(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    let rec = file.layers_mut().last_mut().unwrap();
    rec.rect = Default::default();
    for ch in &mut rec.channels {
        *ch = ChannelData::encode(ch.id, Compression::Raw, &[], 0, 0, 8, Version::Psd).unwrap();
    }
    let f = photocraft_compose::flatten(&reimport(&file).document);
    assert!(f.get(0, 0)[0] < 0.01 && f.get(31, 31)[0] < 0.01, "covers everything: {:?}", f.get(0, 0));
}
