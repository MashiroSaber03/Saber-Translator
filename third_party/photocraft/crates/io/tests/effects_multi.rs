//! Multiple instances of an effect (Photoshop CC's several strokes / shadows / overlays) and
//! multi-subpath shape components survive a PSD round trip and render the same.

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::vector::{Path, PathOp, ShapeLayer, Subpath};
use photocraft_doc::*;
use photocraft_geom::{Rect, Size};
use photocraft_io::{ExportOptions, export, import};

fn stroke(size: f32, position: StrokePosition, paint: FxPaint, opacity: f32) -> Effect {
    Effect::Stroke(StrokeFx { common: FxCommon::new(BlendMode::Normal, opacity), size, position, paint })
}

fn roundtrip(doc: &Document) -> Document {
    let r = export(doc, "x.psd", &ExportOptions::default()).expect("export");
    import("x.psd", &r.bytes).expect("import").document
}

fn max_diff(a: &Document, b: &Document) -> f32 {
    let (x, y) = (photocraft_compose::flatten(a).px, photocraft_compose::flatten(b).px);
    assert_eq!(x.len(), y.len());
    x.iter().zip(&y).flat_map(|(p, q)| (0..4).map(move |c| (p[c] * p[3] - q[c] * q[3]).abs())).fold(0.0, f32::max)
}

#[test]
fn several_strokes_and_shadows_roundtrip() {
    let mut d = Document::with_background("t", Size::new(64, 48), ColorMode::Rgb, SampleType::U8, Color::WHITE);
    let mut l = Layer::raster("blob", PixelFormat::RGBA8);
    l.surface_mut().unwrap().fill_rect(Rect::new(20, 14, 44, 34), &[0.8, 0.2, 0.1, 1.0]);
    let g = Gradient { stops: vec![(0.0, Color::rgb(0.0, 0.0, 1.0)), (1.0, Color::rgb(1.0, 1.0, 0.0))], angle: 30.0, ..Gradient::default() };
    l.effects.items = vec![
        stroke(2.0, StrokePosition::Outside, FxPaint::Color(Color::rgb(1.0, 1.0, 1.0)), 1.0),
        stroke(5.0, StrokePosition::Outside, FxPaint::Gradient(g), 0.8),
        stroke(3.0, StrokePosition::Inside, FxPaint::Color(Color::rgb(0.0, 0.5, 0.0)), 1.0),
        stroke(4.0, StrokePosition::Center, FxPaint::Color(Color::rgb(0.0, 0.0, 0.0)), 0.5),
        Effect::default_drop_shadow(),
        Effect::DropShadow(Shadow { distance: 9.0, angle: 45.0, use_global_light: false, ..shadow_of(Effect::default_drop_shadow()) }),
    ];
    d.layers.push(l);
    let back = roundtrip(&d);
    let fx = &back.layers[1].effects.items;
    let strokes: Vec<&StrokeFx> = fx.iter().filter_map(|e| if let Effect::Stroke(s) = e { Some(s) } else { None }).collect();
    assert_eq!(strokes.len(), 4);
    assert_eq!(
        strokes.iter().map(|s| (s.size, s.position)).collect::<Vec<_>>(),
        vec![(2.0, StrokePosition::Outside), (5.0, StrokePosition::Outside), (3.0, StrokePosition::Inside), (4.0, StrokePosition::Center)]
    );
    assert!(matches!(strokes[1].paint, FxPaint::Gradient(_)));
    assert!((strokes[3].common.opacity - 0.5).abs() < 1e-3);
    assert_eq!(fx.iter().filter(|e| matches!(e, Effect::DropShadow(_))).count(), 2);
    let m = max_diff(&d, &back);
    assert!(m <= 1.0 / 255.0 + 1e-5, "render changed by {m}");
}

fn shadow_of(e: Effect) -> Shadow {
    match e {
        Effect::DropShadow(s) => s,
        _ => unreachable!("a drop shadow"),
    }
}

#[test]
fn joined_shape_components_roundtrip() {
    let mut d = Document::with_background("t", Size::new(40, 40), ColorMode::Rgb, SampleType::U8, Color::WHITE);
    let outer = Subpath::polygon(&[(5.0, 5.0), (35.0, 5.0), (35.0, 35.0), (5.0, 35.0)]);
    let hole = Subpath::polygon(&[(12.0, 12.0), (12.0, 28.0), (28.0, 28.0), (28.0, 12.0)]).with_op(PathOp::Join);
    let path = Path::new(vec![outer, hole]);
    let mut sh = ShapeLayer { path, fill: Some(Fill::Solid(Color::rgb(0.2, 0.4, 0.9))), stroke: None, live: None, cache: None, psd_raw: None };
    sh.cache = Some(photocraft_vector::render_shape(&sh, PixelFormat::RGBA8, d.bounds()));
    let mut l = Layer::new("ring", LayerContent::Shape(sh));
    l.effects.items = vec![stroke(2.0, StrokePosition::Outside, FxPaint::Color(Color::rgb(1.0, 0.0, 0.0)), 1.0)];
    d.layers.push(l);
    // The hole is stroked too (the outline is the joined component).
    let px = |doc: &Document, x: i32, y: i32| photocraft_compose::render(doc, Rect::from_xywh(x, y, 1, 1)).px[0];
    assert!(px(&d, 20, 20)[1] > 0.9, "hole shows the background");
    assert!(px(&d, 20, 13)[0] > 0.9 && px(&d, 20, 13)[1] < 0.1, "stroke inside the hole: {:?}", px(&d, 20, 13));
    let back = roundtrip(&d);
    let LayerContent::Shape(sh) = &back.layers[1].content else { panic!("shape layer") };
    assert_eq!(sh.path.subpaths.iter().map(|s| s.op).collect::<Vec<_>>(), vec![PathOp::Combine, PathOp::Join]);
    let m = max_diff(&d, &back);
    assert!(m <= 1.0 / 255.0 + 1e-5, "render changed by {m}");
}
