//! Synthetic documents exercising every part of the model.
#![allow(dead_code)]

use std::sync::Arc;

use photocraft_color::{BlendMode, Color, ColorMode, SampleType};
use photocraft_doc::text::{CharStyle, ParagraphRun, ParagraphStyle, TextRun, TextShape, TextWarp};
use photocraft_doc::*;
use serde_json::json;

pub struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Self {
        Rng(s)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Scribble deterministic pixels into a surface, including negative
/// coordinates and far-away tiles. Float surfaces get HDR values.
pub fn scribble(s: &mut Surface, seed: u64, hdr: bool) {
    let mut r = Rng::new(seed);
    let ch = s.channels();
    for &(x0, y0) in &[(-12, -7), (0, 0), (250, 250), (600, 30)] {
        for dy in 0..9 {
            for dx in 0..11 {
                let px: Vec<f32> = (0..ch).map(|_| if hdr { r.f() * 4.0 - 1.0 } else { r.f() }).collect();
                s.write_pixel(x0 + dx, y0 + dy, &px);
            }
        }
    }
}

pub fn blob(seed: u8, n: usize) -> Arc<Vec<u8>> {
    Arc::new((0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect())
}

pub fn rich_doc(mode: ColorMode, depth: SampleType) -> Document {
    let hdr = depth == SampleType::F32;
    let mut d = Document::with_background("Rich", Size::new(300, 280), mode, depth, Color::WHITE);
    d.resolution_dpi = 300.0;
    d.icc_profile = Some(blob(1, 500));
    let pf = d.pixel_format();

    let mut paint = Layer::raster("Paint", pf);
    scribble(paint.surface_mut().unwrap(), 1, hdr);
    paint.blend = BlendMode::Multiply;
    paint.opacity = 0.75;
    paint.fill_opacity = 0.5;
    paint.clipped = true;
    paint.label = LabelColor::Violet;
    paint.locks.pixels = true;
    let mut mask = LayerMask::hide_all();
    scribble(&mut mask.surface, 2, false);
    mask.density = 0.8;
    mask.feather = 3.5;
    mask.linked = false;
    paint.mask = Some(mask);
    paint.effects.items.push(Effect::ColorOverlay { common: FxCommon::new(BlendMode::Normal, 0.8), color: Color::rgb(1.0, 0.2, 0.1) });
    paint.effects.items.push(Effect::GradientOverlay { common: FxCommon::new(BlendMode::Normal, 0.8), gradient: Gradient::default(), dither: true });
    paint.effects.psd_raw = Some(blob(3, 64));
    paint.psd_blocks.push((*b"vmsk", blob(4, 40)));
    paint.psd_blocks.push((*b"lnsr", blob(5, 4)));
    paint.psd_id = Some(42);

    let mut inner = Layer::raster("Inner", pf);
    scribble(inner.surface_mut().unwrap(), 6, hdr);
    let nested = Layer::group("Nested", vec![inner]);
    let mut adj = Layer::new(
        "Hue/Sat",
        LayerContent::Adjustment(Adjustment::HueSaturation {
            hue: 30.0,
            saturation: -10.0,
            lightness: 5.0,
            colorize: false,
            ranges: photocraft_doc::adjust::HueRange::defaults(),
        }),
    );
    adj.visible = false;
    let mut group = Layer::group("Group", vec![nested, adj]);
    if let LayerContent::Group(g) = &mut group.content {
        g.expanded = false;
    }

    let mut fill = Layer::new("Fill", LayerContent::Fill(Fill::Solid(Color::rgba(0.1, 0.2, 0.3, 0.9))));
    let mut fc = Surface::new(pf);
    scribble(&mut fc, 7, false);
    fill.fill_cache = Some(FillCache { fill: Fill::Solid(Color::rgba(0.1, 0.2, 0.3, 0.9)), surface: fc });
    let grad = Layer::new(
        "Gradient",
        LayerContent::Fill(Fill::Gradient {
            stops: vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
            angle: 45.0,
            scale: 1.5,
            style: GradientStyle::default(),
            reverse: true,
            opacity_stops: vec![(0.0, 1.0), (1.0, 0.25)],
            midpoints: vec![0.3],
            offset: (0.1, -0.2),
            dither: true,
            align: false,
        }),
    );

    let mut cache = Surface::new(pf);
    scribble(&mut cache, 8, false);
    let text = Layer::new(
        "Type",
        LayerContent::Text(TextLayer {
            text: "Hello ✓ world".into(),
            font_family: "Inter".into(),
            size_pt: 24.0,
            color: Color::rgb(0.2, 0.4, 0.6),
            transform: Affine::translate(10.5, 20.25),
            cache: Some(cache.clone()),
            psd_raw: Some(blob(9, 100)),
            runs: vec![
                TextRun { len: 1, style: CharStyle { kerning: text::Kerning::Off, kern: 120.0, ..CharStyle::default() } },
                TextRun { len: 5, style: CharStyle::default() },
                TextRun { len: 9, style: CharStyle { kerning: text::Kerning::Optical, ..CharStyle::default() } },
            ],
            paragraphs: vec![ParagraphRun { len: 15, style: ParagraphStyle::default() }],
            shape: TextShape::Box { x: 1.0, y: 2.0, width: 100.0, height: 50.0 },
            overflow_visible: true,
            block_align: text::BlockAlign::Center,
            orientation: text::Orientation::Vertical,
            antialias: text::AntiAlias::Crisp,
            warp: Some(TextWarp { style: "warpArc".into(), value: 25.0, ..Default::default() }),
        }),
    );
    let mut shape = Layer::new(
        "Shape",
        LayerContent::Shape(ShapeLayer {
            path: vector_path(),
            fill: Some(Fill::Solid(Color::BLACK)),
            stroke: Some(ShapeStroke {
                width: 2.5,
                paint: Fill::Solid(Color::rgb(0.0, 0.5, 1.0)),
                opacity: 0.75,
                align: StrokeAlign::Outside,
                cap: LineCap::Round,
                join: LineJoin::Bevel,
                miter_limit: 4.0,
                dashes: vec![2.0, 1.0],
                dash_offset: 0.5,
            }),
            live: Some(LiveShape::Rect { rect: [1.0, 2.0, 30.0, 40.0], radii: [4.0, 0.0, 4.0, 0.0] }),
            cache: Some(cache.clone()),
            psd_raw: Some(blob(10, 30)),
        }),
    );
    shape.label = LabelColor::Green;
    let mut vm = VectorMask::new(vector_path());
    vm.enabled = false;
    vm.density = 0.4;
    vm.feather = 3.0;
    vm.path.inverted = true;
    paint.vector_mask = Some(vm);
    let smart = Layer::new(
        "Smart",
        LayerContent::Smart(SmartObject {
            contents_id: SmartContentsId::fresh(),
            source: SmartSource::Embedded { file_name: "inner.png".into(), bytes: blob(11, 2000) },
            transform: Affine::scale(0.5),
            smart_filters: vec![SmartFilter {
                command: "filter.blur.gaussian".into(),
                params: json!({"radius": 2.5, "nested": {"a": [1, 2, 3]}}),
                blend: BlendMode::Screen,
                opacity: 0.6,
                visible: true,
            }],
            cache: Some(cache),
            psd_raw: None,
            filters_enabled: false,
            filter_mask: Some(LayerMask {
                surface: Surface::with_default(PixelFormat::GRAY8, &[1.0]),
                enabled: true,
                linked: false,
                density: 0.75,
                feather: 0.0,
            }),
            warp: Some(photocraft_geom::warp::Warp::custom(photocraft_geom::warp::BezierMesh::identity([0.0, 0.0, 48.0, 24.0], 2, 1), [0.0, 0.0, 48.0, 24.0])),
            stack_mode: Some(photocraft_doc::StackMode::Median),
            // Distort / Perspective placement, so the round trip covers it.
            perspective: Some([1.0, 0.1, 2.0, 0.05, 1.0, 3.0, 0.001, 0.0005, 1.0]),
        }),
    );
    let linked = Layer::new(
        "Linked",
        LayerContent::Smart(SmartObject {
            contents_id: SmartContentsId::fresh(),
            source: SmartSource::Linked { path: "/tmp/linked.psd".into() },
            transform: Affine::IDENTITY,
            smart_filters: vec![],
            cache: None,
            psd_raw: Some(blob(12, 8)),
            filters_enabled: true,
            filter_mask: None,
            warp: None,
            stack_mode: None,
            perspective: None,
        }),
    );
    d.layers.extend([paint, group, fill, grad, text, shape, smart, linked]);

    let mut ch = Surface::new(PixelFormat::GRAY8.with_sample(depth));
    scribble(&mut ch, 13, false);
    let mut alpha = AlphaChannel::new("Alpha 1", ch.clone());
    alpha.color = Color::rgb(0.0, 0.5, 1.0);
    alpha.opacity = 0.35;
    alpha.indicates = photocraft_doc::ColorIndicates::SelectedAreas;
    d.channels.push(alpha);
    d.channels.push(AlphaChannel { spot: Some((Color::rgb(1.0, 0.0, 0.5), 0.7)), ..AlphaChannel::new("Spot", ch.clone()) });
    d.quick_mask = Some(AlphaChannel::new("Quick Mask", ch));
    d.quick_mask_unselected = true;
    let mut pat = Surface::new(PixelFormat::new(photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U16, true));
    scribble(&mut pat, 21, false);
    d.patterns.push(photocraft_doc::Pattern::new("$$$/Patterns/Test=Scribble", pat, 32, 24));
    d.color_table = Some(photocraft_doc::ColorTable { colors: vec![[0, 0, 0], [255, 128, 0]], transparent: Some(1) });
    d.duotone = Some(photocraft_doc::Duotone {
        inks: vec![photocraft_doc::DuotoneInk::new("Black", [0.0; 3]), photocraft_doc::DuotoneInk::new("PANTONE 151 C", [1.0, 0.5, 0.0])],
        psd_raw: Some(vec![1, 2, 3]),
    });
    d.guides =
        Guides { horizontal: vec![10.0, 20.5], vertical: vec![100.25], colors: vec![GuideColor { vertical: true, position: 100.25, rgb: [0x33, 0x66, 0xff] }] };
    d.ruler_origin = [12.5, -4.25];
    let mut sel = Surface::new(PixelFormat::GRAY8);
    scribble(&mut sel, 14, false);
    d.selection = Some(sel);
    d.metadata.xmp = Some("<x:xmpmeta/>".into());
    d.metadata.text = vec![("Author".into(), "Synthetic é 雪".into()), ("Comment".into(), "first".into()), ("Comment".into(), "second".into())];
    d.metadata.exif = Some(blob(15, 64));
    d.metadata.psd_resources.push((1036, "thumb".into(), blob(16, 20)));
    d.metadata.psd_global_blocks.push((*b"8BIM", *b"Patt", blob(17, 50)));
    d.global_light = GlobalLight { angle: 120.0, altitude: 30.0 };
    d.paths.push(NamedPath { name: "Path 1".into(), path: vector_path(), psd_raw: Some(blob(18, 52)) });
    d.paths.push(NamedPath { name: "Empty".into(), path: Path::default(), psd_raw: None });
    let mut wp = vector_path();
    wp.fill_rule = FillRule::EvenOdd;
    d.work_path = Some(wp);
    d.clipping_path = Some(ClippingPath { name: "Path 1".into(), flatness: 2.0 });
    d
}

/// A path using every knot kind and path operation.
pub fn vector_path() -> Path {
    Path::new(vec![
        Subpath {
            closed: true,
            op: PathOp::Combine,
            knots: vec![
                Knot::corner(1.5, 2.25),
                Knot::smooth(photocraft_geom::Point::new(40.0, 5.0), photocraft_geom::Point::new(30.0, 0.0), photocraft_geom::Point::new(50.0, 10.0)),
                Knot::corner(10.0, 60.0),
            ],
        },
        Subpath::polygon(&[(5.0, 5.0), (9.0, 5.0), (9.0, 9.0)]).with_op(PathOp::Subtract),
        Subpath::polyline(&[(0.0, 0.0), (3.0, 4.0)]).with_op(PathOp::Exclude),
        Subpath::polygon(&[(0.0, 0.0), (3.0, 4.0), (3.0, 0.0)]).with_op(PathOp::Intersect),
    ])
}

/// A fresh, empty directory under the system temp dir, unique per test process and call.
pub fn temp_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("photocraft-format-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}
