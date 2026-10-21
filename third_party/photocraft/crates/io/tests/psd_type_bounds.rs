//! Imported type-layer geometry (#148): PSD type layers built the way Photoshop writes them (a
//! `TySh` with the text-space → document transform, `EngineData` with `StyleRun`/`ParagraphRun`
//! sizes in text-space units, `BoxBounds` for paragraph text, `bounds`/`boundingBox` and the
//! trailing integer bounds) must import with the right model, so that
//! - the layer's pixels (Photoshop's cache) keep their full extent: nothing is clipped,
//! - re-rendering the imported model with our engine lands on the same pixels (no transform
//!   scale applied twice or not at all, no 72-vs-document-dpi mix-up, no box height used as a
//!   clip for point text), and
//! - the document composite matches the file's merged composite.
//!
//! Fonts: the bundled Inter (deterministic), so our render is the "Photoshop" render here.

mod common;

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::text::TextShape;
use photocraft_doc::{Document, Layer, LayerContent, Size, TextLayer};
use photocraft_geom::{Affine, Rect};
use photocraft_psd::descriptor::{Descriptor, Id, UnicodeString, Value as D};
use photocraft_raster::Surface;
use photocraft_text::engine_data::Value as E;
use photocraft_text::psd::{TySh, write_tysh};

/// One character run: (UTF-16 length, font size in text-space units, explicit leading).
struct Run {
    len: usize,
    size: f64,
    leading: Option<f64>,
}

fn real(v: f64) -> E {
    E::Real(v)
}

fn style(size: f64, leading: Option<f64>) -> E {
    E::Dict(vec![
        ("Font".into(), E::Int(0)),
        ("FontSize".into(), real(size)),
        ("AutoLeading".into(), E::Bool(leading.is_none())),
        ("Leading".into(), real(leading.unwrap_or(0.0))),
        ("FillColor".into(), E::Dict(vec![("Type".into(), E::Int(1)), ("Values".into(), E::Array(vec![real(1.0), real(0.0), real(0.0), real(0.0)]))])),
    ])
}

/// Photoshop-shaped EngineData for `text` (paragraphs separated by `\r`).
fn engine_data(text: &str, runs: &[Run], box_bounds: Option<[f64; 4]>, font: &str) -> E {
    let ed_text = format!("{text}\r");
    let n16 = ed_text.encode_utf16().count() as i64;
    let mut sruns = Vec::new();
    let mut slens = Vec::new();
    for (i, r) in runs.iter().enumerate() {
        sruns.push(E::Dict(vec![("StyleSheet".into(), E::Dict(vec![("StyleSheetData".into(), style(r.size, r.leading))]))]));
        slens.push(E::Int(r.len as i64 + i64::from(i + 1 == runs.len())));
    }
    let para = E::Dict(vec![("Justification".into(), E::Int(0)), ("AutoLeading".into(), real(1.2))]);
    let mut photoshop = vec![("ShapeType".into(), E::Int(i64::from(box_bounds.is_some())))];
    match box_bounds {
        Some(b) => photoshop.push(("BoxBounds".into(), E::Array(b.iter().map(|v| real(*v)).collect()))),
        None => photoshop.push(("PointBase".into(), E::Array(vec![real(0.0), real(0.0)]))),
    }
    let shape = E::Dict(vec![
        ("ShapeType".into(), E::Int(i64::from(box_bounds.is_some()))),
        ("Cookie".into(), E::Dict(vec![("Photoshop".into(), E::Dict(photoshop))])),
    ]);
    let resources = E::Dict(vec![
        ("FontSet".into(), E::Array(vec![E::Dict(vec![("Name".into(), E::String(font.into())), ("Script".into(), E::Int(0))])])),
        ("StyleSheetSet".into(), E::Array(vec![E::Dict(vec![("Name".into(), E::String("Normal RGB".into())), ("StyleSheetData".into(), style(12.0, None))])])),
        ("ParagraphSheetSet".into(), E::Array(vec![E::Dict(vec![("Name".into(), E::String("Normal RGB".into())), ("Properties".into(), para.clone())])])),
    ]);
    E::Dict(vec![
        (
            "EngineDict".into(),
            E::Dict(vec![
                ("Editor".into(), E::Dict(vec![("Text".into(), E::String(ed_text))])),
                (
                    "ParagraphRun".into(),
                    E::Dict(vec![
                        ("RunArray".into(), E::Array(vec![E::Dict(vec![("ParagraphSheet".into(), E::Dict(vec![("Properties".into(), para)]))])])),
                        ("RunLengthArray".into(), E::Array(vec![E::Int(n16)])),
                    ]),
                ),
                ("StyleRun".into(), E::Dict(vec![("RunArray".into(), E::Array(sruns)), ("RunLengthArray".into(), E::Array(slens))])),
                ("Rendered".into(), E::Dict(vec![("Shapes".into(), E::Dict(vec![("Children".into(), E::Array(vec![shape]))]))])),
            ]),
        ),
        ("ResourceDict".into(), resources),
    ])
}

/// A `TySh` block as Photoshop writes it.
fn tysh(text: &str, runs: &[Run], box_bounds: Option<[f64; 4]>, transform: Affine, ink: [f64; 4]) -> Vec<u8> {
    tysh_with_font(text, runs, box_bounds, transform, ink, "Inter-Regular")
}

/// A `TySh` block whose runs use the font with PostScript name `font`.
fn tysh_with_font(text: &str, runs: &[Run], box_bounds: Option<[f64; 4]>, transform: Affine, ink: [f64; 4], font: &str) -> Vec<u8> {
    let rect = |cls: &str| {
        D::Descriptor(
            Descriptor::new(cls)
                .with("Left", D::UnitFloat { unit: *b"#Pnt", value: ink[0] })
                .with("Top ", D::UnitFloat { unit: *b"#Pnt", value: ink[1] })
                .with("Rght", D::UnitFloat { unit: *b"#Pnt", value: ink[2] })
                .with("Btom", D::UnitFloat { unit: *b"#Pnt", value: ink[3] }),
        )
    };
    let desc = Descriptor::new("TxLr")
        .with("Txt ", D::Text(UnicodeString::new_nul(text)))
        .with("textGridding", D::Enumerated { type_id: Id::new("textGridding"), value: Id::new("None") })
        .with("Ornt", D::Enumerated { type_id: Id::new("Ornt"), value: Id::new("Hrzn") })
        .with("AntA", D::Enumerated { type_id: Id::new("Annt"), value: Id::new("AnSm") })
        .with("bounds", rect("bounds"))
        .with("boundingBox", rect("boundingBox"))
        .with("TextIndex", D::Integer(0))
        .with("EngineData", D::RawData(photocraft_text::engine_data::write(&engine_data(text, runs, box_bounds, font))));
    // Unwarped type: Photoshop writes four zero f32s after the warp descriptor.
    write_tysh(&TySh { transform, text: desc, warp: None, bounds: [0.0; 4] })
}

struct Case {
    name: &'static str,
    dpi: f32,
    text: &'static str,
    runs: Vec<Run>,
    box_bounds: Option<[f64; 4]>,
    transform: Affine,
    /// Expected Character-panel size of the first run, in points.
    size_pt: f32,
}

fn cases() -> Vec<Case> {
    let r = |len: usize, size: f64| Run { len, size, leading: None };
    let t = |x: f64, y: f64| Affine::translate(x, y);
    // Photoshop keeps the font size in text space and the scale in the transform.
    let scaled = |s: f64, x: f64, y: f64| Affine { m: [s, 0.0, 0.0, s, x, y] };
    let rotated = |deg: f64, s: f64, x: f64, y: f64| {
        let (sin, cos) = deg.to_radians().sin_cos();
        Affine { m: [s * cos, s * sin, -s * sin, s * cos, x, y] }
    };
    vec![
        Case {
            name: "point 36pt @72",
            dpi: 72.0,
            text: "Headline 105: Autumn collection",
            runs: vec![r(31, 36.0)],
            box_bounds: None,
            transform: t(40.0, 80.0),
            size_pt: 36.0,
        },
        Case {
            name: "point 36pt @300",
            dpi: 300.0,
            text: "Headline 105: Autumn collection",
            runs: vec![r(31, 150.0)],
            box_bounds: None,
            transform: t(40.0, 200.0),
            size_pt: 36.0,
        },
        Case {
            name: "point 12×3 scaled",
            dpi: 72.0,
            text: "Headline 105: Autumn collection",
            runs: vec![r(31, 12.0)],
            box_bounds: None,
            transform: scaled(3.0, 40.0, 80.0),
            size_pt: 12.0,
        },
        Case {
            name: "point rotated",
            dpi: 72.0,
            text: "Autumn collection",
            runs: vec![r(17, 18.0)],
            box_bounds: None,
            transform: rotated(-20.0, 2.0, 60.0, 300.0),
            size_pt: 18.0,
        },
        Case {
            name: "point multi-line",
            dpi: 72.0,
            text: "Headline\rAutumn collection\rgy",
            runs: vec![Run { len: 29, size: 36.0, leading: Some(50.0) }],
            box_bounds: None,
            transform: t(40.0, 60.0),
            size_pt: 36.0,
        },
        Case {
            name: "point mixed sizes",
            dpi: 72.0,
            text: "Big small Big",
            runs: vec![r(4, 60.0), r(6, 14.0), r(3, 60.0)],
            box_bounds: None,
            transform: t(30.0, 120.0),
            size_pt: 60.0,
        },
        Case {
            name: "paragraph box",
            dpi: 72.0,
            text: "Paragraph text that wraps over several lines in its box",
            runs: vec![r(55, 24.0)],
            box_bounds: Some([0.0, 0.0, 260.0, 200.0]),
            transform: t(20.0, 20.0),
            size_pt: 24.0,
        },
        Case {
            name: "paragraph box scaled",
            dpi: 144.0,
            text: "Paragraph text that wraps over several lines in its box",
            runs: vec![r(55, 20.0)],
            box_bounds: Some([0.0, 0.0, 200.0, 150.0]),
            transform: scaled(1.5, 20.0, 20.0),
            size_pt: 10.0,
        },
    ]
}

fn ink_alpha(s: &Surface, r: Rect) -> Vec<f32> {
    let n = s.channels();
    s.read_region(r).chunks_exact(n).map(|p| p[n - 1]).collect()
}

fn iou(a: &Surface, b: &Surface) -> f32 {
    let u = a.content_bounds().union(&b.content_bounds());
    let (x, y) = (ink_alpha(a, u), ink_alpha(b, u));
    let inter: f32 = x.iter().zip(&y).map(|(p, q)| p.min(*q)).sum();
    let union: f32 = x.iter().zip(&y).map(|(p, q)| p.max(*q)).sum();
    if union > 0.0 { inter / union } else { 1.0 }
}

fn close(a: Rect, b: Rect, tol: i32) -> bool {
    (a.x0 - b.x0).abs() <= tol && (a.y0 - b.y0).abs() <= tol && (a.x1 - b.x1).abs() <= tol && (a.y1 - b.y1).abs() <= tol
}

/// Builds a one-layer document whose type layer carries `data` as its `TySh` and "Photoshop's"
/// pixels (our render of the parsed model), exports it to PSD and imports it back.
fn import_case(c: &Case) -> (Rect, TextLayer, Document, Vec<[f32; 4]>) {
    let (w, h) = (900, 700);
    let mut doc = Document::new("t", Size::new(w, h), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = c.dpi;
    // Photoshop's text `bounds` are the logical bounds in text space; measure them like it does.
    let probe = photocraft_text::psd::text_layer_from_tysh(&tysh(c.text, &c.runs, c.box_bounds, c.transform, [0.0; 4]), c.dpi).unwrap();
    let mut engine = photocraft_text::TextEngine::new();
    let ink = engine.layout(&probe, c.dpi).bounds().unwrap_or([0.0; 4]).map(f64::from);
    let data = tysh(c.text, &c.runs, c.box_bounds, c.transform, ink);
    let mut t = photocraft_text::psd::text_layer_from_tysh(&data, c.dpi).unwrap();
    engine.render_layer(&mut t, c.dpi, doc.pixel_format());
    let drawn = t.cache.as_ref().unwrap().content_bounds();
    t.psd_raw = Some(std::sync::Arc::new(data));
    doc.layers.push(Layer::new(c.name, LayerContent::Text(t)));
    let bytes = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap().bytes;
    let file = photocraft_psd::PsdFile::from_bytes(&bytes).unwrap();
    let merged = photocraft_io::merged_composite(&file).unwrap();
    let back = photocraft_io::import("t.psd", &bytes).unwrap().document;
    let LayerContent::Text(bt) = &back.layers[0].content else { panic!("{}: not a type layer", c.name) };
    (drawn, bt.clone(), back, merged)
}

#[test]
fn imported_type_layers_keep_their_full_bounds() {
    for c in cases() {
        let (drawn, t, doc, merged) = import_case(&c);
        assert!(!drawn.is_empty(), "{}: nothing drawn", c.name);
        // Model: size in points at the document resolution, transform and shape as written.
        assert!((t.runs[0].style.size_pt - c.size_pt).abs() < 0.01, "{}: size {} pt, want {}", c.name, t.runs[0].style.size_pt, c.size_pt);
        assert_eq!(t.transform, c.transform, "{}", c.name);
        match (c.box_bounds, t.shape.clone()) {
            (None, TextShape::Point) => {}
            (Some(b), TextShape::Box { x, y, width, height }) => {
                assert_eq!([x, y, x + width, y + height].map(f64::from), b, "{}", c.name)
            }
            (want, got) => panic!("{}: shape {got:?}, want box {want:?}", c.name),
        }
        // The imported pixels (what the canvas shows, and what Properties measures) are the
        // whole of Photoshop's rendering.
        let cache = t.cache.as_ref().unwrap();
        assert_eq!(cache.content_bounds(), drawn, "{}: imported pixels clipped", c.name);
        let layer_bounds = photocraft_compose::layer_bounds(&doc.layers[0], doc.bounds());
        assert_eq!(layer_bounds, drawn, "{}: layer bounds", c.name);
        // Re-rendering the imported model (what the type tool does on click) draws the same
        // text in the same place: nothing clipped, no scale or dpi drift.
        let mut engine = photocraft_text::TextEngine::new();
        let (_, ours) = engine.render(&t, doc.resolution_dpi, doc.pixel_format());
        let ours_rect = ours.surface.content_bounds();
        assert!(close(ours_rect, drawn, 1), "{}: re-render {ours_rect:?} vs file {drawn:?}", c.name);
        let o = iou(&ours.surface, cache);
        assert!(o > 0.97, "{}: re-render overlap {o}", c.name);
        // And the document composite is the file's merged composite.
        let px = photocraft_compose::flatten(&doc).px;
        let worst = common::max_diff(&px, &merged);
        assert!(worst <= 2.0 / 255.0, "{}: composite differs from the merged image by {worst}", c.name);
    }
}

#[test]
fn point_text_height_follows_the_type_size() {
    // 36 pt at 72 dpi: caps and ascenders of "Headline" are about 0.73 em tall; a line with a
    // descender spans about 0.95 em. Properties' H is the pixel height of the layer.
    let c = &cases()[0];
    let (drawn, ..) = import_case(c);
    assert!((24..=30).contains(&drawn.height()), "no-descender line: H {}", drawn.height());
    let c = Case { text: "Headline: Autumn typography", runs: vec![Run { len: 27, size: 36.0, leading: None }], ..cases().remove(0) };
    let (drawn, ..) = import_case(&c);
    assert!((32..=38).contains(&drawn.height()), "line with descenders: H {}", drawn.height());
    // Same text at 300 dpi: 36 pt = 150 px.
    let c = Case { dpi: 300.0, runs: vec![Run { len: 27, size: 150.0, leading: None }], ..c };
    let (drawn300, ..) = import_case(&c);
    let ratio = drawn300.height() as f32 / drawn.height() as f32;
    assert!((ratio - 300.0 / 72.0).abs() < 0.2, "300 dpi H {} vs 72 dpi H {}", drawn300.height(), drawn.height());
}

#[test]
fn type_layers_without_pixels_are_rendered_on_import() {
    // ag-psd, GIMP and other writers leave type layers without image data (Photoshop re-renders
    // them on open). The model must import and the layer must be drawn from it, not shown as
    // empty pixels until it is edited.
    let c = &cases()[0];
    let mut doc = Document::new("t", Size::new(900, 700), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = c.dpi;
    let data = tysh(c.text, &c.runs, c.box_bounds, c.transform, [40.0, 50.0, 400.0, 110.0]);
    let mut t = photocraft_text::psd::text_layer_from_tysh(&data, c.dpi).unwrap();
    t.cache = None;
    t.psd_raw = Some(std::sync::Arc::new(data));
    doc.layers.push(Layer::new(c.name, LayerContent::Text(t)));
    let bytes = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("t.psd", &bytes).unwrap().document;
    let LayerContent::Text(bt) = &back.layers[0].content else { panic!("not a type layer") };
    assert_eq!(bt.text, c.text);
    let drawn = bt.cache.as_ref().expect("rendered on import").content_bounds();
    assert!(!drawn.is_empty(), "nothing rendered");
    // The same text rendered by the engine lands on the same pixels.
    let mut engine = photocraft_text::TextEngine::new();
    let (_, ours) = engine.render(bt, back.resolution_dpi, back.pixel_format());
    assert!(close(ours.surface.content_bounds(), drawn, 1), "{:?} vs {drawn:?}", ours.surface.content_bounds());
    // With pixels in the file, they stay the cache (Photoshop's rendering is authoritative).
    let (drawn, t, ..) = import_case(c);
    assert_eq!(t.cache.as_ref().map(|s| s.content_bounds()), Some(drawn));
}

#[test]
fn postscript_font_names_resolve_to_the_installed_family_on_import() {
    // The family guessed from a PostScript name is not always the installed one: the bundled
    // JetBrains Mono is "JetBrainsMono-Regular", which splits into "Jet Brains Mono". The run
    // must name the family the font database really has, so nothing reports it missing.
    let c = &cases()[0];
    let mut doc = Document::new("t", Size::new(900, 700), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = c.dpi;
    let data = tysh_with_font(c.text, &c.runs, c.box_bounds, c.transform, [40.0, 50.0, 400.0, 110.0], "JetBrainsMono-Regular");
    let probe = photocraft_text::psd::text_layer_from_tysh(&data, c.dpi).unwrap();
    assert_eq!(probe.runs[0].style.font_family, "Jet Brains Mono", "the guess the import must correct");
    let mut t = probe;
    t.cache = None;
    t.psd_raw = Some(std::sync::Arc::new(data));
    doc.layers.push(Layer::new(c.name, LayerContent::Text(t)));
    let bytes = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("t.psd", &bytes).unwrap().document;
    let LayerContent::Text(bt) = &back.layers[0].content else { panic!("not a type layer") };
    assert_eq!(bt.runs[0].style.font_family, "JetBrains Mono");
    assert_eq!(bt.runs[0].style.postscript_name.as_deref(), Some("JetBrainsMono-Regular"));
    assert!(bt.cache.as_ref().is_some_and(|s| !s.content_bounds().is_empty()));
}

#[test]
fn hostile_type_sizes_without_pixels_import_without_a_huge_render() {
    // The import renders type layers that come without pixels; the model is untrusted, so an
    // absurd size or scale must not make opening the file allocate gigabytes (or fail).
    let scaled = |s: f64| Affine { m: [s, 0.0, 0.0, s, 10.0, 10.0] };
    for (size, transform, big) in
        [(12.0, scaled(1100.0), true), (1.0e30, scaled(1.0), false), (12.0, scaled(1.0e12), false), (f64::MAX, scaled(f64::MAX), false)]
    {
        let runs = vec![Run { len: 1, size, leading: None }];
        let data = tysh("M", &runs, None, transform, [10.0, 10.0, 20.0, 20.0]);
        let mut t = photocraft_text::psd::text_layer_from_tysh(&data, 72.0).unwrap();
        if big {
            // Precondition: the renderer's own guard (256 Mpx) would have drawn this one.
            let mut engine = photocraft_text::TextEngine::new();
            let layout = engine.layout(&t, 72.0);
            let ink = photocraft_text::render::ink_rect(&layout, &t.transform);
            let px = u64::from(ink.width()) * u64::from(ink.height());
            assert!(px > 64 * 1024 * 1024 && px <= 256 * 1024 * 1024, "{ink:?}");
        }
        t.cache = None;
        t.psd_raw = Some(std::sync::Arc::new(data));
        let mut doc = Document::new("t", Size::new(64, 64), ColorMode::Rgb, SampleType::U8);
        doc.layers.push(Layer::new("hostile", LayerContent::Text(t)));
        let bytes = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap().bytes;
        let back = photocraft_io::import("t.psd", &bytes).unwrap().document;
        let LayerContent::Text(bt) = &back.layers[0].content else { panic!("not a type layer") };
        let cache = bt.cache.as_ref().expect("a cache, even if empty");
        assert!(cache.content_bounds().is_empty(), "size {size}: drew {:?}", cache.content_bounds());
    }
}

#[test]
fn blank_engine_text_is_not_rendered_on_import() {
    // Photoshop draws nothing when the engine text is blank, whatever the descriptor says.
    let data = tysh("", &[Run { len: 0, size: 24.0, leading: None }], None, Affine::translate(10.0, 40.0), [10.0, 10.0, 20.0, 20.0]);
    assert!(photocraft_text::psd::engine_text_is_blank(&data));
    assert!(!photocraft_text::psd::engine_text_is_blank(&tysh("M", &[Run { len: 1, size: 24.0, leading: None }], None, Affine::IDENTITY, [0.0; 4])));
    assert!(!photocraft_text::psd::engine_text_is_blank(b"not a TySh"));
}
