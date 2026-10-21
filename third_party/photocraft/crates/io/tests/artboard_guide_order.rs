//! Regression tests for issue #722: an artboard's `guideIndeces` must name the
//! guides in resource 1032's own order. `psd_export` writes 1032 vertical
//! first and then horizontal, but `guides_in` numbered horizontal first, so a
//! document with guides of both orientations associated the artboard with the
//! wrong guides.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Artboard, ArtboardBackground, Document, Guides, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_io::comps_map::guides_in;

#[test]
fn guides_in_numbers_vertical_first_like_resource_1032() {
    // One vertical guide at x=20 inside the rect, one horizontal at y=90 outside it.
    // In 1032's order the vertical guide is index 0 and the horizontal is index 1,
    // so the artboard rect (x 10..30, y 0..100) must select index 0.
    let g = Guides { horizontal: vec![90.0], vertical: vec![20.0], ..Default::default() };
    let r = Rect::new(10, 0, 30, 50); // contains x=20 (vertical), not y=90
    assert_eq!(guides_in(&g, r), vec![0], "the vertical guide is 1032 index 0");

    // Both inside: vertical (1032 index 0) before horizontal (1032 index 1).
    let g2 = Guides { horizontal: vec![50.0], vertical: vec![50.0], ..Default::default() };
    let r2 = Rect::new(0, 0, 100, 100);
    assert_eq!(guides_in(&g2, r2), vec![0, 1], "vertical must be numbered before horizontal");
}

fn doc_with_board(rect: Rect) -> Document {
    let mut d = Document::new("s", Size::new(100, 100), ColorMode::Rgb, SampleType::U8);
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
    let mut a = Layer::raster("A", fmt);
    a.surface_mut().unwrap().fill_rect(Rect::new(1, 1, 2, 2), &[1.0, 0.0, 0.0, 1.0]);
    let mut g1 = Layer::group("Board", vec![a]);
    if let LayerContent::Group(g) = &mut g1.content {
        g.artboard = Some(Artboard { rect, background: ArtboardBackground::White, preset: String::new() });
    }
    d.layers = vec![g1];
    d.guides = Guides { horizontal: vec![90.0], vertical: vec![20.0], ..Default::default() };
    d
}

#[test]
fn exported_artboard_indices_match_guides_resource() {
    use photocraft_io::ExportOptions;
    let d = doc_with_board(Rect::new(10, 0, 30, 50));
    let bytes = export(&d, "x.psd", &ExportOptions::default()).expect("export").bytes;
    let f = photocraft_psd::PsdFile::from_bytes(&bytes).expect("parse");
    let res = f.resources.iter().find(|r| r.id == 1032).expect("guides resource");
    // 1032: version(4) hRes(4) vRes(4) count(4), then 5 bytes per guide: loc(4) dir(1).
    let dta = &res.data;
    let n = u32::from_be_bytes(dta[12..16].try_into().unwrap()) as usize;
    let mut order: Vec<u8> = Vec::new();
    for i in 0..n {
        let o = 16 + i * 5;
        order.push(dta[o + 4]);
    }
    assert_eq!(order, vec![0, 1], "1032 must be vertical-first");

    let artb = f.layers().iter().find(|r| r.name() == "Board").and_then(|r| r.block(b"artb")).expect("artb block");
    // Find guideIndeces in the descriptor bytes: key(12) + type(4) + count(4) + elements.
    let hay = &artb.data;
    let pos = hay.windows(12).position(|w| w == b"guideIndeces").expect("guideIndeces key");
    let count = u32::from_be_bytes(hay[pos + 16..pos + 20].try_into().unwrap()) as usize;
    assert_eq!(count, 1, "exactly one guide inside the board");
    // element: type(4) + int(4)
    let idx = i32::from_be_bytes(hay[pos + 24..pos + 28].try_into().unwrap());
    assert_eq!(idx, 0, "the inside guide is the vertical one, 1032 index 0");
}

use photocraft_io::export;
