//! Artboards: contents clipped to the board, background painted behind them, backdrop untouched
//! outside. Checked at 8, 16 and 32-bit.

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_compose::{render, render_tiled};
use photocraft_doc::{Artboard, ArtboardBackground, Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size};

fn artboard_doc(depth: SampleType, background: ArtboardBackground, blend: BlendMode) -> Document {
    let mut d = Document::new("a", Size::new(40, 20), ColorMode::Rgb, depth);
    let fmt = PixelFormat::new(ColorMode::Rgb, depth, true);
    // A red layer spanning the whole canvas, inside an artboard covering x 10..30, y 5..15.
    let mut red = Layer::raster("red", fmt);
    red.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 40, 10), &[1.0, 0.0, 0.0, 1.0]);
    let mut g = Layer::group("Artboard 1", vec![red]);
    g.blend = blend;
    if let LayerContent::Group(gr) = &mut g.content {
        gr.artboard = Some(Artboard { rect: Rect::new(10, 5, 30, 15), background, preset: String::new() });
    }
    d.layers.push(g);
    d
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2.0 / 255.0)
}

#[test]
fn contents_are_clipped_and_background_painted() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for blend in [BlendMode::PassThrough, BlendMode::Normal] {
            let d = artboard_doc(depth, ArtboardBackground::White, blend);
            let buf = render(&d, d.bounds());
            // Outside the board: nothing (the red layer is clipped away).
            assert!(close(buf.get(2, 2), [0.0; 4]), "{depth:?} {:?}", buf.get(2, 2));
            assert!(close(buf.get(35, 8), [0.0; 4]));
            // Inside, over the red: red; inside below it: the white board background.
            assert!(close(buf.get(15, 7), [1.0, 0.0, 0.0, 1.0]), "{:?}", buf.get(15, 7));
            assert!(close(buf.get(15, 12), [1.0, 1.0, 1.0, 1.0]), "{:?}", buf.get(15, 12));
            // Edges are half-open.
            assert!(close(buf.get(29, 14), [1.0, 1.0, 1.0, 1.0]));
            assert!(close(buf.get(30, 14), [0.0; 4]));
            // Tiled rendering agrees with one pass.
            assert_eq!(render_tiled(&d, d.bounds(), 7).px, buf.px);
        }
    }

    // Proxy pixel (x,y) samples (4*x,4*y); unaligned half-open artboard edges must agree.
    let original = artboard_doc(SampleType::F32, ArtboardBackground::White, BlendMode::PassThrough);
    let source = render(&original, original.bounds());
    let proxy = photocraft_compose::proxy::proxy_document(&original, 4);
    let reduced = render(&proxy, proxy.bounds());
    for y in 0..proxy.size.height as i32 {
        for x in 0..proxy.size.width as i32 {
            assert!(close(reduced.get(x, y), source.get(4 * x, 4 * y)), "proxy artboard differs at ({x}, {y})");
        }
    }
}

#[test]
fn background_kinds() {
    let at = |bg| render(&artboard_doc(SampleType::U8, bg, BlendMode::PassThrough), Rect::from_xywh(20, 12, 1, 1)).px[0];
    assert!(close(at(ArtboardBackground::Black), [0.0, 0.0, 0.0, 1.0]));
    assert!(close(at(ArtboardBackground::Transparent), [0.0; 4]));
    assert!(close(at(ArtboardBackground::Custom(Color::rgb(0.0, 0.0, 1.0))), [0.0, 0.0, 1.0, 1.0]));
}

#[test]
fn backdrop_outside_the_board_is_untouched() {
    let mut d = artboard_doc(SampleType::U8, ArtboardBackground::Transparent, BlendMode::PassThrough);
    let mut bg = Layer::raster("bg", d.pixel_format());
    bg.surface_mut().unwrap().fill_rect(d.bounds(), &[0.0, 1.0, 0.0, 1.0]);
    d.layers.insert(0, bg);
    let buf = render(&d, d.bounds());
    assert!(close(buf.get(2, 2), [0.0, 1.0, 0.0, 1.0]));
    assert!(close(buf.get(15, 7), [1.0, 0.0, 0.0, 1.0]));
    assert!(close(buf.get(15, 12), [0.0, 1.0, 0.0, 1.0]));
}
