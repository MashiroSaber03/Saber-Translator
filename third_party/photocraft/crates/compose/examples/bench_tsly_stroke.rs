//! Reproduce tall-document TSL-off stroke-effect compositing from issue #2102.
//! Run in release mode: `cargo run --release -p photocraft-compose --example bench_tsly_stroke`.

use std::error::Error;
use std::time::Instant;

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_compose::render_bands;
use photocraft_doc::{Document, Effect, FxCommon, FxPaint, Layer, Size, StrokeFx, StrokePosition};
use photocraft_geom::Rect;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 103_675;
const BAND_ROWS: i32 = 4352;

fn document_with_strokes(layer_count: usize) -> Result<Document, Box<dyn Error>> {
    let mut document = Document::new("#2102 tall TSL-off strokes", Size::new(WIDTH, HEIGHT), ColorMode::Rgb, SampleType::U8);
    let step = HEIGHT / (layer_count as u32 + 1);
    for i in 0..layer_count {
        let y = step.saturating_mul(i as u32 + 1);
        let mut layer = Layer::raster("Outside stroke", PixelFormat::RGBA8);
        let surface = layer.surface_mut().ok_or("raster layer has no surface")?;
        surface.fill_rect(Rect::new(360, y as i32, 1560, y as i32 + 120), &[0.2, 0.5, 0.9, 1.0]);
        layer.effects.items.push(Effect::Stroke(StrokeFx {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            size: 6.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::rgb(0.8, 0.1, 0.2)),
        }));
        layer.advanced.transparency_shapes = false;
        document.layers.push(layer);
    }
    Ok(document)
}

fn main() -> Result<(), Box<dyn Error>> {
    for layer_count in [3, 10] {
        let document = document_with_strokes(layer_count)?;
        let start = Instant::now();
        render_bands(&document, document.bounds(), BAND_ROWS, |_| -> Result<(), &'static str> { Ok(()) })?;
        println!("{WIDTH}×{HEIGHT}, {layer_count} TSL-off 6 px outside strokes: {:.3} s", start.elapsed().as_secs_f64());
    }
    Ok(())
}
