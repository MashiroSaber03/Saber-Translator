//! Cold shape rendering on a synthetic 24 MP canvas, excluding fixture construction.
//! Run in release mode; keep concurrent CPU-heavy work stopped while measuring.
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{LineCap, ShapeLayer, ShapeStroke};
use photocraft_geom::{Rect, Size};
use std::time::Instant;

fn main() {
    let clip = Rect::from_size(Size::new(6000, 4000));
    for (name, dashes, cap) in [("solid", vec![], LineCap::Butt), ("dashed", vec![4.0, 2.0], LineCap::Butt), ("dotted", vec![0.0, 2.0], LineCap::Round)] {
        let shape = ShapeLayer {
            path: photocraft_vector::shapes::rounded_rect(32.0, 32.0, 5936.0, 3936.0, [0.0; 4]),
            stroke: Some(ShapeStroke { width: 8.0, dashes, cap, ..Default::default() }),
            ..Default::default()
        };
        for _ in 0..3 {
            let start = Instant::now();
            let surface = photocraft_vector::render_shape(&shape, PixelFormat::new(ColorMode::Rgb, SampleType::U8, true), clip);
            println!("{name}: {:.3} ms, bounds {:?}", start.elapsed().as_secs_f64() * 1000.0, surface.content_bounds());
        }
    }
}
