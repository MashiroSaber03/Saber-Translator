//! Timing: flatten a 6016² document with 5 layers (mask, adjustment, blend modes).
//! `cargo run --release -p photocraft-compose --example bench_flatten [size]`
use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Adjustment, Document, Layer, LayerContent, LayerMask, Size};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

pub fn bench_doc(n: u32) -> Document {
    let mut d = Document::with_background("bench", Size::new(n, n), ColorMode::Rgb, SampleType::U8, Color::WHITE);
    let r = Rect::new(0, 0, n as i32, n as i32);
    let fill = |seed: u8| {
        let mut s = Surface::new(PixelFormat::RGBA8);
        let row: Vec<u8> = (0..n).flat_map(|x| [(x as u8).wrapping_mul(seed), (x / 3) as u8, seed, 200]).collect();
        let bytes: Vec<u8> = (0..n).flat_map(|_| row.iter().copied()).collect();
        s.write_interleaved(r, &bytes);
        s
    };
    let mut a = Layer::new("masked", LayerContent::Raster(fill(3)));
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::new(0, 0, n as i32 / 2, n as i32), &[0.3]);
    a.mask = Some(m);
    d.layers.push(a);
    let mut b = Layer::new("multiply", LayerContent::Raster(fill(7)));
    b.blend = BlendMode::Multiply;
    d.layers.push(b);
    d.layers.push(Layer::new("curves", LayerContent::Adjustment(Adjustment::identity_curves())));
    let mut c = Layer::new("screen", LayerContent::Raster(fill(11)));
    c.blend = BlendMode::Screen;
    c.opacity = 0.7;
    c.fill_opacity = 0.8;
    d.layers.push(c);
    d
}

fn main() {
    let n: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6016);
    let d = bench_doc(n);
    let t = std::time::Instant::now();
    let out = photocraft_compose::flatten(&d);
    println!("flatten {n}x{n}, {} layers: {:.2?} (px0 {:?})", d.layers.len(), t.elapsed(), out.px[0]);
}
