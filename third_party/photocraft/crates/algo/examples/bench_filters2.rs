//! Timing for the heavier second-batch filters on a ~24 MP (6000×4000) RGBA8 layer.
//! `cargo run --release -p photocraft-algo --example bench_filters2 [filter-name-substring]`
use photocraft_algo::*;
use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() {
    let only = std::env::args().nth(1);
    let (w, h) = (6000, 4000);
    let r = Rect::new(0, 0, w, h);
    let mut s = Surface::new(PixelFormat::RGBA8);
    // Busy deterministic content (gradients plus texture) so edge-aware filters do real work.
    let bytes: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| [(x % 256) as u8, ((x * 3 + y) / 11 % 256) as u8, ((x ^ y) % 256) as u8, 255])).collect();
    s.write_interleaved(r, &bytes);
    let cases = vec![
        FilterParams::OilPaint { stylization: 4.0, cleanliness: 5.0, scale: 1.0, bristle_detail: 5.0, lighting: true, angle: -60.0, shine: 1.0 },
        FilterParams::ReduceNoise { strength: 6.0, preserve_details: 60.0, reduce_color_noise: 45.0, sharpen_details: 25.0, remove_jpeg_artifact: false },
        FilterParams::LensBlur {
            radius: 15.0,
            blades: 6,
            curvature: 0.0,
            rotation: 0.0,
            depth: DepthSource::None,
            focal_distance: 0.0,
            invert_depth: false,
            brightness: 20.0,
            threshold: 230.0,
            noise: 0.0,
            distribution: Distribution::Uniform,
            monochromatic: false,
            seed: 0,
            depth_map: None,
        },
        FilterParams::SmartBlur { radius: 3.0, threshold: 25.0, quality: BlurQuality::High, mode: SmartBlurMode::Normal },
        FilterParams::ShapeBlur { radius: 10.0, shape: BlurShape::Star },
        FilterParams::TiltShift { blur: 15.0, center_x: 0.5, center_y: 0.5, angle: 0.0, focus: 0.1, transition: 0.15 },
        FilterParams::IrisBlur { pins: vec![IrisPin::default()] },
        FilterParams::FieldBlur { pins: vec![FieldPin { x: 0.2, y: 0.2, blur: 0.0 }, FieldPin { x: 0.8, y: 0.8, blur: 30.0 }] },
        FilterParams::SpinBlur { pins: vec![SpinPin::default()] },
        FilterParams::PathBlur { paths: vec![BlurPath::default()] },
        FilterParams::Crystallize { cell_size: 10.0, seed: 0 },
        FilterParams::ColorHalftone { max_radius: 8.0, angles: [108.0, 162.0, 90.0, 45.0] },
        FilterParams::LightingEffects {
            lights: vec![Light::default()],
            gloss: 0.0,
            metallic: 0.0,
            exposure: 0.0,
            ambience: 8.0,
            texture: TextureChannel::Luminance,
            height: 50.0,
            white_is_high: true,
        },
        FilterParams::Relight { angle: 45.0, elevation: 40.0, intensity: 40.0, ambient: 55.0, warmth: 0.0, softness: 25.0 },
    ];
    for p in cases {
        if only.as_deref().is_some_and(|o| !p.label().to_lowercase().contains(&o.to_lowercase())) {
            continue;
        }
        let area = output_area(&p, s.content_bounds(), r, None).intersect(&r);
        let t = std::time::Instant::now();
        let out = apply(&s, &p, area, r, None);
        println!("{:<18} {w}x{h} RGBA8: {:.2?}", p.label(), t.elapsed());
        drop(out);
    }
}
