//! Reproducible end-to-end Shape Blur timings; no native CPU flags required.
//! Arguments: width height repetitions radius (or all) depth (8/16/32) shape (or all).
use std::{hint::black_box, time::Instant};

use photocraft_algo::{BlurShape, FilterParams, apply_in};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| args.get(i).map(String::as_str).unwrap_or(default).to_owned();
    let w: i32 = arg(0, "6000").parse()?;
    let h: i32 = arg(1, "4000").parse()?;
    let repeats: usize = arg(2, "3").parse()?;
    if !(1..=8000).contains(&w) || !(1..=8000).contains(&h) || !(1..=20).contains(&repeats) {
        return Err("size or repetition count outside benchmark limits".into());
    }
    let radius = arg(3, "all");
    let radii = if radius == "all" { vec![5.0, 25.0, 100.0] } else { vec![radius.parse::<f32>()?] };
    if radii.iter().any(|r| !r.is_finite() || !(0.0..=1000.0).contains(r)) {
        return Err("radius outside 0..=1000".into());
    }
    let depth = arg(4, "8");
    let sample = match depth.as_str() {
        "8" => SampleType::U8,
        "16" => SampleType::U16,
        "32" => SampleType::F32,
        _ => return Err("depth must be 8, 16 or 32".into()),
    };
    let only = arg(5, "all");
    let bounds = Rect::new(0, 0, w, h);
    let mut src = Surface::new(PixelFormat::new(ColorMode::Rgb, sample, true));
    // Write in rows: input setup does not need a full-image float buffer.
    for y in 0..h {
        let data: Vec<f32> = (0..w)
            .flat_map(|x| [(x % 256) as f32 / 255.0, ((x * 3 + y) % 256) as f32 / 255.0, ((x ^ y) % 256) as f32 / 255.0, ((x + y) % 5) as f32 / 4.0])
            .collect();
        src.write_region(Rect::new(0, y, w, y + 1), &data);
    }
    for shape in [
        BlurShape::Circle,
        BlurShape::Ring,
        BlurShape::Square,
        BlurShape::Diamond,
        BlurShape::Triangle,
        BlurShape::Hexagon,
        BlurShape::Star,
        BlurShape::Heart,
        BlurShape::Cross,
    ] {
        if only != "all" && only != format!("{shape:?}").to_lowercase() {
            continue;
        }
        for &radius in &radii {
            let p = FilterParams::ShapeBlur { radius, shape };
            drop(black_box(apply_in(&src, &p, bounds, bounds, None, bounds)));
            let mut times = Vec::new();
            for _ in 0..repeats {
                let start = Instant::now();
                let result = apply_in(&src, &p, bounds, bounds, None, bounds);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                drop(black_box(result));
            }
            times.sort_by(f64::total_cmp);
            let median = times.get(times.len() / 2).copied().unwrap_or(0.0);
            println!("{w},{h},{depth},{shape:?},{radius},{median:.3},{times:?}");
        }
    }
    Ok(())
}
