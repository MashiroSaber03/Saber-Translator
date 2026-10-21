//! Full filter timing, including tile reads, filtering, writeback and pruning.
//! RAYON_NUM_THREADS=16 cargo run --release -p photocraft-algo --example bench_pixelate -- facet 6000 4000 u8 3 output.bin
//! Optional final argument after output.bin: rgb (default), gray, cmyk or lab.
//! Run the same example at the base commit and the PR commit, then compare output.bin with cmp.
use photocraft_algo::{FilterParams, apply_in};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let case = args.first().map(String::as_str).unwrap_or("facet");
    let parse = |i: usize, fallback: i32| -> Result<i32, Box<dyn std::error::Error>> {
        Ok(match args.get(i) {
            Some(s) => s.parse()?,
            None => fallback,
        })
    };
    let (w, h, repeats) = (parse(1, 6000)?, parse(2, 4000)?, parse(4, 3)?);
    if !(1..=10000).contains(&w) || !(1..=10000).contains(&h) || !(1..=9).contains(&repeats) {
        return Err("dimensions must be 1..10000, repeats 1..9".into());
    }
    let depth = match args.get(3).map(String::as_str).unwrap_or("u8") {
        "u8" => SampleType::U8,
        "u16" => SampleType::U16,
        "f32" => SampleType::F32,
        _ => return Err("depth must be u8, u16 or f32".into()),
    };
    let params = match case {
        "facet" => FilterParams::Facet,
        "halftone" => FilterParams::ColorHalftone { max_radius: 8.0, angles: [108.0, 162.0, 90.0, 45.0] },
        "halftone-small" => FilterParams::ColorHalftone { max_radius: 2.0, angles: [108.0, 162.0, 90.0, 45.0] },
        "halftone-large" => FilterParams::ColorHalftone { max_radius: 64.0, angles: [108.0, 162.0, 90.0, 45.0] },
        _ => return Err("case must be facet, halftone, halftone-small or halftone-large".into()),
    };
    let mode = match args.get(6).map(String::as_str).unwrap_or("rgb") {
        "rgb" => ColorMode::Rgb,
        "gray" => ColorMode::Grayscale,
        "cmyk" => ColorMode::Cmyk,
        "lab" => ColorMode::Lab,
        _ => return Err("mode must be rgb, gray, cmyk or lab".into()),
    };
    let format = PixelFormat::new(mode, depth, true);
    let n = format.channels();
    let bounds = Rect::new(-17, -29, w - 17, h - 29);
    let mut source = Surface::new(format);
    for tile in bounds.tiles() {
        let r = tile.rect().intersect(&bounds);
        let mut pixels = Vec::with_capacity(r.width() as usize * r.height() as usize * n);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let seed = (x as u32).wrapping_mul(0x9e37_79b9) ^ (y as u32).wrapping_mul(0x85eb_ca6b);
                for c in 0..n {
                    let v = seed.rotate_left(c as u32 * 7).wrapping_add(c as u32 * 71) % 65536;
                    pixels.push(if c + 1 == n && seed.is_multiple_of(5) { 0.0 } else { v as f32 / 65535.0 });
                }
            }
        }
        source.write_region(r, &pixels);
    }
    let warm = Rect::new(bounds.x0, bounds.y0, bounds.x0 + w.min(128), bounds.y0 + h.min(128));
    drop(apply_in(&source, &params, warm, bounds, None, bounds));
    let mut times = Vec::new();
    for run in 0..repeats {
        let start = Instant::now();
        let result = apply_in(&source, &params, bounds, bounds, None, bounds);
        let seconds = start.elapsed().as_secs_f64();
        println!("{case} {w}x{h} {depth:?} {mode:?} run={} seconds={seconds:.6}", run + 1);
        times.push(seconds);
        if run + 1 == repeats
            && let Some(path) = args.get(5)
        {
            let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
            for tile in bounds.tiles() {
                for sample in result.read_region(tile.rect().intersect(&bounds)) {
                    file.write_all(&sample.to_bits().to_le_bytes())?;
                }
            }
            file.flush()?;
        }
    }
    times.sort_by(f64::total_cmp);
    if let Some(median) = times.get(times.len() / 2) {
        println!("MEDIAN {case} {w}x{h} {depth:?} {mode:?} seconds={median:.6}");
    }
    Ok(())
}
