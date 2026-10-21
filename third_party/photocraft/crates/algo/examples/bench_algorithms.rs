//! Full filter timing, including tile reads, filtering, writeback and pruning.
//! RAYON_NUM_THREADS=16 cargo run --release -p photocraft-algo --example bench_algorithms -- extrude 6000 4000 u8 3 output.bin
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
    let case = args.first().map(String::as_str).unwrap_or("extrude");
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
    let extrude = |size, depth| FilterParams::Extrude {
        kind: photocraft_algo::ExtrudeType::Blocks,
        size,
        depth,
        level_based: false,
        solid_front: false,
        mask_incomplete: false,
        seed: 17,
    };
    let params = match case {
        "extrude" => extrude(30.0, 30.0),
        "extrude-dense" => extrude(8.0, 128.0),
        "extrude-small" => extrude(8.0, 64.0),
        "wind" => FilterParams::Wind { method: photocraft_algo::WindMethod::Wind, from_right: false, seed: 17 },
        "blast" => FilterParams::Wind { method: photocraft_algo::WindMethod::Blast, from_right: false, seed: 17 },
        "blast-right" => FilterParams::Wind { method: photocraft_algo::WindMethod::Blast, from_right: true, seed: 17 },
        "spin" => FilterParams::SpinBlur {
            pins: vec![photocraft_algo::SpinPin { radius_x: 1.0, radius_y: 1.0, blur_angle: 30.0, angle: 17.0, ..Default::default() }],
        },
        "spin-default" => FilterParams::SpinBlur { pins: vec![Default::default()] },
        "lens" | "lens-ca" | "trap" | "trap-wide" | "trap-default" => FilterParams::Invert,
        _ => return Err("unknown case".into()),
    };
    let mode = match args.get(6).map(String::as_str).unwrap_or("rgb") {
        "rgb" => ColorMode::Rgb,
        "gray" => ColorMode::Grayscale,
        "cmyk" => ColorMode::Cmyk,
        "lab" => ColorMode::Lab,
        _ => return Err("mode must be rgb, gray, cmyk or lab".into()),
    };
    if case.starts_with("trap") && mode != ColorMode::Cmyk {
        return Err("Trap needs CMYK samples".into());
    }
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
    let run_filter = |area: Rect| match case {
        "lens" | "lens-ca" => photocraft_algo::lens::correct(
            &source,
            area,
            &photocraft_algo::lens::LensCorrection {
                distortion: 12.0,
                angle: 2.5,
                vignette_amount: 20.0,
                red_cyan: if case == "lens-ca" { 12.0 } else { 0.0 },
                blue_yellow: if case == "lens-ca" { -8.0 } else { 0.0 },
                ..Default::default()
            },
        ),
        "trap" | "trap-wide" | "trap-default" => {
            let mut data = source.read_region(area);
            photocraft_algo::trap::trap(
                &mut data,
                area.width() as usize,
                area.height() as usize,
                n,
                if case == "trap-wide" {
                    50
                } else if case == "trap-default" {
                    1
                } else {
                    10
                },
            );
            let mut result = source.clone();
            result.write_region(area, &data);
            result.prune();
            result
        }
        _ => apply_in(&source, &params, area, bounds, None, bounds),
    };
    let warm = Rect::new(bounds.x0, bounds.y0, bounds.x0 + w.min(128), bounds.y0 + h.min(128));
    drop(run_filter(warm));
    let mut times = Vec::new();
    for run in 0..repeats {
        let start = Instant::now();
        let result = run_filter(bounds);
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
