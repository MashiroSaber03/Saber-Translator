//! Paired, end-to-end Radial Blur benchmark against the current upstream sampler.
//! cargo run --release -j 12 -p photocraft-algo --example bench_radial
//! Defaults: 6000x4000 RGBA8, Good, amount 100, 12 Rayon workers, three runs.
//! Optional positional arguments: width height amount repeats.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use photocraft_algo::{Ctx, FilterParams, Image, RadialMethod, RadialQuality, TILE, apply_in, kernel};
    use photocraft_color::PixelFormat;
    use photocraft_geom::Rect;
    use photocraft_raster::Surface;
    use rayon::prelude::*;
    use std::time::Instant;

    // Match apply_tiled_with's current upstream pipeline for this benchmark's
    // full-canvas, unselected global filter. kernel retains the upstream sampler.
    // Both paths read one normalized source, batch the same 256px tiles within
    // the same 256MiB result budget, write the same surface depth, and prune.
    fn upstream(source: &Surface, params: &FilterParams, rect: Rect) -> Surface {
        let mut output = source.clone();
        let image = Image::read(source, rect);
        let format = source.format();
        let ctx = Ctx { bounds: rect, mode: format.mode, alpha: format.alpha };
        let tiles: Vec<Rect> = rect.tiles().map(|tile| tile.rect().intersect(&rect)).collect();
        let per_tile = TILE as usize * TILE as usize * format.channels() * size_of::<f32>();
        let group = ((256 << 20) / per_tile).max(rayon::current_num_threads());
        for chunk in tiles.chunks(group) {
            let results: Vec<_> = chunk.par_iter().map(|tile| (*tile, kernel(params, &image, *tile, &ctx))).collect();
            for (tile, samples) in results {
                output.write_region(tile, &samples);
            }
        }
        output.prune();
        output
    }

    let args: Vec<_> = std::env::args().skip(1).collect();
    let parse = |i: usize, default: i32| -> Result<i32, Box<dyn std::error::Error>> {
        Ok(match args.get(i) {
            Some(value) => value.parse()?,
            None => default,
        })
    };
    let width = parse(0, 6000)?;
    let height = parse(1, 4000)?;
    let amount = parse(2, 100)?;
    let repeats = parse(3, 3)?;
    if !(1..=10000).contains(&width) || !(1..=10000).contains(&height) || !(0..=100).contains(&amount) || !(1..=7).contains(&repeats) {
        return Err("expected dimensions 1..10000, amount 0..100, repeats 1..7".into());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(12).build()?;
    let rect = Rect::new(0, 0, width, height);
    let mut source = Surface::new(PixelFormat::RGBA8);
    for tile in rect.tiles() {
        let area = tile.rect().intersect(&rect);
        let mut values = Vec::new();
        values.try_reserve_exact(area.width() as usize * area.height() as usize * 4)?;
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                let seed = (x as u32).wrapping_mul(0x9e37_79b9) ^ (y as u32).wrapping_mul(0x85eb_ca6b);
                for channel in 0..4 {
                    let level = seed.rotate_left(channel * 7).wrapping_add(channel * 71) % 256;
                    values.push(if channel == 3 && seed.is_multiple_of(5) { 0.0 } else { level as f32 / 255.0 });
                }
            }
        }
        source.write_region(area, &values);
    }
    pool.install(|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Small untimed warmup initializes worker/code paths without doubling
        // the cost of each 24MP measurement. Full-size runs include page faults.
        let warm = Rect::new(0, 0, width.min(128), height.min(128));
        for method in [RadialMethod::Spin, RadialMethod::Zoom] {
            let params = FilterParams::RadialBlur { amount: amount as f32, method, quality: RadialQuality::Good, center_x: 0.5, center_y: 0.5 };
            upstream(&source, &params, warm);
            apply_in(&source, &params, warm, warm, None, warm);
            let (mut before, mut after) = (Vec::new(), Vec::new());
            for run in 0..repeats {
                let execute = |original: bool| {
                    let start = Instant::now();
                    let output = if original { upstream(&source, &params, rect) } else { apply_in(&source, &params, rect, rect, None, rect) };
                    (output, start.elapsed().as_secs_f64())
                };
                let ((original, old_time), (optimized, new_time)) = if run % 2 == 0 {
                    let first = execute(true);
                    (first, execute(false))
                } else {
                    let first = execute(false);
                    (execute(true), first)
                };
                // Exact comparison after timing, including alpha and native U8
                // writeback; benchmark errors never silently publish a speedup.
                for tile in rect.tiles() {
                    let area = tile.rect().intersect(&rect);
                    if original.read_region(area) != optimized.read_region(area) {
                        return Err(format!("output mismatch for {method:?} at {area:?}").into());
                    }
                }
                before.push(old_time);
                after.push(new_time);
                println!("{method:?} {width}x{height} Good amount={amount} run={}: upstream={old_time:.6}s optimized={new_time:.6}s exact=true", run + 1);
            }
            before.sort_by(f64::total_cmp);
            after.sort_by(f64::total_cmp);
            let i = before.len() / 2;
            let old = before.get(i).copied().ok_or("missing upstream timing")?;
            let new = after.get(i).copied().ok_or("missing optimized timing")?;
            println!("MEDIAN {method:?}: upstream={old:.6}s optimized={new:.6}s speedup={:.3}x exact=true workers=12 repeats={repeats}", old / new);
        }
        Ok(())
    })
    .map_err(|error| -> Box<dyn std::error::Error> { error })
}

#[cfg(target_arch = "wasm32")]
fn main() {}
