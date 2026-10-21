//! Synthetic existing preview workload, including proxy creation, command/history and CPU
//! flattening. Excludes scheduling, egui frames, display conversion, GPU upload/presentation.
use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_ui_egui::{filter_dialog, proxy};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = |key: &str, default| std::env::var(key).ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(default);
    let (w, h, reps) = (n("PHOTOCRAFT_UNSHARP_WIDTH", 6000), n("PHOTOCRAFT_UNSHARP_HEIGHT", 4000), n("PHOTOCRAFT_UNSHARP_REPS", 5));
    if w == 0 || h == 0 || w > 8192 || h > 8192 || !(1..=15).contains(&reps) {
        return Err("invalid benchmark dimensions/reps".into());
    }
    let depth = match n("PHOTOCRAFT_UNSHARP_DEPTH", 8) {
        8 => SampleType::U8,
        16 => SampleType::U16,
        32 => SampleType::F32,
        _ => return Err("invalid depth".into()),
    };
    let mut doc = Document::with_background("Synthetic Unsharp Mask", Size::new(w, h), ColorMode::Rgb, depth, Color::WHITE);
    let layer = doc.layers.first_mut().ok_or("missing layer")?;
    let id = layer.id;
    let LayerContent::Raster(s) = &mut layer.content else { return Err("raster required".into()) };
    for y in 0..h as i32 {
        let data: Vec<_> = (0..w as i32)
            .flat_map(|x| [((x * 7 + y * 13) % 257) as f32 / 256.0, ((x ^ y) % 251) as f32 / 250.0, ((x + y * 11) % 127) as f32 / 126.0, 1.0])
            .collect();
        s.write_region(Rect::new(0, y, w as i32, y + 1), &data);
    }
    let cmd = "filter.sharpen.unsharpMask";
    let cases = std::env::var("PHOTOCRAFT_UNSHARP_CASES").unwrap_or_else(|_| "0.6:150:0,3:150:8,10:300:16,30:150:0".into());
    for case in cases.split(',') {
        let v: Vec<f64> = case.split(':').map(str::parse).collect::<Result<_, _>>()?;
        let (&radius, &amount, &threshold) = (v.first().ok_or("radius")?, v.get(1).ok_or("amount")?, v.get(2).ok_or("threshold")?);
        for reduced in [true, false] {
            let k = filter_dialog::preview_factor(cmd, &json!({"radius":radius}), proxy::preview_factor(&doc, reduced));
            let mut times = Vec::new();
            let mut checksum = 0u64;
            for rep in 0..reps + 2 {
                let start = Instant::now();
                let result = filter_dialog::preview_document(&doc, Some(id), cmd, &json!({"radius":radius,"amount":amount,"threshold":threshold}), k)
                    .ok_or("preview failed")?;
                let pixels = photocraft_compose::flatten(&result);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                // Outside timing: stable hash over the full flattened result, no image file needed.
                let hash = pixels.px.iter().flatten().fold(0u64, |h, v| h.wrapping_mul(1099511628211).wrapping_add(u64::from(v.to_bits())));
                if rep == 0 {
                    checksum = hash;
                } else if checksum != hash {
                    return Err("unstable preview output".into());
                }
                if rep >= 2 {
                    times.push(ms);
                }
            }
            let mut sorted = times.clone();
            sorted.sort_by(f64::total_cmp);
            println!(
                "{}",
                json!({"width":w,"height":h,"depth":n("PHOTOCRAFT_UNSHARP_DEPTH",8),"radius":radius,"amount":amount,"threshold":threshold,"proxy_factor":k,"runs_ms":times,"median_ms":sorted.get(sorted.len()/2),"checksum":checksum,"workers":rayon::current_num_threads(),"scope":"existing preview_document + CPU flatten; excludes GUI and GPU"})
            );
        }
    }
    Ok(())
}
