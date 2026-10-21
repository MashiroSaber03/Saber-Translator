//! 24 MP timing of the red-eye kernel on a window around one click (not a full-image scan).
//! `cargo run --release -p photocraft-algo --example bench_redeye`
use std::time::Instant;

use photocraft_algo::redeye;

fn main() {
    let (doc_w, doc_h) = (6016usize, 4000usize);
    let click = (doc_w as i32 / 2, doc_h as i32 / 2);
    let pupil = 50.0f32;
    let radius = redeye::search_radius(pupil);
    let halo = redeye::halo_radius();
    let ext = radius.ceil() as i32 + halo;
    let x0 = click.0.saturating_sub(ext);
    let y0 = click.1.saturating_sub(ext);
    let area_w = (2 * ext + 1) as usize;
    let area_h = area_w;
    let window_px = area_w.saturating_mul(area_h);
    let doc_px = doc_w.saturating_mul(doc_h);
    if window_px == 0 || window_px > (redeye::MAX_WINDOW as usize).saturating_pow(2) {
        eprintln!("window {area_w}×{area_h} exceeds the 512² cap");
        std::process::exit(1);
    }
    if window_px >= doc_px {
        eprintln!("window {window_px} px is not smaller than the {doc_px} px document (full-image scan)");
        std::process::exit(1);
    }

    // Full 24.1 MP RGBA8 document; the kernel is given only the search window + halo.
    let mut doc = vec![0u8; doc_px.saturating_mul(4)];
    for y in 0..area_h {
        for x in 0..area_w {
            let dx = x0 + x as i32 - click.0;
            let dy = y0 + y as i32 - click.1;
            let d = (dx as f32).hypot(dy as f32);
            let gx = (x0 as usize).saturating_add(x);
            let gy = (y0 as usize).saturating_add(y);
            if gx >= doc_w || gy >= doc_h {
                continue;
            }
            let i = gy.saturating_mul(doc_w).saturating_add(gx).saturating_mul(4);
            let rgb = if d <= 1.5 {
                [255u8, 255, 255]
            } else if d <= 18.0 {
                [242, 26, 26]
            } else {
                [97, 71, 41]
            };
            if let Some(px) = doc.get_mut(i..i.saturating_add(4)) {
                px[0] = rgb[0];
                px[1] = rgb[1];
                px[2] = rgb[2];
                px[3] = 255;
            }
        }
    }

    let mut px = vec![[0.0f32; 4]; window_px];
    let origin = (x0, y0);
    for y in 0..area_h {
        for x in 0..area_w {
            let gx = (x0 as usize).saturating_add(x);
            let gy = (y0 as usize).saturating_add(y);
            let slot = match px.get_mut(y.saturating_mul(area_w).saturating_add(x)) {
                Some(s) => s,
                None => continue,
            };
            if gx >= doc_w || gy >= doc_h {
                *slot = [0.38, 0.28, 0.16, 1.0];
                continue;
            }
            let i = gy.saturating_mul(doc_w).saturating_add(gx).saturating_mul(4);
            let r = doc.get(i).copied().unwrap_or(0) as f32 / 255.0;
            let g = doc.get(i + 1).copied().unwrap_or(0) as f32 / 255.0;
            let b = doc.get(i + 2).copied().unwrap_or(0) as f32 / 255.0;
            let a = doc.get(i + 3).copied().unwrap_or(255) as f32 / 255.0;
            *slot = [r, g, b, a];
        }
    }

    let mut warm = px.clone();
    let _ = redeye::apply(&mut warm, area_w, area_h, click, origin, pupil, 50.0);

    let t = Instant::now();
    let n = redeye::apply(&mut px, area_w, area_h, click, origin, pupil, 50.0);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "redeye  document {doc_w}×{doc_h} ({:.2} MP)  window {area_w}×{area_h} (+{halo} px halo, {window_px} px)  pixels {n}  {ms:.3} ms",
        doc_px as f64 / 1_000_000.0
    );
    if n == 0 {
        eprintln!("expected to correct the synthetic pupil");
        std::process::exit(1);
    }
}
