//! Colour-managed canvas display cost on a 36 MP document (#46).
//!
//! ```sh
//! cargo run --release -p photocraft-engine --example bench_display -- [--size 7360x4912] [--reps 5]
//! ```
//!
//! Times, per profile: the composite (shared by every path), the CPU canvas conversion before
//! (`Buffer::to_rgba8`, no colour management) and after (`CanvasDisplay::to_rgba8`), the
//! per-frame display lookup (`display_signature`, what the GPU canvas checks each frame) and
//! building the 33³ GPU display LUT (once per profile/monitor change).

use std::time::Instant;

use photocraft_cms::Builtin;
use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, Size};
use photocraft_engine::Session;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect(),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) =
        arg(&args, "--size").and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap_or(7360), b.parse().unwrap_or(4912)))).unwrap_or((7360, 4912));
    let reps: usize = arg(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    println!("document {w}×{h} ({:.1} MP), median of {reps}", (w * h) as f64 / 1e6);
    println!("{:<22} {:>10} {:>14} {:>14} {:>12} {:>10}", "profile", "composite", "CPU before", "CPU after", "per frame", "LUT build");
    for (name, profile, depth) in [
        ("sRGB (untagged)", None, SampleType::U8),
        ("Display P3", Some(Builtin::DisplayP3), SampleType::U8),
        ("ProPhoto", Some(Builtin::ProPhotoCompat), SampleType::U16),
        ("linear sRGB (EXR)", Some(Builtin::LinearSrgb), SampleType::F32),
    ] {
        let mut d = Document::with_background("b", Size::new(w, h), ColorMode::Rgb, depth, Color::rgb(0.7, 0.4, 0.2));
        d.icc_profile = profile.map(|b| b.profile().to_bytes());
        let s = Session::new();
        let mut buf = None;
        let composite = time(reps, || buf = Some(photocraft_compose::flatten(&d)));
        let buf = buf.expect("composited");
        let before = time(reps, || {
            std::hint::black_box(buf.to_rgba8());
        });
        let display = s.color.canvas_display(&d).expect("display");
        let after = time(reps, || {
            let d = s.color.canvas_display(&d).expect("display");
            std::hint::black_box(if d.is_identity() { buf.to_rgba8() } else { d.to_rgba8(&buf) });
        });
        let frame = time(reps * 200, || {
            std::hint::black_box(s.color.display_signature(&d));
        });
        let lut = time(reps, || {
            std::hint::black_box(s.color.canvas_lut(&d, 33).expect("lut"));
        });
        let tag = if display.is_identity() { " (identity)" } else { "" };
        println!("{:<22} {composite:>8.1}ms {before:>12.1}ms {after:>12.1}ms {:>10.1}µs {lut:>8.1}ms{tag}", name, frame * 1000.0);
    }
}
