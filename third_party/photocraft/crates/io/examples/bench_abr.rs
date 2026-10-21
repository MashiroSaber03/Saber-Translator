//! Brush import and painting benchmark: parse + map a synthetic 100-brush ABR v6 file (256 px
//! RLE-compressed tips, full dynamics descriptors), then paint a 5000-dab stroke with an imported
//! brush on a 6000×6000 RGBA8 layer.
//!
//! `cargo run --release -p photocraft-io --example bench_abr`
#![allow(clippy::unwrap_used)]

use std::time::Instant;

use photocraft_color::PixelFormat;
use photocraft_paint::{StrokePoint, render_stroke};
use photocraft_psd::abr::{AbrSample, write_v6};
use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value};
use photocraft_raster::Surface;

fn prc(v: f64) -> Value {
    Value::UnitFloat { unit: *b"#Prc", value: v }
}

fn var(control: i32, jitter: f64) -> Value {
    Value::Descriptor(Descriptor::new("brVr").with("bVTy", Value::Integer(control)).with("jitter", prc(jitter)))
}

fn main() {
    let n = 256u32;
    let samples: Vec<AbrSample> = (0..100)
        .map(|i| {
            let data = (0..n * n)
                .map(|k| {
                    let (x, y) = ((k % n) as f32 - 128.0, (k / n) as f32 - 128.0);
                    let r = (x * x + y * y).sqrt() / 128.0;
                    (((1.0 - r) * 255.0).clamp(0.0, 255.0) as u8) & if (k + i) % 7 == 0 { 0x7f } else { 0xff }
                })
                .collect();
            AbrSample { id: format!("$tip-{i}"), width: n, height: n, depth: 8, data }
        })
        .collect();
    let presets: Vec<Descriptor> = (0..100)
        .map(|i| {
            Descriptor::new("brushPreset")
                .with("Nm  ", Value::Text(UnicodeString::new_nul(&format!("Brush {i}"))))
                .with(
                    "Brsh",
                    Value::Descriptor(
                        Descriptor::new("sampledBrush")
                            .with("Dmtr", Value::UnitFloat { unit: *b"#Pxl", value: 60.0 })
                            .with("Spcn", prc(25.0))
                            .with("sampledData", Value::Text(UnicodeString::new_nul(&format!("$tip-{i}")))),
                    ),
                )
                .with("useTipDynamics", Value::Boolean(true))
                .with("szVr", var(2, 20.0))
                .with("angleDynamics", var(0, 100.0))
                .with("useScatter", Value::Boolean(true))
                .with("scatterDynamics", var(0, 120.0))
                .with("Cnt ", Value::Integer(2))
                .with("usePaintDynamics", Value::Boolean(true))
                .with("opVr", var(2, 0.0))
        })
        .collect();
    let bytes = write_v6(2, &samples, &[], &presets, true).unwrap();
    // `--write <path>`: also save a small showcase file (procedural tips) for UI snapshots.
    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = args.iter().position(|a| a == "--write").and_then(|i| args.get(i + 1)) {
        use photocraft_paint::procedural::*;
        let tips = [("Leafy", leaf_tip(96)), ("Charcoal Stick", charcoal_tip(96, 3)), ("Sea Sponge", sponge_tip(96, 5)), ("Star Burst", star_tip(96))];
        let samples: Vec<AbrSample> = tips
            .iter()
            .enumerate()
            .map(|(i, (_, t))| AbrSample {
                id: format!("$show-{i}"),
                width: t.width,
                height: t.height,
                depth: 16,
                data: t.data.iter().flat_map(|v| v.to_be_bytes()).collect(),
            })
            .collect();
        let presets: Vec<Descriptor> = tips
            .iter()
            .enumerate()
            .map(|(i, (name, _))| {
                let mut d = presets[i].clone();
                d.items.retain(|(k, _)| !k.is("Nm  ") && !k.is("Brsh"));
                d.with("Nm  ", Value::Text(UnicodeString::new_nul(name))).with(
                    "Brsh",
                    Value::Descriptor(
                        Descriptor::new("sampledBrush")
                            .with("Dmtr", Value::UnitFloat { unit: *b"#Pxl", value: 50.0 })
                            .with("Spcn", prc(if i == 1 { 10.0 } else { 45.0 }))
                            .with("sampledData", Value::Text(UnicodeString::new_nul(&format!("$show-{i}")))),
                    ),
                )
            })
            .collect();
        std::fs::write(path, write_v6(2, &samples, &[], &presets, true).unwrap()).unwrap();
        println!("wrote {path}");
    }
    let mut best = f64::MAX;
    let mut imp = None;
    for _ in 0..5 {
        let t = Instant::now();
        imp = Some(photocraft_io::abr_map::read_abr(&bytes, "Bench").unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    let imp = imp.unwrap();
    println!("ABR import: {} brushes, {:.1} MB file, best of 5: {best:.1} ms", imp.presets.len(), bytes.len() as f64 / 1e6);

    // A long pressure-varied stroke with the first imported brush (size, angle, scatter, opacity dynamics).
    let mut brush = imp.presets[0].brush.clone();
    brush.seed = 1;
    let mut s = Surface::new(PixelFormat::RGBA8);
    let size = 6000.0;
    let pts: Vec<StrokePoint> = (0..=2000)
        .map(|i| {
            let t = i as f64 / 2000.0;
            StrokePoint::new(200.0 + t * (size - 400.0), size / 2.0 + (t * 24.0).sin() * 1500.0, (0.3 + 0.7 * (t * 25.0).sin().abs()) as f32)
        })
        .collect();
    let dabs = photocraft_paint::dabs(&photocraft_paint::Stroke { brush: brush.clone(), points: pts.clone() }).len();
    let t = Instant::now();
    render_stroke(&mut s, &brush, &pts, None, false, 1.0);
    println!("Stroke: {dabs} primary dabs (scatter count 2), {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
}
