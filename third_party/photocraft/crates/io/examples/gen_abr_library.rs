//! Generates a large, redistributable ABR v6 brush library for memory and import testing (#1843).
//!
//! Real painting packs are commercial or "personal use only", so we can't test against them. This
//! writes a deterministic stand-in with the same shape: hundreds of presets with big 16-bit sampled
//! tips (RLE-compressed, like Photoshop CC), full dynamics descriptors, split over several files.
//! Every tip comes from `photocraft_paint::procedural` (original, seeded generators), so the output
//! carries no third-party rights. Nothing it writes is committed; regenerate it when needed.
//!
//! `cargo run --release -p photocraft-io --example gen_abr_library -- --out <dir>`
//!
//! Options (defaults in brackets): `--brushes N` [600], `--edge PX` largest tip edge [2000],
//! `--min-edge PX` smallest tip edge [800], `--depth 8|16` [16], `--per-file N` presets per file [100],
//! `--raw` store tips uncompressed (bigger files). Each file stays under the reader's 512 MB decoded cap.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use photocraft_paint::procedural::*;
use photocraft_paint::tile::GrayTile;
use photocraft_psd::abr::{AbrSample, MAX_TOTAL_BYTES, write_v6};
use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value};
use std::path::PathBuf;
use std::time::Instant;

const KINDS: [&str; 9] = ["Chalk", "Spatter", "Bristle", "Charcoal", "Leaf", "Grass", "Sponge", "Star", "Rake"];

fn tip(kind: usize, n: u32, seed: u32) -> GrayTile {
    match kind {
        0 => chalk_tip(n, seed),
        1 => spatter_tip(n, seed, 20 + seed % 40),
        2 => bristle_tip(n, seed),
        3 => charcoal_tip(n, seed),
        4 => leaf_tip(n),
        5 => grass_tip(n, seed, 8 + seed % 16),
        6 => sponge_tip(n, seed),
        7 => star_tip(n),
        _ => rake_tip(n, seed, 6 + seed % 12),
    }
}

fn prc(v: f64) -> Value {
    Value::UnitFloat { unit: *b"#Prc", value: v }
}

fn var(control: i32, jitter: f64) -> Value {
    Value::Descriptor(Descriptor::new("brVr").with("bVTy", Value::Integer(control)).with("jitter", prc(jitter)))
}

fn text(s: &str) -> Value {
    Value::Text(UnicodeString::new_nul(s))
}

/// A cheap deterministic 0..1 value per (brush, salt), to vary the presets.
fn h(i: u32, salt: u32) -> f64 {
    let mut x = (u64::from(i) << 32 | u64::from(salt)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x % 10_000) as f64 / 10_000.0
}

fn preset(i: u32, id: &str, name: &str, edge: u32) -> Descriptor {
    let mut d = Descriptor::new("brushPreset").with("Nm  ", text(name)).with(
        "Brsh",
        Value::Descriptor(
            Descriptor::new("sampledBrush")
                .with("Dmtr", Value::UnitFloat { unit: *b"#Pxl", value: f64::from(edge) * (0.1 + 0.4 * h(i, 1)) })
                .with("Angl", Value::UnitFloat { unit: *b"#Ang", value: (h(i, 2) * 360.0 - 180.0).round() })
                .with("Rndn", prc((50.0 + 50.0 * h(i, 3)).round()))
                .with("Spcn", prc((5.0 + 60.0 * h(i, 4)).round()))
                .with("Intr", Value::Boolean(true))
                .with("flipX", Value::Boolean(h(i, 5) < 0.2))
                .with("sampledData", text(id)),
        ),
    );
    d = d.with("useTipDynamics", Value::Boolean(true)).with("szVr", var(2, (h(i, 6) * 40.0).round())).with("angleDynamics", var(0, (h(i, 7) * 100.0).round()));
    if h(i, 8) < 0.5 {
        d = d
            .with("useScatter", Value::Boolean(true))
            .with("scatterDynamics", var(0, (h(i, 9) * 200.0).round()))
            .with("Cnt ", Value::Integer(1 + (h(i, 10) * 4.0) as i32));
    }
    if h(i, 11) < 0.6 {
        d = d.with("usePaintDynamics", Value::Boolean(true)).with("opVr", var(2, 0.0)).with("prVr", var(2, 0.0));
    }
    d
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let num = |name: &str, default: u32| arg(name).map_or(default, |v| v.parse().expect("numeric option"));
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "target/abr-library".into()));
    let brushes = num("--brushes", 600);
    let max_edge = num("--edge", 2000).clamp(16, photocraft_psd::abr::MAX_EDGE);
    let min_edge = num("--min-edge", 800).clamp(16, max_edge);
    let depth = num("--depth", 16) as u16;
    assert!(matches!(depth, 8 | 16), "--depth must be 8 or 16");
    let per_file = num("--per-file", 100).max(1);
    let rle = !args.iter().any(|a| a == "--raw");
    std::fs::create_dir_all(&out).unwrap();

    let started = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (mut file_no, mut first, mut total_bytes, mut total_decoded) = (0u32, 0u32, 0u64, 0u64);
    while first < brushes {
        // Fill one file up to `per_file` presets, keeping its decoded samples well under the reader's cap.
        let mut ids = Vec::new();
        let mut decoded = 0usize;
        let mut i = first;
        while i < brushes && ids.len() < per_file as usize {
            let edge = min_edge + ((max_edge - min_edge) as f64 * h(i, 0)) as u32;
            let bytes = edge as usize * edge as usize * usize::from(depth / 8);
            if !ids.is_empty() && decoded + bytes > MAX_TOTAL_BYTES * 3 / 4 {
                break;
            }
            decoded += bytes;
            ids.push((i, edge));
            i += 1;
        }
        let samples: Vec<AbrSample> = std::thread::scope(|s| {
            let chunk = ids.len().div_ceil(threads);
            let handles: Vec<_> = ids
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || {
                        part.iter()
                            .map(|&(i, edge)| {
                                let t = tip(i as usize % KINDS.len(), edge, i);
                                let data = if depth == 16 {
                                    t.data.iter().flat_map(|v| v.to_be_bytes()).collect()
                                } else {
                                    t.data.iter().map(|v| (v >> 8) as u8).collect()
                                };
                                AbrSample { id: format!("$pc-gen-{i:05}"), width: edge, height: edge, depth, data }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let presets: Vec<Descriptor> = ids
            .iter()
            .map(|&(i, edge)| preset(i, &format!("$pc-gen-{i:05}"), &format!("{} {:03} ({edge} px)", KINDS[i as usize % KINDS.len()], i + 1), edge))
            .collect();
        let bytes = write_v6(2, &samples, &[], &presets, rle).unwrap();
        // Sanity: our own importer must accept every preset.
        let imported = photocraft_io::abr_map::read_abr(&bytes, "Generated").unwrap();
        assert_eq!(imported.presets.len(), ids.len(), "every generated preset imports");
        file_no += 1;
        let path = out.join(format!("photocraft-test-brushes-{file_no:02}.abr"));
        std::fs::write(&path, &bytes).unwrap();
        println!("{}: {} brushes, {:.1} MB on disk, {:.1} MB decoded", path.display(), ids.len(), bytes.len() as f64 / 1e6, decoded as f64 / 1e6);
        total_bytes += bytes.len() as u64;
        total_decoded += decoded as u64;
        first = i;
    }
    println!(
        "{brushes} brushes in {file_no} files: {:.1} MB on disk, {:.1} MB of decoded tips, {:.1} s",
        total_bytes as f64 / 1e6,
        total_decoded as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
}
