//! Mutation stress test: corrupt real or synthetic raw files at random and
//! check that decoding and developing never panics.
//!
//! ```text
//! cargo run --release -p photocraft-raw --example mutate -- ITERATIONS FILE...
//! ```

use photocraft_raw::{Demosaic, DevelopOptions, Limits, decode, develop_sensor, embedded_preview, identify};

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let limits = Limits { max_width: 16384, max_height: 16384, max_pixels: 1 << 26, max_alloc: 1 << 30 };
    let mut seed = 0x5EED_u64;
    let mut rng = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    let mut panics = 0;
    for f in args {
        let Ok(orig) = std::fs::read(&f) else { continue };
        for it in 0..n {
            let mut b = orig.clone();
            match rng() % 4 {
                0 => b.truncate(rng() as usize % b.len().max(1)),
                1 => {
                    // Header / IFD region.
                    let span = b.len().clamp(1, 1 << 16);
                    for _ in 0..1 + rng() % 8 {
                        let at = rng() as usize % span;
                        b[at] = rng() as u8;
                    }
                }
                _ => {
                    for _ in 0..1 + rng() % 16 {
                        let at = rng() as usize % b.len().max(1);
                        b[at] = rng() as u8;
                    }
                }
            }
            let r = std::panic::catch_unwind(|| {
                let _ = identify(&b);
                let _ = embedded_preview(&b);
                if let Ok(s) = decode(&b, &limits) {
                    let _ = develop_sensor(&s, &DevelopOptions { demosaic: Demosaic::Bilinear, limits, ..Default::default() });
                }
            });
            if r.is_err() {
                panics += 1;
                let path = format!("{f}.panic-{it}");
                eprintln!("PANIC on mutation {it} of {f}; saved {path}");
                let _ = std::fs::write(path, &b);
            }
        }
        println!("{f}: {n} mutations done");
    }
    println!("panics: {panics}");
}
