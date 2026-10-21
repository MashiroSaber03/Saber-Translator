//! Opens the files written by `photocraft-psd`'s `big_psb` example and checks their pixels.
//!
//! `cargo run --release -p photocraft-io --example big_psb_check -- <file.psb>...`

fn sample(x: u32, y: u32, c: u32, rle: bool) -> u8 {
    if c == 3 {
        return 255;
    }
    let x = if rle { 0 } else { x };
    (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)).wrapping_add(c.wrapping_mul(101)) & 0xff) as u8
}

fn main() {
    for path in std::env::args().skip(1) {
        let rle = path.contains("rle");
        let t = std::time::Instant::now();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                println!("{path}: read failed: {e}");
                continue;
            }
        };
        let read_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let res = match photocraft_io::import(&path, &bytes) {
            Ok(r) => r,
            Err(e) => {
                println!("{path}: import failed: {e}");
                continue;
            }
        };
        drop(bytes);
        // With SAVE set: save as PSB, then check the re-opened file instead.
        let res = if std::env::var_os("SAVE").is_some() {
            let t = std::time::Instant::now();
            let out = match photocraft_io::export(&res.document, "x.psb", &photocraft_io::ExportOptions::default()) {
                Ok(o) => o,
                Err(e) => {
                    println!("{path}: save failed: {e}");
                    continue;
                }
            };
            println!("{path}: saved {:.2} GB in {:.1} s, warnings {:?}", out.bytes.len() as f64 / 1e9, t.elapsed().as_secs_f64(), out.warnings);
            drop(res);
            match photocraft_io::import("x.psb", &out.bytes) {
                Ok(r) => r,
                Err(e) => {
                    println!("{path}: re-open failed: {e}");
                    continue;
                }
            }
        } else {
            res
        };
        let doc = &res.document;
        println!(
            "{path}: read {read_s:.1} s, import {:.1} s, {}x{}, {} layer(s)",
            t.elapsed().as_secs_f64(),
            doc.size.width,
            doc.size.height,
            doc.layers.len()
        );
        for w in &res.warnings {
            println!("  warning: {w}");
        }
        let Some(s) = doc.layers.first().and_then(|l| l.surface()) else {
            println!("  FAIL: no pixel layer");
            continue;
        };
        let (w, h) = (doc.size.width, doc.size.height);
        let mut bad = 0;
        let points = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w / 2, h / 2), (w / 3, h * 2 / 3), (12_345 % w, (h - 7) % h)];
        for &(x, y) in &points {
            let got = s.pixel(x as i32, y as i32);
            let want: Vec<f32> = (0..4).map(|c| f32::from(sample(x, y, c, rle)) / 255.0).collect();
            if got.len() != 4 || got.iter().zip(&want).any(|(g, w)| (g - w).abs() > 1.0 / 510.0) {
                bad += 1;
                println!("  mismatch at ({x}, {y}): got {got:?}, want {want:?}");
            }
        }
        println!("  {}", if bad == 0 { "OK: all sampled pixels match" } else { "FAIL" });
    }
}
