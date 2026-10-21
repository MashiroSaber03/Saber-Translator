//! Inspect and time raw decoding.
//!
//! ```text
//! cargo run --release -p photocraft-raw --example rawinfo -- [--dump] [--demosaic bilinear|mhc|ahd] [--ppm out.ppm] FILE...
//! ```
//!
//! `--ppm` writes the developed image (ProPhoto, gamma 1.8, 16-bit) as a binary PPM;
//! `--png` writes an sRGB 8-bit preview at most 1600 pixels wide (for eyeballing).

use std::time::Instant;

use photocraft_raw::{Demosaic, DevelopOptions, decode, develop_sensor, dump_structure, embedded_preview, identify};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dump = false;
    let mut ppm: Option<String> = None;
    let mut png: Option<String> = None;
    let mut opts = DevelopOptions::default();
    let mut files = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dump" => dump = true,
            "--ppm" => ppm = args.next(),
            "--png" => png = args.next(),
            "--demosaic" => opts.demosaic = args.next().and_then(|s| Demosaic::from_id(&s)).unwrap_or_default(),
            "--write-synthetic" => {
                // `--write-synthetic DIR`: write 24 MP synthetic DNG (uncompressed, LJ92 tiles) and CR2 files.
                use photocraft_raw::testgen::{Cr2Spec, DngSpec, DngStorage, mosaic, scene};
                let dir = args.next().unwrap_or_else(|| ".".into());
                let (w, h) = (6000, 4000);
                let data = mosaic(&scene(w, h), w, [0, 1, 1, 2], 512, 15000);
                let mut spec = DngSpec::cfa(w, h, data.clone());
                spec.bits = 14;
                spec.black = vec![512];
                spec.white = 15000;
                spec.as_shot_neutral = Some([0.5, 1.0, 0.7]);
                spec.color_matrix1 = Some((21, [0.9, -0.3, -0.1, -0.4, 1.3, 0.1, -0.1, 0.2, 0.6]));
                let _ = std::fs::write(format!("{dir}/synthetic-uncompressed.dng"), spec.build());
                spec.storage = DngStorage::Lj92Tiles { width: 256, height: 256 };
                let _ = std::fs::write(format!("{dir}/synthetic-lj92.dng"), spec.build());
                let cr2 = Cr2Spec {
                    width: w,
                    height: h,
                    data,
                    precision: 14,
                    components: 4,
                    slices: vec![w / 2, w - w / 2],
                    borders: None,
                    wb_rggb: None,
                    orientation: 1,
                    model_id: None,
                };
                let _ = std::fs::write(format!("{dir}/synthetic.cr2"), cr2.build());
            }
            "--synthetic" => {
                // `--synthetic 6000x4000`: time a generated 14-bit DNG (uncompressed and
                // lossless-JPEG tiled) and a CR2 of that size.
                let dims = args.next().unwrap_or_default();
                let (w, h) = dims.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))).unwrap_or((6000, 4000));
                synthetic(w, h, &opts);
            }
            _ => files.push(a),
        }
    }
    for f in files {
        let bytes = match std::fs::read(&f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{f}: {e}");
                continue;
            }
        };
        println!("== {f}: {:?}", identify(&bytes));
        if dump {
            println!("{}", dump_structure(&bytes));
        }
        if let Some(p) = embedded_preview(&bytes) {
            println!("   preview: {}x{} JPEG, {} bytes", p.width, p.height, p.jpeg.len());
            if let Some(dir) = png.as_ref().filter(|d| std::path::Path::new(d).is_dir()) {
                let stem = std::path::Path::new(&f).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let _ = std::fs::write(format!("{dir}/{stem}.preview.jpg"), p.jpeg);
            }
        }
        let t0 = Instant::now();
        let sensor = match decode(&bytes, &opts.limits) {
            Ok(s) => s,
            Err(e) => {
                println!("   decode: {e}");
                continue;
            }
        };
        let t1 = Instant::now();
        if std::env::var_os("RAWINFO_COLUMNS").is_some() {
            // Mean of each of the first 200 columns over the middle half of the rows.
            let (w, h) = (sensor.width, sensor.height);
            let means: Vec<String> = (0..200.min(w))
                .map(|x| {
                    let rows = h / 4..3 * h / 4;
                    let n = rows.len().max(1);
                    let s: u64 = rows.map(|y| u64::from(sensor.data[y * w + x])).sum();
                    format!("{x}:{}", s / n as u64)
                })
                .collect();
            println!("   columns {}", means.join(" "));
        }
        println!(
            "   sensor {}x{}x{} cfa {:?} active {:?} crop {:?} black {:?} white {:?} wb {:?} neutral {:?} orient {} BE {}",
            sensor.width,
            sensor.height,
            sensor.samples,
            sensor.cfa.as_ref().map(|c| c.phase(sensor.crop.x, sensor.crop.y)),
            sensor.active,
            sensor.crop,
            &sensor.black.values[..sensor.black.values.len().min(4)],
            sensor.white,
            sensor.camera_wb,
            sensor.color.as_shot_neutral,
            sensor.orientation,
            sensor.baseline_exposure
        );
        let dev = match develop_sensor(&sensor, &opts) {
            Ok(d) => d,
            Err(e) => {
                println!("   develop: {e}");
                continue;
            }
        };
        let t2 = Instant::now();
        println!(
            "   developed {}x{} ({:?}) decode {:.0} ms, develop {:.0} ms; wb {:?}; warnings {:?}",
            dev.width,
            dev.height,
            opts.demosaic,
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            dev.info.wb_multipliers,
            dev.warnings
        );
        if let Some(p) = &png {
            // A directory gets one `<file stem>.png` per input.
            let path = if std::path::Path::new(p).is_dir() {
                let stem = std::path::Path::new(&f).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                format!("{p}/{stem}.png")
            } else {
                p.clone()
            };
            write_png(&path, &dev);
        }
        if let Some(p) = &ppm {
            let mut out = format!("P6\n{} {}\n65535\n", dev.width, dev.height).into_bytes();
            for v in &dev.rgb {
                out.extend_from_slice(&v.to_be_bytes());
            }
            if let Err(e) = std::fs::write(p, out) {
                eprintln!("{p}: {e}");
            }
        }
    }
}

fn synthetic(w: usize, h: usize, opts: &DevelopOptions) {
    use photocraft_raw::testgen::{Cr2Spec, DngSpec, DngStorage, mosaic, scene};
    let data = mosaic(&scene(w, h), w, [0, 1, 1, 2], 512, 15000);
    let mut spec = DngSpec::cfa(w, h, data.clone());
    spec.bits = 14;
    spec.black = vec![512];
    spec.white = 15000;
    spec.as_shot_neutral = Some([0.5, 1.0, 0.7]);
    spec.color_matrix1 = Some((21, [0.9, -0.3, -0.1, -0.4, 1.3, 0.1, -0.1, 0.2, 0.6]));
    let cr2 = Cr2Spec {
        width: w,
        height: h,
        data,
        precision: 14,
        components: 4,
        slices: vec![w / 2, w - w / 2],
        borders: None,
        wb_rggb: None,
        orientation: 1,
        model_id: None,
    };
    let files = [
        ("DNG uncompressed", spec.build()),
        ("DNG LJ92 256x256 tiles", {
            spec.storage = DngStorage::Lj92Tiles { width: 256, height: 256 };
            spec.build()
        }),
        ("CR2 (one LJ92 stream)", cr2.build()),
    ];
    for (name, b) in files {
        for m in [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd] {
            let t0 = Instant::now();
            let s = decode(&b, &opts.limits).expect("synthetic decode");
            let t1 = Instant::now();
            let d = develop_sensor(&s, &DevelopOptions { demosaic: m, ..opts.clone() }).expect("synthetic develop");
            let t2 = Instant::now();
            println!(
                "{name:24} {w}x{h} {:8}: decode {:6.0} ms + develop {:6.0} ms = {:6.0} ms ({} px out)",
                m.id(),
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
                (t2 - t0).as_secs_f64() * 1e3,
                d.rgb.len() / 3
            );
        }
    }
}

/// ProPhoto (gamma 1.8) → sRGB 8-bit, box-downscaled to at most 1600 px wide.
fn write_png(path: &str, dev: &photocraft_raw::Developed) {
    // Linear ProPhoto → XYZ D50 → linear sRGB (Bradford-adapted D50 → D65), combined.
    const M: [[f32; 3]; 3] = [[2.0341, -0.7276, -0.3065], [-0.2289, 1.2317, -0.0028], [-0.0086, -0.1534, 1.1620]];
    let (w, h) = (dev.width as usize, dev.height as usize);
    // RAWINFO_CROP="x,y,w,h" writes that region at 100 % instead of a downscaled whole.
    let crop: Option<Vec<usize>> = std::env::var("RAWINFO_CROP").ok().map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect());
    let (x0, y0, k, ow, oh) = match crop.as_deref() {
        Some(&[x, y, cw, ch]) if x + cw <= w && y + ch <= h => (x, y, 1, cw, ch),
        _ => {
            let k = w.div_ceil(1600).max(1);
            (0, 0, k, w / k, h / k)
        }
    };
    let mut out = Vec::with_capacity(ow * oh * 3);
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0.0f32; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let i = ((y0 + oy * k + dy) * w + x0 + ox * k + dx) * 3;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += (f32::from(dev.rgb[i + c]) / 65535.0).powf(1.8);
                    }
                }
            }
            let lin = acc.map(|v| v / (k * k) as f32);
            for row in M {
                let v = (row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2]).clamp(0.0, 1.0);
                let e = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
                out.push((e * 255.0 + 0.5) as u8);
            }
        }
    }
    let Ok(f) = std::fs::File::create(path) else { return eprintln!("{path}: cannot create") };
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), ow as u32, oh as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    if let Err(e) = enc.write_header().and_then(|mut w| w.write_image_data(&out)) {
        eprintln!("{path}: {e}");
    }
}
