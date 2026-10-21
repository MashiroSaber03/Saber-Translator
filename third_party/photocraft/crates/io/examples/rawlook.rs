//! Measure the camera-JPEG colour fit on real raw files: times the fit, compares the
//! neutral fallback and the fitted look with the camera's embedded JPEG (mean CIE76 ΔE in
//! sRGB-clipped Lab D50, measured on the full develop output against the largest preview,
//! independently of the fit's own grid) and can write 1200-px sRGB PNGs of both renderings
//! and of the JPEG.
//!
//! ```sh
//! cargo run --release -p photocraft-io --example rawlook -- [--png DIR] FILE...
//! cargo run --release -p photocraft-io --example rawlook -- --synthetic   # time a 24 MP synthetic NEF
//! ```

use std::time::Instant;

use photocraft_cms::{Builtin, Intent, Transform};
use photocraft_codecs::{self as codecs, ChannelLayout, EncodeOptions, Format, Image};
use photocraft_raw::{DevelopOptions, develop_sensor};

/// Box-downsamples interleaved RGB to at most `edge` pixels on the long side.
fn shrink(px: &[f32], w: usize, h: usize, edge: usize) -> (Vec<f32>, usize, usize) {
    let f = w.max(h).div_ceil(edge).max(1);
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![0.0f32; ow * oh * 3];
    for y in 0..oh * f {
        for x in 0..ow * f {
            let (i, o) = ((y * w + x) * 3, ((y / f) * ow + x / f) * 3);
            for k in 0..3 {
                out[o + k] += px[i + k];
            }
        }
    }
    let n = (f * f) as f32;
    out.iter_mut().for_each(|v| *v /= n);
    (out, ow, oh)
}

/// Resamples to exactly `tw × th` by box averaging over the matching source area.
fn resample(px: &[f32], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tw * th * 3];
    for ty in 0..th {
        let (y0, y1) = (ty * h / th, ((ty + 1) * h / th).max(ty * h / th + 1));
        for tx in 0..tw {
            let (x0, x1) = (tx * w / tw, ((tx + 1) * w / tw).max(tx * w / tw + 1));
            let mut s = [0.0f32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    for k in 0..3 {
                        s[k] += px[(y * w + x) * 3 + k];
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f32;
            for k in 0..3 {
                out[(ty * tw + tx) * 3 + k] = s[k] / n;
            }
        }
    }
    out
}

fn lin(v: f32) -> f64 {
    let v = f64::from(v);
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Lab (D50) of display sRGB 0..1 (Bradford-adapted sRGB → XYZ D50).
fn lab(p: &[f32]) -> [f64; 3] {
    let m = [[0.4360747, 0.3850649, 0.1430804], [0.2225045, 0.7168786, 0.0606169], [0.0139322, 0.0971045, 0.7141733]];
    let l = [lin(p[0]), lin(p[1]), lin(p[2])];
    let xyz: [f64; 3] = std::array::from_fn(|i| m[i][0] * l[0] + m[i][1] * l[1] + m[i][2] * l[2]);
    let w = [0.9642, 1.0, 0.8249];
    let f = |t: f64| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
    let [fx, fy, fz] = [0, 1, 2].map(|k| f(xyz[k] / w[k]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn mean_de(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() / 3;
    a.as_chunks::<3>()
        .0
        .iter()
        .zip(b.as_chunks::<3>().0)
        .map(|(p, q)| {
            let (x, y) = (lab(p), lab(q));
            ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
        })
        .sum::<f64>()
        / n as f64
}

/// Developed ProPhoto 16-bit → display sRGB 0..1.
fn to_srgb(rgb: &[u16]) -> Vec<f32> {
    let t = Transform::new(Builtin::ProPhotoCompat.profile(), Builtin::Srgb.profile(), Intent::RelativeColorimetric, false).expect("transform");
    let mut out = vec![0u16; rgb.len()];
    t.convert_u16(rgb, 3, &mut out, 3, false);
    out.iter().map(|v| f32::from(*v) / 65535.0).collect()
}

fn write_png(path: &str, px: &[f32], w: usize, h: usize) {
    let (s, sw, sh) = shrink(px, w, h, 1200);
    let bytes: Vec<u8> = s.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    let img = Image::from_u8(sw as u32, sh as u32, ChannelLayout::Rgb, bytes).expect("image");
    std::fs::write(path, codecs::encode(&img, Format::Png, &EncodeOptions::default()).expect("png")).expect("write");
}

/// Times the fit on a synthetic 6034×4012 NEF-like raw with a 1632×1086 camera-style preview.
fn synthetic() {
    use photocraft_raw::testgen::{TiffBuilder, Val, mosaic};
    let (w, h) = (6034usize, 4012usize);
    let colours: Vec<[f32; 3]> = (0..96u32).map(|i| [0.1 + 0.05 * (i % 9) as f32, 0.1 + 0.04 * (i % 11) as f32, 0.1 + 0.06 * (i % 7) as f32]).collect();
    let at = |x: usize, y: usize| {
        let ramp = 0.15 + 0.85 * ((x * 12) % w) as f32 / w as f32;
        colours[(y * 8 / h) * 12 + x * 12 / w].map(|v| v * ramp)
    };
    let cam: Vec<[f32; 3]> = (0..w * h).map(|i| at(i % w, i / w)).collect();
    let data = mosaic(&cam, w, [0, 1, 1, 2], 0, 4095);
    drop(cam);
    let (pw, ph) = (1632usize, 1086usize);
    let px: Vec<u8> = (0..pw * ph)
        .flat_map(|i| {
            let c = at((i % pw) * w / pw, (i / pw) * h / ph);
            c.map(|v| {
                let v = (1.6 * v / (1.0 + 0.6 * v)).clamp(0.0, 1.0);
                ((if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }) * 255.0).round() as u8
            })
        })
        .collect();
    let jpeg =
        codecs::encode(&Image::from_u8(pw as u32, ph as u32, ChannelLayout::Rgb, px).expect("image"), Format::Jpeg, &EncodeOptions::default()).expect("jpeg");
    let mut t = TiffBuilder::default();
    let strip = t.blob(data.iter().flat_map(|v| v.to_le_bytes()).collect());
    let raw = t.ifd(vec![
        (256, Val::Long(vec![w as u32])),
        (257, Val::Long(vec![h as u32])),
        (258, Val::Short(vec![12])),
        (259, Val::Short(vec![1])),
        (262, Val::Short(vec![32803])),
        (273, Val::Blobs(vec![strip])),
        (277, Val::Short(vec![1])),
        (278, Val::Long(vec![h as u32])),
        (279, Val::Long(vec![(data.len() * 2) as u32])),
        (33421, Val::Short(vec![2, 2])),
        (33422, Val::Byte(vec![0, 1, 1, 2])),
    ]);
    let len = jpeg.len() as u32;
    let jb = t.blob(jpeg);
    let ifd0 =
        t.ifd(vec![(271, Val::Ascii("NIKON CORPORATION".into())), (330, Val::Ifds(vec![raw])), (513, Val::Blobs(vec![jb])), (514, Val::Long(vec![len]))]);
    t.chain = vec![ifd0];
    let bytes = t.build();
    let opts = DevelopOptions::default();
    let sensor = photocraft_raw::decode(&bytes, &opts.limits).expect("decode");
    let _ = photocraft_io::raw::fit_preview_look(&bytes, &sensor, &opts);
    let mut times: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            let r = photocraft_io::raw::fit_preview_look(&bytes, &sensor, &opts);
            assert!(r.is_ok(), "{r:?}");
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    let p = photocraft_raw::embedded_preview(&bytes).expect("preview");
    let t = Instant::now();
    let img = codecs::decode_as(Format::Jpeg, p.jpeg).expect("jpeg");
    let dec = t.elapsed().as_secs_f64() * 1e3;
    let r = photocraft_raw::Reference { width: img.width(), height: img.height(), rgb: img.data() };
    let t = Instant::now();
    let _ = photocraft_raw::fit_look(&sensor, &opts, &r);
    println!("  preview decode {dec:.1} ms, fit_look {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    println!("synthetic {w}x{h}: preview decode + fit median {:.1} ms (min {:.1}, max {:.1})", times[3], times[0], times[6]);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut png: Option<String> = None;
    let mut files = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--png" => png = args.next(),
            "--synthetic" => {
                synthetic();
                return;
            }
            _ => files.push(a),
        }
    }
    println!("| file | fit | ΔE grid fallback → fit | ΔE full fallback → fit | fit ms | develop ms |");
    println!("|---|---|---|---|---|---|");
    for f in files {
        let bytes = std::fs::read(&f).expect("read");
        let name = std::path::Path::new(&f).file_stem().and_then(|s| s.to_str()).unwrap_or("raw").to_string();
        let opts = DevelopOptions::default();
        let sensor = photocraft_raw::decode(&bytes, &opts.limits).expect("decode");
        // Warm up (thread pool), then time the fit.
        let _ = photocraft_io::raw::fit_preview_look(&bytes, &sensor, &opts);
        let t = Instant::now();
        let fit = photocraft_io::raw::fit_preview_look(&bytes, &sensor, &opts);
        let fit_ms = t.elapsed().as_secs_f64() * 1e3;
        let before = develop_sensor(&sensor, &opts).expect("develop");
        let t = Instant::now();
        let after = match &fit {
            Ok(look) => Some(develop_sensor(&sensor, &DevelopOptions { camera_look: Some(look.clone()), ..opts.clone() }).expect("develop")),
            Err(_) => None,
        };
        let dev_ms = t.elapsed().as_secs_f64() * 1e3;
        // Independent check against the largest preview.
        let p = photocraft_raw::embedded_preview(&bytes).expect("preview");
        let raw_jpeg = codecs::decode_as_with(Format::Jpeg, p.jpeg, &codecs::DecodeOptions { keep_orientation: true, ..Default::default() }).expect("jpeg");
        let own = raw_jpeg.meta.exif.as_deref().map_or(1, codecs::exif_orientation);
        let o = if own != 1 { own } else { codecs::exif_orientation(&bytes) };
        let jpeg = raw_jpeg.oriented(o).expect("orient").converted(ChannelLayout::Rgb, photocraft_codecs::SampleType::U8).into_owned();
        let (jw, jh) = (jpeg.width() as usize, jpeg.height() as usize);
        let jpx: Vec<f32> = jpeg.data().iter().map(|v| f32::from(*v) / 255.0).collect();
        let (dw, dh) = (before.width as usize, before.height as usize);
        let (gw, gh) = (300, (300 * dh).div_ceil(dw));
        let jref = resample(&jpx, jw, jh, gw, gh);
        let b = to_srgb(&before.rgb);
        let de_before = mean_de(&resample(&b, dw, dh, gw, gh), &jref);
        let a = after.as_ref().map(|d| to_srgb(&d.rgb));
        let de_after = a.as_ref().map(|a| mean_de(&resample(a, dw, dh, gw, gh), &jref));
        let (verdict, grid) = match &fit {
            Ok(l) => ("used".to_string(), format!("{:.1} → {:.1}", l.delta_e_fallback, l.delta_e)),
            Err(e) => (format!("rejected: {e}"), "-".into()),
        };
        println!("| {name} | {verdict} | {grid} | {de_before:.1} → {} | {fit_ms:.0} | {dev_ms:.0} |", de_after.map_or("-".into(), |v| format!("{v:.1}")));
        if let Ok(l) = &fit {
            eprintln!("{name}: to_xyz {:?}\n  tone {:?}", l.to_xyz, l.tone.as_ref().map(|t| t.knots()));
        }
        if let Some(dir) = &png {
            write_png(&format!("{dir}/{name}-before.png"), &b, dw, dh);
            if let Some(a) = &a {
                write_png(&format!("{dir}/{name}-after.png"), a, dw, dh);
            }
            write_png(&format!("{dir}/{name}-camera-jpeg.png"), &jpx, jw, jh);
        }
    }
}
