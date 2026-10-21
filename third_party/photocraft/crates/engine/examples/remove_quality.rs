//! Quality of the Remove Tool's fill (`photocraft_algo::remove::remove_with`) on real photos.
//! `cargo run --release -p photocraft-engine --example remove_quality -- <images dir> [out dir]`
//!
//! The images are `01.jpg`, `02.jpg`, `03.jpg`, `04.jpg` and `06.jpg`: public-domain photos from
//! Wikimedia Commons, not committed (the list is in `docs/development.md` › Remove Tool quality).
//! Each gets holes over background (discs and thin lines, placed in fractions of the image size)
//! that are filled through a window like the command's. Measured in the hole against the original:
//!
//! * `psnr`: fidelity in dB (a plausible fill need not match, so read it as a trend);
//! * `texture`: mean gradient magnitude of the fill over the original's (1 is ideal; below 1 the
//!   fill is smoother than what it replaces);
//! * `seam`: the same on the band along the hole's edge (above 1 the edge shows).
//!
//! With an out dir, each case is also written as `<image>-<case>.png`: original | hole | fill.
use std::time::Instant;

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType};
use photocraft_raster::Interrupt;

/// A hole: a disc (centre in fractions of the width and height, radius in fractions of the width)
/// or a line (ends in fractions, width in pixels).
#[derive(Clone, Copy)]
enum Shape {
    Disc { c: (f64, f64), r: f64 },
    Line { a: (f64, f64), b: (f64, f64), width: f64 },
}

fn cases(image: &str) -> Vec<(&'static str, Shape)> {
    use Shape::{Disc, Line};
    match image {
        // The Tetons and the Snake River: forest, river bend, ridge, clouds.
        "01" => vec![
            ("forest", Disc { c: (0.75, 0.62), r: 0.03 }),
            ("river-bend", Disc { c: (0.45, 0.72), r: 0.045 }),
            ("ridge", Disc { c: (0.5, 0.28), r: 0.035 }),
            ("clouds", Disc { c: (0.25, 0.12), r: 0.06 }),
            ("wire", Line { a: (0.3, 0.55), b: (0.7, 0.85), width: 5.0 }),
        ],
        // A dam in a canyon: rock, water, the dam's wall.
        "02" => vec![
            ("rock", Disc { c: (0.85, 0.45), r: 0.04 }),
            ("water", Disc { c: (0.6, 0.7), r: 0.035 }),
            ("dam-wall", Disc { c: (0.35, 0.25), r: 0.025 }),
            ("wire", Line { a: (0.05, 0.6), b: (0.95, 0.5), width: 4.0 }),
        ],
        // Oak trees by a gravel path: grass, gravel, a trunk, branches.
        "03" => vec![
            ("grass", Disc { c: (0.75, 0.85), r: 0.045 }),
            ("gravel", Disc { c: (0.25, 0.85), r: 0.035 }),
            ("trunk", Disc { c: (0.48, 0.45), r: 0.02 }),
            ("branches", Disc { c: (0.8, 0.2), r: 0.04 }),
        ],
        // A meadow anemone in grass, shallow depth of field.
        "04" => vec![
            ("bokeh", Disc { c: (0.2, 0.35), r: 0.05 }),
            ("stems", Disc { c: (0.85, 0.6), r: 0.04 }),
            ("wire", Line { a: (0.1, 0.1), b: (0.9, 0.3), width: 6.0 }),
        ],
        // A coast from above: surf, sand, mountains.
        "06" => vec![
            ("surf", Disc { c: (0.75, 0.85), r: 0.045 }),
            ("sand", Disc { c: (0.3, 0.75), r: 0.04 }),
            ("mountains", Disc { c: (0.5, 0.12), r: 0.03 }),
            ("wire", Line { a: (0.0, 0.55), b: (1.0, 0.65), width: 5.0 }),
        ],
        _ => Vec::new(),
    }
}

/// The hole of `shape` on a `w × h` image.
fn mask(shape: Shape, w: usize, h: usize) -> Vec<bool> {
    let (wf, hf) = (w as f64, h as f64);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            match shape {
                Shape::Disc { c, r } => (x - c.0 * wf).hypot(y - c.1 * hf) < r * wf,
                Shape::Line { a, b, width } => {
                    let (ax, ay, bx, by) = (a.0 * wf, a.1 * hf, b.0 * wf, b.1 * hf);
                    let (dx, dy) = (bx - ax, by - ay);
                    let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                    (x - ax - t * dx).hypot(y - ay - t * dy) < width / 2.0
                }
            }
        })
        .collect()
}

/// Bounding box `(x0, y0, x1, y1)` of the set cells.
fn bbox(w: usize, m: &[bool]) -> Option<(usize, usize, usize, usize)> {
    let mut b: Option<(usize, usize, usize, usize)> = None;
    for (i, _) in m.iter().enumerate().filter(|(_, v)| **v) {
        let (x, y) = (i % w, i / w);
        b = Some(b.map_or((x, y, x + 1, y + 1), |(x0, y0, x1, y1)| (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1))));
    }
    b
}

/// A `(x0, y0, x1, y1)` crop of an interleaved buffer.
fn crop(src: &[f32], w: usize, ch: usize, r: (usize, usize, usize, usize)) -> Vec<f32> {
    (r.1..r.3).flat_map(|y| src[(y * w + r.0) * ch..(y * w + r.2) * ch].iter().copied()).collect()
}

/// Gradient magnitude (forward differences, summed over channels) at every cell.
fn gradient(img: &[f32], w: usize, h: usize, ch: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (r, d) = ((y * w + (x + 1).min(w - 1)) * ch, ((y + 1).min(h - 1) * w + x) * ch);
            (0..ch).map(|c| (img[r + c] - img[i * ch + c]).abs() + (img[d + c] - img[i * ch + c]).abs()).sum()
        })
        .collect()
}

fn mean_over(v: &[f32], m: &[bool]) -> f64 {
    let (s, n) = v.iter().zip(m).filter(|(_, m)| **m).fold((0.0f64, 0usize), |(s, n), (v, _)| (s + f64::from(*v), n + 1));
    if n == 0 { 0.0 } else { s / n as f64 }
}

struct Score {
    psnr: f64,
    texture: f64,
    seam: f64,
}

fn score(orig: &[f32], out: &[f32], hole: &[bool], w: usize, h: usize, ch: usize) -> Score {
    let (mut se, mut n) = (0.0f64, 0usize);
    for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
        for c in 0..ch {
            se += f64::from(out[i * ch + c] - orig[i * ch + c]).powi(2);
            n += 1;
        }
    }
    let psnr = if se == 0.0 { 99.0 } else { 10.0 * (n as f64 / se).log10() };
    let (go, gf) = (gradient(orig, w, h, ch), gradient(out, w, h, ch));
    // The edge band: cells within 2 px of the hole's boundary, on both sides.
    let grown = photocraft_algo::remove::dilate(w, h, hole, 2);
    let inner: Vec<bool> = {
        let outside: Vec<bool> = hole.iter().map(|m| !*m).collect();
        photocraft_algo::remove::dilate(w, h, &outside, 2).iter().map(|o| !*o).collect()
    };
    let band: Vec<bool> = grown.iter().zip(&inner).map(|(g, i)| *g && !*i).collect();
    let ratio = |m: &[bool]| {
        let o = mean_over(&go, m);
        if o > 0.0 { mean_over(&gf, m) / o } else { 1.0 }
    };
    Score { psnr, texture: ratio(hole), seam: ratio(&band) }
}

fn write_png(path: &std::path::Path, data: &[f32], w: usize, h: usize, ch: usize) {
    let layout = if ch == 1 { ChannelLayout::Gray } else { ChannelLayout::Rgb };
    let img = Image::from_normalized(w as u32, h as u32, layout, SampleType::U8, data).expect("image");
    std::fs::write(path, photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).expect("encode")).expect("write");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(args.get(1).expect("usage: remove_quality <images dir> [out dir]"));
    let out_dir = args.get(2).map(std::path::PathBuf::from);
    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d).expect("out dir");
    }
    println!("{:<4} {:<11} {:>7} {:>8} {:>6} {:>9}", "img", "case", "psnr", "texture", "seam", "ms");
    let (mut sum_psnr, mut sum_tex, mut sum_seam, mut sum_ms, mut count) = (0.0, 0.0, 0.0, 0.0, 0);
    for name in ["01", "02", "03", "04", "06"] {
        let Ok(bytes) = std::fs::read(dir.join(format!("{name}.jpg"))) else {
            println!("{name}: missing, skipped");
            continue;
        };
        let img = photocraft_codecs::decode(&bytes).expect("decode");
        let (w, h) = (img.width() as usize, img.height() as usize);
        let ch = img.layout().channels();
        let px = img.to_normalized();
        for (case, shape) in cases(name) {
            let full = mask(shape, w, h);
            let Some(b) = bbox(w, &full) else { continue };
            // The command's window: the hole plus three quarters of its extent (32..512 px).
            let ext = (b.2 - b.0).max(b.3 - b.1);
            let m = (ext * 3 / 4).clamp(32, 512);
            let win = (b.0.saturating_sub(m), b.1.saturating_sub(m), (b.2 + m).min(w), (b.3 + m).min(h));
            let (ww, wh) = (win.2 - win.0, win.3 - win.1);
            let orig = crop(&px, w, ch, win);
            let hole: Vec<bool> = (win.1..win.3).flat_map(|y| (win.0..win.2).map(move |x| (x, y))).map(|(x, y)| full[y * w + x]).collect();
            let mut holed = orig.clone();
            for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
                holed[i * ch..(i + 1) * ch].fill(1.0);
            }
            let t = Instant::now();
            let out = photocraft_algo::remove::remove_with(ww, wh, ch, &holed, &hole, 0x0052_e30e, &Interrupt::NONE).expect("never cancelled");
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let s = score(&orig, &out, &hole, ww, wh, ch);
            println!("{name:<4} {case:<11} {:>7.2} {:>8.3} {:>6.3} {ms:>9.1}", s.psnr, s.texture, s.seam);
            (sum_psnr, sum_tex, sum_seam, sum_ms, count) = (sum_psnr + s.psnr, sum_tex + s.texture, sum_seam + s.seam, sum_ms + ms, count + 1);
            if let Some(d) = &out_dir {
                // Original | hole | fill, cropped to the hole plus half its extent.
                let pad = ext / 2 + 8;
                let lb = b.0 - win.0;
                let tb = b.1 - win.1;
                let r = (lb.saturating_sub(pad), tb.saturating_sub(pad), (lb + (b.2 - b.0) + pad).min(ww), (tb + (b.3 - b.1) + pad).min(wh));
                let (cw, chh) = (r.2 - r.0, r.3 - r.1);
                let parts = [crop(&orig, ww, ch, r), crop(&holed, ww, ch, r), crop(&out, ww, ch, r)];
                let mut sheet = vec![0.0f32; cw * 3 * chh * ch];
                for (k, p) in parts.iter().enumerate() {
                    for y in 0..chh {
                        let dst = (y * cw * 3 + k * cw) * ch;
                        sheet[dst..dst + cw * ch].copy_from_slice(&p[y * cw * ch..(y + 1) * cw * ch]);
                    }
                }
                write_png(&d.join(format!("{name}-{case}.png")), &sheet, cw * 3, chh, ch);
            }
        }
    }
    if count > 0 {
        let n = f64::from(count);
        println!("{:<16} {:>7.2} {:>8.3} {:>6.3} {:>9.1}", "mean", sum_psnr / n, sum_tex / n, sum_seam / n, sum_ms / n);
    }
}
