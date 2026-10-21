//! Writes multi-gigabyte PSB files for large-file tests (#375, #216), straight from the Adobe
//! PSD spec and independent of this crate's writer, so the fixtures don't inherit its limits.
//!
//! `cargo run --release -p photocraft-psd --example big_psb -- <out dir> [case...]`
//!
//! Every sample follows [`sample`], so a reader can check any pixel without the file. RLE cases
//! keep each row constant (tiny files, full decoded size); raw cases vary along x too.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// One generated file.
struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    depth: u16,
    /// Raw (0) or RLE (1).
    rle: bool,
    /// One full-canvas RGBA layer, or a flattened file (composite only).
    layered: bool,
}

const CASES: &[Case] = &[
    // Composite of 2.52 GB in one decode.
    Case { name: "flat8-29k-rle", width: 29_000, height: 29_000, depth: 8, rle: true, layered: false },
    // A 4.8 GB file: offsets past 4 GiB, composite of 4.8 GB.
    Case { name: "flat8-40k-raw", width: 40_000, height: 40_000, depth: 8, rle: false, layered: false },
    // 4.38 GB file: layer channels of 625 MB, composite of 1.9 GB, data past 4 GiB.
    Case { name: "layer8-25k-raw", width: 25_000, height: 25_000, depth: 8, rle: false, layered: true },
    // Layer channels of 900 MB, composite of 2.7 GB.
    Case { name: "layer8-30k-rle", width: 30_000, height: 30_000, depth: 8, rle: true, layered: true },
    // Layer channels of 2.18 GB each.
    Case { name: "layer16-33k-rle", width: 33_000, height: 33_000, depth: 16, rle: true, layered: true },
];

/// The 8-bit sample of channel `c` (0..3 colour, 3 = alpha) at (`x`, `y`). RLE files ignore `x`.
/// 16-bit files store `v * 257`.
pub fn sample(x: u32, y: u32, c: u32, rle: bool) -> u8 {
    if c == 3 {
        return 255;
    }
    let x = if rle { 0 } else { x };
    (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)).wrapping_add(c.wrapping_mul(101)) & 0xff) as u8
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().map_or_else(|| std::env::temp_dir().join("photocraft-big-psb"), std::path::PathBuf::from);
    std::fs::create_dir_all(&out)?;
    let wanted = args.get(1..).unwrap_or(&[]);
    for case in CASES.iter().filter(|c| wanted.is_empty() || wanted.iter().any(|w| w == c.name)) {
        let path = out.join(format!("{}.psb", case.name));
        let t = std::time::Instant::now();
        write_case(case, &path)?;
        let len = std::fs::metadata(&path)?.len();
        println!("{}: {:.2} GB in {:.1} s", path.display(), len as f64 / 1e9, t.elapsed().as_secs_f64());
    }
    Ok(())
}

fn row_bytes(c: &Case) -> u64 {
    u64::from(c.width) * u64::from(c.depth / 8)
}

/// PackBits for a row of `n` equal bytes: runs of 128, then the remainder.
fn rle_row(n: u64, b: u8) -> Vec<u8> {
    let mut v = Vec::new();
    let mut left = n;
    while left > 0 {
        let run = left.min(128);
        if run == 1 {
            v.extend_from_slice(&[0, b]);
        } else {
            v.extend_from_slice(&[(257 - run) as u8, b]);
        }
        left -= run;
    }
    v
}

/// The bytes of one row of channel `ch`.
fn row(c: &Case, y: u32, ch: u32, buf: &mut Vec<u8>) {
    buf.clear();
    if c.rle {
        buf.extend(rle_row(row_bytes(c), sample(0, y, ch, true)));
        return;
    }
    for x in 0..c.width {
        let s = sample(x, y, ch, false);
        buf.push(s);
        if c.depth == 16 {
            buf.push(s);
        }
    }
}

/// Size of a channel block (compression word, RLE counts, rows) of `planes` planes.
fn channel_len(c: &Case, planes: u64) -> u64 {
    let rows = planes * u64::from(c.height);
    if c.rle { 2 + rows * 4 + rows * rle_row(row_bytes(c), 0).len() as u64 } else { 2 + rows * row_bytes(c) }
}

/// Writes the compression word, the counts and the rows of `chans`.
fn write_planes(w: &mut impl Write, c: &Case, chans: &[u32]) -> io::Result<()> {
    w.write_all(&u16::from(c.rle).to_be_bytes())?;
    if c.rle {
        let n = rle_row(row_bytes(c), 0).len() as u32;
        for _ in 0..chans.len() as u64 * u64::from(c.height) {
            w.write_all(&n.to_be_bytes())?;
        }
    }
    let mut buf = Vec::new();
    for &ch in chans {
        for y in 0..c.height {
            row(c, y, ch, &mut buf);
            w.write_all(&buf)?;
        }
    }
    Ok(())
}

fn write_case(c: &Case, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 << 20, File::create(path)?);
    // Header: PSB (version 2), 3 channels, RGB.
    w.write_all(b"8BPS")?;
    w.write_all(&2u16.to_be_bytes())?;
    w.write_all(&[0; 6])?;
    w.write_all(&3u16.to_be_bytes())?;
    w.write_all(&c.height.to_be_bytes())?;
    w.write_all(&c.width.to_be_bytes())?;
    w.write_all(&c.depth.to_be_bytes())?;
    w.write_all(&3u16.to_be_bytes())?;
    // Colour mode data and image resources: empty.
    w.write_all(&0u32.to_be_bytes())?;
    w.write_all(&0u32.to_be_bytes())?;

    if c.layered {
        // Transparency (-1), then R, G, B: one block per channel.
        let ids: [(i16, u32); 4] = [(-1, 3), (0, 0), (1, 1), (2, 2)];
        let ch_len = channel_len(c, 1);
        let name = b"Layer 1";
        let mut rec = Vec::new();
        for v in [0i32, 0, c.height as i32, c.width as i32] {
            rec.extend_from_slice(&v.to_be_bytes());
        }
        rec.extend_from_slice(&4u16.to_be_bytes());
        for (id, _) in ids {
            rec.extend_from_slice(&id.to_be_bytes());
            rec.extend_from_slice(&ch_len.to_be_bytes());
        }
        rec.extend_from_slice(b"8BIMnorm");
        rec.extend_from_slice(&[255, 0, 0, 0]);
        // Extra data: no mask, no blending ranges, the Pascal name padded to 4.
        let mut extra = Vec::new();
        extra.extend_from_slice(&0u32.to_be_bytes());
        extra.extend_from_slice(&0u32.to_be_bytes());
        extra.push(name.len() as u8);
        extra.extend_from_slice(name);
        while extra.len() % 4 != 0 {
            extra.push(0);
        }
        rec.extend_from_slice(&(extra.len() as u32).to_be_bytes());
        rec.extend_from_slice(&extra);

        let mut layer_info = 2 + rec.len() as u64 + 4 * ch_len;
        let pad = layer_info % 2;
        layer_info += pad;
        // Layer and mask information: layer info (8-byte length) + empty global mask (4 bytes).
        w.write_all(&(8 + layer_info + 4).to_be_bytes())?;
        w.write_all(&layer_info.to_be_bytes())?;
        w.write_all(&1i16.to_be_bytes())?;
        w.write_all(&rec)?;
        for (_, ch) in ids {
            write_planes(&mut w, c, &[ch])?;
        }
        w.write_all(&vec![0; pad as usize])?;
        w.write_all(&0u32.to_be_bytes())?;
    } else {
        w.write_all(&0u64.to_be_bytes())?;
    }
    // The merged image.
    write_planes(&mut w, c, &[0, 1, 2])?;
    w.flush()
}
