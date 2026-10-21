//! TIFF decode timings on a synthetic ~30 MP image (6336×4752). Ignored by default; run in
//! release:
//!
//! ```sh
//! cargo test --release -p photocraft-codecs --test tiff_bench -- --ignored --nocapture
//! ```

mod common;
use common::tiffgen::*;
use photocraft_codecs::*;
use std::time::Instant;

const W: u32 = 6336;
const H: u32 = 4752;

fn pixels(spp: usize) -> Vec<u8> {
    // A smooth gradient with a little texture: compresses like a photo, not like noise.
    let mut v = Vec::with_capacity(W as usize * H as usize * spp);
    let mut s = 0x1234_5678u32;
    for y in 0..H {
        for x in 0..W {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            for c in 0..spp {
                let g = (x * (c as u32 + 1) / 25 + y / 19) as u8;
                v.push(g.wrapping_add((s >> (8 * c)) as u8 & 3));
            }
        }
    }
    v
}

/// The decoder before FILE-215-8, for a same-run comparison: the `tiff` crate decodes the whole
/// image into one buffer (`read_image`, first plane only), which is then copied into an `Image`.
fn legacy(bytes: &[u8]) -> Result<Image, String> {
    use tiff::decoder::{Decoder, DecodingResult, Limits as TLimits};
    let mut l = TLimits::default();
    l.decoding_buffer_size = 8 << 30;
    l.intermediate_buffer_size = 1 << 30;
    let mut dec = Decoder::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?.with_limits(l);
    let (w, h) = dec.dimensions().map_err(|e| e.to_string())?;
    let layout = match dec.colortype().map_err(|e| e.to_string())? {
        tiff::ColorType::RGB(_) => ChannelLayout::Rgb,
        tiff::ColorType::Gray(_) => ChannelLayout::Gray,
        c => return Err(format!("{c:?}")),
    };
    match dec.read_image().map_err(|e| e.to_string())? {
        DecodingResult::U8(v) => Image::from_u8(w, h, layout, v),
        DecodingResult::U16(v) => Image::from_u16(w, h, layout, &v),
        _ => return Err("sample type".into()),
    }
    .map_err(|e| e.to_string())
}

fn best_of(f: impl Fn() -> Result<Image, String>) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        f()?;
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(best)
}

fn time(name: &str, bytes: &[u8]) {
    let new = best_of(|| decode(bytes).map_err(|e| e.to_string()));
    let old = best_of(|| legacy(bytes));
    let show = |r: &Result<f64, String>| match r {
        Ok(ms) => format!("{ms:>7.1} ms"),
        Err(e) => format!("error ({})", e.chars().take(48).collect::<String>()),
    };
    println!("{name:<34} new {}   before {}   ({:.1} MB file)", show(&new), show(&old), bytes.len() as f64 / 1e6);
}

#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn decode_30mp() {
    let rgb = pixels(3);
    let img = Image::from_u8(W, H, ChannelLayout::Rgb, rgb.clone()).unwrap();
    for (name, c) in
        [("strips, none", TiffCompression::None), ("strips, LZW + predictor", TiffCompression::Lzw), ("strips, Deflate + predictor", TiffCompression::Deflate)]
    {
        let bytes = encode(&img, Format::Tiff, &EncodeOptions { tiff_compression: c, ..Default::default() }).unwrap();
        time(&format!("RGB 8 {name}"), &bytes);
    }
    let img16 = img.convert(ChannelLayout::Rgb, SampleType::U16);
    let bytes = encode(&img16, Format::Tiff, &EncodeOptions { tiff_compression: TiffCompression::Deflate, ..Default::default() }).unwrap();
    time("RGB 16 strips, Deflate + predictor", &bytes);
    let bytes = encode(&img16, Format::Tiff, &EncodeOptions { tiff_compression: TiffCompression::None, ..Default::default() }).unwrap();
    time("RGB 16 strips, none", &bytes);

    // Hand-built layouts the encoder does not write: one big strip, tiles, planar.
    let le = cfg!(target_endian = "little");
    let one = build(le, false, &[strips(W, H, 8, 3, 2, H, &rgb)], &[0]);
    time("RGB 8 one strip, none", &one.bytes);
    let tiles = tile_split(&rgb, W as usize, H as usize, 3, 1, 256, 256, false);
    let mut d = Dir { entries: base(W, H, 8, 3, 2), chunks: tiles, tiled: true };
    d = d.tag(322, LONG, &[256]).tag(323, LONG, &[256]);
    time("RGB 8 tiles 256, none", &build(le, false, &[d], &[0]).bytes);
    let planes = planar_strips(&rgb, W as usize, H as usize, 3, 1, 64);
    let mut d = Dir { entries: base(W, H, 8, 3, 2), chunks: planes, tiled: false };
    d = d.tag(278, LONG, &[64]).tag(284, SHORT, &[2]);
    time("RGB 8 planar strips, none", &build(le, false, &[d], &[0]).bytes);
    let d = strips(W, H, 8, 3, 2, 64, &rgb);
    time("RGB 8 BigTIFF strips, none", &build(le, true, &[d], &[0]).bytes);
}
