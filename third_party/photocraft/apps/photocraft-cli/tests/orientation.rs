//! EXIF orientation end to end through the `photocraft-cli` binary (#285): a
//! phone-style JPEG (stored landscape, Orientation = 6) converts to an upright
//! portrait, an explicit rotation turns it back, and a `.pcraft` round trip
//! changes nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

use photocraft_codecs::{ChannelLayout, DecodeOptions, EncodeOptions, Format, Image, SampleType, decode, decode_with, exif_orientation};
use serde_json::Value;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_photocraft-cli"))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pc-cli-orient-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn ok(cmd: &mut Command) -> String {
    let o = cmd.output().unwrap();
    let (out, err) = (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned());
    assert!(o.status.success(), "exit {:?}\nstdout: {out}\nstderr: {err}", o.status);
    out
}

/// EXIF block: Orientation = `o` (SHORT) and an inline Software tag, little-endian.
fn exif(o: u16) -> Vec<u8> {
    let mut v = b"II*\0\x08\0\0\0\x02\0".to_vec();
    v.extend_from_slice(&[0x12, 0x01, 3, 0, 1, 0, 0, 0]);
    v.extend_from_slice(&o.to_le_bytes());
    v.extend_from_slice(&[0, 0]);
    v.extend_from_slice(&[0x31, 0x01, 2, 0, 4, 0, 0, 0]);
    v.extend_from_slice(b"PC1\0");
    v.extend_from_slice(&[0, 0, 0, 0]);
    v
}

/// Upright 16×32 portrait: red top-left quarter, green top-right, blue bottom half.
fn upright_color(x: usize, y: usize) -> [u8; 3] {
    match (x < 8, y < 16) {
        (true, true) => [230, 20, 20],
        (false, true) => [20, 230, 20],
        _ => [20, 20, 230],
    }
}

/// The portrait as a camera stores it for Orientation = 6: 32×16, with EXIF.
fn phone_jpeg() -> Vec<u8> {
    let (w, h) = (16, 32);
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // Upright (x, y) sits at stored (y, w - 1 - x) in a 32-wide image.
            let (sx, sy) = (y, w - 1 - x);
            px[(sy * h + sx) * 3..][..3].copy_from_slice(&upright_color(x, y));
        }
    }
    let stored = Image::from_u8(h as u32, w as u32, ChannelLayout::Rgb, px).unwrap();
    let jpeg =
        photocraft_codecs::encode(&stored, Format::Jpeg, &EncodeOptions { jpeg_quality: 100, jpeg_chroma_subsampling: false, ..Default::default() }).unwrap();
    let mut seg = b"Exif\0\0".to_vec();
    seg.extend_from_slice(&exif(6));
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&seg);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// The file as stored (no orientation applied), so its own dimensions and tag can be checked.
fn stored(path: &Path) -> Image {
    decode_with(&std::fs::read(path).unwrap(), &DecodeOptions { keep_orientation: true, ..Default::default() }).unwrap()
}

fn assert_portrait_pixels(img: &Image) {
    let img = img.convert(ChannelLayout::Rgb, SampleType::U8);
    for (x, y) in [(3, 4), (12, 4), (3, 28), (12, 28)] {
        let i = (y * 16 + x) * 3;
        let got = &img.data()[i..i + 3];
        let want = upright_color(x, y);
        assert!(got.iter().zip(want).all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() <= 12), "({x},{y}): {got:?} vs {want:?}");
    }
}

#[test]
fn orientation_6_jpeg_converts_upright_and_rotates_back() {
    let d = tmp("e2e");
    let jpg = d.join("IMG_0006.JPG");
    std::fs::write(&jpg, phone_jpeg()).unwrap();

    // convert → PNG: the stored pixels are already upright, and any EXIF says 1.
    let png = d.join("upright.png");
    ok(bin().arg("convert").arg(&jpg).arg(&png));
    let raw = stored(&png);
    assert_eq!(raw.dimensions(), (16, 32), "stored upright, not landscape");
    assert_portrait_pixels(&raw);
    if let Some(e) = raw.meta.exif.as_deref() {
        assert_eq!(exif_orientation(e), 1, "written EXIF orientation");
    }
    assert_eq!(exif_orientation(&std::fs::read(&png).unwrap()), 1);
    assert_eq!(decode(&std::fs::read(&png).unwrap()).unwrap().dimensions(), (16, 32), "no second turn on reopen");

    // An explicit Image › Image Rotation › 90° Clockwise makes it landscape again.
    let turned = d.join("turned.png");
    ok(bin().arg("run").arg(&jpg).args(["--cmd", "image.imageRotation.90cw", "--out"]).arg(&turned));
    let t = stored(&turned);
    assert_eq!(t.dimensions(), (32, 16));
    if let Some(e) = t.meta.exif.as_deref() {
        assert_eq!(exif_orientation(e), 1);
    }
    // The portrait's top-left (red) is now top-right; its bottom (blue) is on the left.
    let t = t.convert(ChannelLayout::Rgb, SampleType::U8);
    let at = |x: usize, y: usize| <[u8; 3]>::try_from(&t.data()[(y * 32 + x) * 3..][..3]).unwrap();
    let near = |a: [u8; 3], b: [u8; 3]| a.iter().zip(b).all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() <= 12);
    assert!(near(at(28, 3), upright_color(3, 4)), "{:?}", at(28, 3));
    assert!(near(at(28, 12), upright_color(12, 4)), "{:?}", at(28, 12));
    assert!(near(at(4, 8), upright_color(8, 28)), "{:?}", at(4, 8));

    // .pcraft save, reopen and save again: identical document and pixels.
    let (a, b) = (d.join("a.pcraft"), d.join("b.pcraft"));
    ok(bin().arg("convert").arg(&jpg).arg(&a));
    ok(bin().arg("convert").arg(&a).arg(&b));
    // Everything but the file's own path and saved document name must match.
    let info = |p: &Path| {
        let mut v = serde_json::from_str::<Value>(&ok(bin().arg("info").arg(p))).unwrap();
        v.as_object_mut().unwrap().remove("file");
        v.as_object_mut().unwrap().remove("path");
        v
    };
    let (mut ia, mut ib) = (info(&a), info(&b));
    assert_eq!((ia["width"].as_u64(), ia["height"].as_u64()), (Some(16), Some(32)));
    assert_eq!(ia["name"], "a.pcraft");
    assert_eq!(ib["name"], "b.pcraft");
    ia.as_object_mut().unwrap().remove("name");
    ib.as_object_mut().unwrap().remove("name");
    assert_eq!(ia, ib);
    let (pa, pb) = (d.join("a.png"), d.join("b.png"));
    ok(bin().arg("convert").arg(&a).arg(&pa));
    ok(bin().arg("convert").arg(&b).arg(&pb));
    let (xa, xb) = (stored(&pa), stored(&pb));
    assert_eq!(xa.dimensions(), (16, 32));
    assert_eq!(xa.to_rgba8(), xb.to_rgba8());
    assert_eq!(xa.meta.exif, xb.meta.exif);
    assert_eq!(xa.to_rgba8(), stored(&png).to_rgba8(), "the .pcraft round trip matches the direct conversion");
    std::fs::remove_dir_all(d).unwrap();
}
