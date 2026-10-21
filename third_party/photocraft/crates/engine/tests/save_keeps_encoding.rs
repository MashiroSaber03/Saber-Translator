//! #2399: a save writes a JPEG or WebP back the way it was opened. A JPEG keeps the quality
//! estimated from its quantisation tables, a lossy WebP stays lossy and a lossless one lossless,
//! unless the save asks for a quality; the remembered settings survive a .pcraft round trip.

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType, WebpEncoding, jpeg_quality, webp_encoding};
use photocraft_engine::Session;
use photocraft_engine::file_cmds::open_bytes_as;
use serde_json::json;

fn photo() -> Image {
    let (w, h) = (48u32, 32u32);
    let px = (0..w * h).flat_map(|i| {
        let (x, y) = (i % w, i / w);
        [(x * 5) as u8, (y * 7) as u8, ((x * y) % 251) as u8]
    });
    Image::from_raw(w, h, ChannelLayout::Rgb, SampleType::U8, px.collect()).unwrap()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("photocraft-2399-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Opens `bytes` as `name`, saves a copy to `out` with `params`, and returns the saved file.
fn open_and_save(name: &str, bytes: &[u8], out: &str, params: serde_json::Value) -> (Session, Vec<u8>) {
    let mut s = Session::new();
    open_bytes_as(&mut s, name, bytes, None, None).unwrap();
    let path = tmp(out);
    let mut p = params;
    p["path"] = json!(path.to_string_lossy());
    s.execute("file.saveACopy", p).unwrap();
    let saved = std::fs::read(&path).unwrap();
    (s, saved)
}

#[test]
fn a_jpeg_saves_at_its_own_quality() {
    for q in [35u8, 60, 77, 95] {
        let src = photocraft_codecs::encode(&photo(), Format::Jpeg, &EncodeOptions { jpeg_quality: q, ..Default::default() }).unwrap();
        let (s, saved) = open_and_save("in.jpg", &src, &format!("q{q}.jpg"), json!({}));
        let remembered = s.active().unwrap().doc.file_encoding.jpeg_quality.unwrap();
        assert!(remembered.abs_diff(q) <= 2, "opened at {q}, remembered {remembered}");
        let est = jpeg_quality(&saved).unwrap();
        assert!(est.abs_diff(q) <= 2, "opened at {q}, saved at {est}");
    }
    // A quality the save asks for still wins (Photoshop's 0–12 scale: 12 is the top).
    let src = photocraft_codecs::encode(&photo(), Format::Jpeg, &EncodeOptions { jpeg_quality: 40, ..Default::default() }).unwrap();
    let (_, saved) = open_and_save("in.jpg", &src, "explicit.jpg", json!({"quality": 12}));
    assert_eq!(jpeg_quality(&saved), Some(100));
}

#[test]
fn a_webp_stays_lossy_or_lossless() {
    let lossy = photocraft_codecs::encode(&photo(), Format::WebP, &EncodeOptions { webp_lossless: false, webp_quality: 60, ..Default::default() }).unwrap();
    let (s, saved) = open_and_save("in.webp", &lossy, "lossy.webp", json!({}));
    let e = s.active().unwrap().doc.file_encoding;
    assert_eq!(e.webp_lossless, Some(false));
    let Some(WebpEncoding::Lossy { quality: Some(q) }) = webp_encoding(&saved) else { panic!("saved {:?}", webp_encoding(&saved)) };
    assert!(q.abs_diff(60) <= 2, "lossy WebP at 60 saved at {q}");

    let lossless = photocraft_codecs::encode(&photo(), Format::WebP, &EncodeOptions::default()).unwrap();
    let (s, saved) = open_and_save("in.webp", &lossless, "lossless.webp", json!({}));
    assert_eq!(s.active().unwrap().doc.file_encoding.webp_lossless, Some(true));
    assert_eq!(webp_encoding(&saved), Some(WebpEncoding::Lossless));
    assert_eq!(photocraft_codecs::decode(&saved).unwrap().data(), photocraft_codecs::decode(&lossless).unwrap().data(), "lossless round trip");
}

#[test]
fn a_new_document_and_a_png_save_at_the_defaults() {
    let png = photocraft_codecs::encode(&photo(), Format::Png, &EncodeOptions::default()).unwrap();
    let (s, saved) = open_and_save("in.png", &png, "from-png.jpg", json!({}));
    assert!(s.active().unwrap().doc.file_encoding.is_empty());
    assert_eq!(jpeg_quality(&saved), Some(EncodeOptions::default().jpeg_quality));
    let (_, saved) = open_and_save("in.png", &png, "from-png.webp", json!({}));
    assert_eq!(webp_encoding(&saved), Some(WebpEncoding::Lossless));
}

#[test]
fn the_remembered_encoding_survives_pcraft() {
    let src = photocraft_codecs::encode(&photo(), Format::Jpeg, &EncodeOptions { jpeg_quality: 70, ..Default::default() }).unwrap();
    let mut s = Session::new();
    open_bytes_as(&mut s, "in.jpg", &src, None, None).unwrap();
    let doc = &s.active().unwrap().doc;
    let bytes = photocraft_format::save_to_bytes(doc, &photocraft_format::SaveOptions::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    assert_eq!(back.file_encoding, doc.file_encoding);
    assert!(back.file_encoding.jpeg_quality.is_some());
}
