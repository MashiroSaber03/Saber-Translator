//! #1691: a camera JPEG with its resolution only in EXIF opens at that resolution, and a JPEG
//! saved after Image Size (resolution only) carries the new resolution in JFIF and EXIF alike.

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, Metadata, SampleType, exif_resolution};
use photocraft_engine::Session;
use photocraft_engine::file_cmds::open_bytes_as;
use serde_json::json;

/// Little-endian EXIF: Make "NIKON", XResolution 300/1, YResolution 300/1, ResolutionUnit 2.
fn camera_exif() -> Vec<u8> {
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(&8u32.to_le_bytes());
    v.extend_from_slice(&4u16.to_le_bytes());
    let data = 8 + 2 + 4 * 12 + 4;
    let entry = |v: &mut Vec<u8>, tag: u16, ty: u16, count: u32, value: u32| {
        v.extend_from_slice(&tag.to_le_bytes());
        v.extend_from_slice(&ty.to_le_bytes());
        v.extend_from_slice(&count.to_le_bytes());
        v.extend_from_slice(&value.to_le_bytes());
    };
    entry(&mut v, 0x010F, 2, 6, data as u32);
    entry(&mut v, 282, 5, 1, data as u32 + 6);
    entry(&mut v, 283, 5, 1, data as u32 + 14);
    entry(&mut v, 296, 3, 1, 2);
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(b"NIKON\0");
    for _ in 0..2 {
        v.extend_from_slice(&300u32.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
    }
    v
}

/// A camera-style JPEG: no JFIF header, the resolution only in EXIF.
fn camera_jpeg() -> Vec<u8> {
    let meta = Metadata { exif: Some(camera_exif()), ..Default::default() };
    let img = Image::from_raw(16, 8, ChannelLayout::Rgb, SampleType::U8, vec![120; 16 * 8 * 3]).unwrap().with_meta(meta);
    let b = photocraft_codecs::encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
    assert_eq!(&b[2..4], &[0xFF, 0xE0], "the encoder starts with JFIF");
    let len = u16::from_be_bytes([b[4], b[5]]) as usize;
    [&b[..2], &b[4 + len..]].concat()
}

/// (marker, payload) of each APPn segment.
fn app_segments(b: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= b.len() && b[i] == 0xFF && b[i + 1] != 0xDA {
        let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        if (0xE0..=0xEF).contains(&b[i + 1]) {
            out.push((b[i + 1], b[i + 4..i + 2 + len].to_vec()));
        }
        i += 2 + len;
    }
    out
}

#[test]
fn exif_resolution_opens_and_saves_consistently() {
    let src = camera_jpeg();
    assert!(app_segments(&src).iter().all(|(m, _)| *m != 0xE0));
    let mut s = Session::new();
    open_bytes_as(&mut s, "DSC_0001.JPG", &src, None, None).unwrap();
    assert_eq!(s.active().unwrap().doc.resolution_dpi, 300.0);

    s.execute("image.imageSize", json!({"resolution": 240, "resample": "none"})).unwrap();
    let d = &s.active().unwrap().doc;
    assert_eq!((d.resolution_dpi, d.size.width, d.size.height), (240.0, 16, 8));

    let dir = std::env::temp_dir().join(format!("photocraft-1691-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("saved.jpg");
    s.execute("file.saveACopy", json!({"path": out.to_string_lossy()})).unwrap();
    let saved = std::fs::read(&out).unwrap();
    println!("saved {}", out.display());

    let segs = app_segments(&saved);
    let (_, jfif) = segs.iter().find(|(m, p)| *m == 0xE0 && p.starts_with(b"JFIF\0")).unwrap();
    assert_eq!((jfif[7], u16::from_be_bytes([jfif[8], jfif[9]]), u16::from_be_bytes([jfif[10], jfif[11]])), (1, 240, 240));
    let (_, exif) = segs.iter().find(|(m, p)| *m == 0xE1 && p.starts_with(b"Exif\0\0")).unwrap();
    assert_eq!(exif_resolution(exif), Some((240.0, 240.0)));
    assert!(exif.windows(6).any(|w| w == b"NIKON\0"), "the rest of the EXIF is kept");
    assert_eq!(photocraft_io::import("saved.jpg", &saved).unwrap().document.resolution_dpi, 240.0);
}
