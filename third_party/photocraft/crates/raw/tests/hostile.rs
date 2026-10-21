//! Truncated and corrupted raw files must fail cleanly, never panic, and
//! never allocate beyond the limits.

use photocraft_raw::testgen::{Cr2Spec, DngSpec, DngStorage, RafPacking, RafSpec, XTRANS, mosaic, mosaic_cfa, orf, rw2, scene, sony_craw, tiff_ep};
use photocraft_raw::*;

/// Losslessly encodes 16-bit RGB as a JPEG XL codestream.
fn jxl_rgb16(rgb: &[u16], w: usize, h: usize) -> Vec<u8> {
    use zune_core::{bit_depth::BitDepth, colorspace::ColorSpace, options::EncoderOptions};
    let bytes: Vec<u8> = rgb.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut out = Vec::new();
    zune_jpegxl::JxlSimpleEncoder::new(&bytes, EncoderOptions::new(w, h, ColorSpace::RGB, BitDepth::Sixteen)).encode(&mut out).unwrap();
    out
}

fn samples() -> Vec<Vec<u8>> {
    let (w, h) = (24, 12);
    let cfa = mosaic(&scene(w, h), w, [0, 1, 1, 2], 64, 4000);
    let mut out = Vec::new();
    let mut d = DngSpec::cfa(w, h, cfa.clone());
    d.bits = 12;
    d.white = 4000;
    d.black = vec![64];
    d.as_shot_neutral = Some([0.5, 1.0, 0.7]);
    d.color_matrix1 = Some((21, [3.2, -1.5, -0.5, -1.0, 1.9, 0.04, 0.06, -0.2, 1.06]));
    out.push(d.build());
    d.storage = DngStorage::Lj92Tiles { width: 16, height: 8 };
    d.active_area = Some([1, 2, 11, 22]);
    d.default_crop = Some(([1, 1], [16, 8]));
    out.push(d.build());
    d.storage = DngStorage::Strips { rows: 5 };
    d.big_endian = true;
    out.push(d.build());
    // LinearRaw in one JPEG XL strip (DNG 1.7, as Samsung Expert RAW writes it).
    let rgb: Vec<u16> = scene(w, h).iter().flat_map(|p| p.map(|v| (v * 30000.0) as u16)).collect();
    let mut j = DngSpec::cfa(w, h, rgb.clone());
    j.samples = 3;
    j.storage = DngStorage::JxlStrips { rows: h };
    j.jxl_segments = vec![jxl_rgb16(&rgb, w, h)];
    j.profile_gain_table_map = Some(vec![0; 32]);
    out.push(j.build());
    // Monochrome LinearRaw (one sample, no CFA, no colour matrix), as film scanners write it (#3497).
    let grey: Vec<u16> = scene(w, h).iter().map(|p| 64 + (p[1] * 3900.0) as u16).collect();
    let mut m = DngSpec::monochrome(w, h, grey);
    m.bits = 12;
    m.white = 4000;
    m.black = vec![64];
    m.storage = DngStorage::Lj92Tiles { width: 16, height: 8 };
    out.push(m.build());
    m.storage = DngStorage::DeflateStrips { rows: 5, predictor: 2 };
    out.push(m.build());
    let mut data = vec![600u16; 40 * 10];
    for (i, v) in data.iter_mut().enumerate() {
        *v = 600 + (i as u16 * 7) % 3000;
    }
    out.push(
        Cr2Spec {
            width: 40,
            height: 10,
            data,
            precision: 14,
            components: 2,
            slices: vec![16, 16, 8],
            borders: Some([8, 2, 39, 9]),
            wb_rggb: Some([2000, 1024, 1024, 1500]),
            orientation: 8,
            model_id: None,
        }
        .build(),
    );
    out.push(tiff_ep("NIKON", w, h, &cfa, [0, 1, 1, 2], 12, vec![]));
    let codes = mosaic(&scene(64, 6), 64, [0, 1, 1, 2], 256, 1900);
    out.push(sony_craw(64, 6, &codes, [8000, 10400, 12900, 14100]));
    out.push(rw2(30, 8, &mosaic(&scene(30, 8), 30, [0, 1, 1, 2], 128, 4095), 12));
    out.push(orf(w, h, &cfa));
    let xt = mosaic_cfa(&scene(w, h), w, &XTRANS, 6, 256, 4000);
    for packing in [RafPacking::U16 { le: true }, RafPacking::Lsb12] {
        let raf =
            RafSpec { bits: 12, packing, black: vec![256; 36], crop: Some([1, 2, 10, 20]), orientation: 8, ..RafSpec::new(w, h, xt.clone(), XTRANS.to_vec()) };
        out.push(raf.build());
    }
    let bayer = RafSpec { packing: RafPacking::Words14, exposure_bias: Some((-100, 100)), ..RafSpec::new(w, h, cfa.clone(), vec![0, 1, 1, 2]) };
    out.push(bayer.build());
    out
}

fn exercise(b: &[u8]) {
    let _ = identify(b);
    let _ = embedded_preview(b);
    let _ = embedded_previews(b);
    let _ = embedded_preview_color_space(b);
    let opts = DevelopOptions { limits: Limits { max_alloc: 64 << 20, ..Limits::default() }, ..Default::default() };
    if let Ok(s) = decode(b, &opts.limits) {
        for m in [Demosaic::Bilinear, Demosaic::Ahd] {
            let _ = develop_sensor(&s, &DevelopOptions { demosaic: m, ..opts.clone() });
        }
        // A colour fit against a gradient of the crop's (capped) shape, then a develop with it.
        let (w, h) = (s.crop.width.min(256), s.crop.height.min(256));
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        if let Ok(look) = fit_look(&s, &opts, &Reference { width: w as u32, height: h as u32, rgb: &rgb }) {
            let _ = develop_sensor(&s, &DevelopOptions { camera_look: Some(look), ..opts.clone() });
        }
    }
}

#[test]
fn originals_develop() {
    for (i, b) in samples().iter().enumerate() {
        let d = develop(b, &DevelopOptions::default());
        assert!(d.is_ok(), "sample {i}: {:?}", d.err());
    }
}

#[test]
fn every_truncation_fails_cleanly() {
    for b in samples() {
        for n in (0..b.len()).step_by(3) {
            exercise(&b[..n]);
        }
    }
}

#[test]
fn random_corruption_never_panics() {
    let mut s = 0xDEAD_BEEF_u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s >> 33
    };
    for b in samples() {
        for _ in 0..400 {
            let mut c = b.clone();
            for _ in 0..1 + next() % 6 {
                let at = next() as usize % c.len();
                c[at] = match next() % 4 {
                    0 => 0xFF,
                    1 => 0,
                    _ => next() as u8,
                };
            }
            exercise(&c);
        }
    }
}

#[test]
fn absurd_dimensions_are_rejected_without_allocating() {
    let mut d = DngSpec::cfa(4, 4, vec![0; 16]);
    d.storage = DngStorage::Strips { rows: 4 };
    let mut b = d.build();
    // Patch the raw IFD's ImageWidth / ImageLength (LONG, inline) to 200000 × 200000.
    let t = 256u16.to_le_bytes();
    let mut patched = 0;
    for i in 0..b.len().saturating_sub(12) {
        if b[i..i + 2] == t && b[i + 2..i + 4] == 4u16.to_le_bytes() && b[i + 8..i + 12] == 4u32.to_le_bytes() {
            b[i + 8..i + 12].copy_from_slice(&200_000u32.to_le_bytes());
            b[i + 20..i + 24].copy_from_slice(&200_000u32.to_le_bytes());
            patched += 1;
        }
    }
    assert!(patched > 0);
    assert!(matches!(decode(&b, &Limits::default()), Err(RawError::LimitExceeded(_))));
}

/// An 8×8 CFA image stored as one lossless-JPEG tile that the file declares 512×512: the
/// reader takes the segment's geometry from the tile tags, not from the image (#2925).
fn oversized_lj92_tile() -> Vec<u8> {
    let cfa = mosaic(&scene(8, 8), 8, [0, 1, 1, 2], 64, 4000);
    let mut d = DngSpec::cfa(8, 8, cfa);
    d.bits = 12;
    d.white = 4000;
    d.black = vec![64];
    d.storage = DngStorage::Lj92Tiles { width: 512, height: 512 };
    d.build()
}

/// Rewrites one inline LONG IFD value; `false` when the tag is not found exactly once.
fn patch_long(b: &mut [u8], tag: u16, to: u32) -> bool {
    let mut patched = 0;
    for i in 0..b.len().saturating_sub(12) {
        if b[i..i + 2] == tag.to_le_bytes() && b[i + 2..i + 4] == 4u16.to_le_bytes() && b[i + 4..i + 8] == 1u32.to_le_bytes() {
            b[i + 8..i + 12].copy_from_slice(&to.to_le_bytes());
            patched += 1;
        }
    }
    patched == 1
}

/// The lossless-JPEG intermediate is sized from the segment, so the plane check alone
/// does not bound it: it has to be charged against `max_alloc` too (#2925).
#[test]
fn lossless_jpeg_intermediate_is_charged_against_max_alloc() {
    let b = oversized_lj92_tile();
    assert!(decode(&b, &Limits::default()).is_ok(), "the file itself is valid");
    // The image is 8 * 8 * 2 = 128 bytes, so the plane check passes, while the declared
    // tile's intermediate is 512 * 512 * 2 bytes — far past a 64 KiB budget.
    let budget = Limits { max_alloc: 64 << 10, ..Limits::default() };
    assert!(matches!(decode(&b, &budget), Err(RawError::LimitExceeded(_))));
    // A budget that does hold the intermediate still decodes it unchanged.
    let roomy = Limits { max_alloc: 4 << 20, ..Limits::default() };
    assert!(decode(&b, &roomy).is_ok(), "a budget that holds the intermediate still decodes it");
    // Ordinary geometry stays inside the same budget: the segment check is not stricter
    // than the image it belongs to.
    let mut normal = DngSpec::cfa(8, 8, mosaic(&scene(8, 8), 8, [0, 1, 1, 2], 64, 4000));
    normal.bits = 12;
    normal.white = 4000;
    normal.black = vec![64];
    normal.storage = DngStorage::Lj92Tiles { width: 4, height: 4 };
    let tight = Limits { max_alloc: 4096, ..Limits::default() };
    assert!(decode(&normal.build(), &tight).is_ok(), "ordinary geometry stays inside the budget");
}

/// A frame may declare up to 4× the segment's samples (the headroom the reader passes);
/// when that does not fit the budget either, the decode is refused rather than allocated.
#[test]
fn lossless_jpeg_frame_larger_than_the_budget_is_refused() {
    let mut b = oversized_lj92_tile();
    // Shrink the declared tile to 256×256: its geometry then allows exactly 4× its own
    // samples — the 262144-sample frame this file actually holds (322 = TileWidth,
    // 323 = TileLength).
    assert!(patch_long(&mut b, 322, 256), "TileWidth patched once");
    assert!(patch_long(&mut b, 323, 256), "TileLength patched once");
    // Holds the image (128 B) and the tile (128 KiB), but not the frame's 512 KiB.
    let budget = Limits { max_alloc: 256 << 10, ..Limits::default() };
    assert!(matches!(decode(&b, &budget), Err(RawError::LimitExceeded(_))));
}
