//! Shared helpers for integration tests: deterministic PRNG, synthetic
//! image generators, comparison metrics.
#![allow(dead_code)]

pub mod tiffgen;

use photocraft_codecs::*;

/// SplitMix64 — tiny deterministic PRNG.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }
}

/// Gradient + noise image with values in [0, 1]. `noise` is the amplitude
/// of uniform noise added to the gradient. Alpha is a separate gradient
/// that includes fully transparent and fully opaque regions.
pub fn synth(w: u32, h: u32, layout: ChannelLayout, sample: SampleType, seed: u64, noise: f32) -> Image {
    let mut rng = Rng::new(seed);
    let nc = layout.channels();
    let mut v = Vec::with_capacity((w * h) as usize * nc);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w.max(2) - 1) as f32;
            let fy = y as f32 / (h.max(2) - 1) as f32;
            for c in 0..nc {
                let is_alpha = layout.has_alpha() && c == nc - 1;
                let base = if is_alpha {
                    ((fx + fy) * 0.75 - 0.25).clamp(0.0, 1.0)
                } else {
                    match c % 4 {
                        0 => fx,
                        1 => fy,
                        2 => 1.0 - fx * 0.5 - fy * 0.5,
                        _ => (fx * fy).sqrt(),
                    }
                };
                let n = (rng.f32() - 0.5) * noise;
                v.push((base + n).clamp(0.0, 1.0));
            }
        }
    }
    Image::from_normalized(w, h, layout, sample, &v).unwrap()
}

/// Default test image: 37x23 with moderate noise.
pub fn test_image(layout: ChannelLayout, sample: SampleType) -> Image {
    synth(37, 23, layout, sample, 0xC0DEC, 0.2)
}

/// PSNR in dB over normalized samples (clamped to [0,1]).
pub fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.dimensions(), b.dimensions());
    assert_eq!(a.layout(), b.layout());
    let (av, bv) = (a.to_normalized(), b.to_normalized());
    let mse: f64 = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| {
            let d = (x.clamp(0.0, 1.0) - y.clamp(0.0, 1.0)) as f64;
            d * d
        })
        .sum::<f64>()
        / av.len() as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (1.0 / mse).log10() }
}

pub fn max_abs_diff(a: &Image, b: &Image) -> f32 {
    a.to_normalized().iter().zip(b.to_normalized()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Formats that can be written in this build.
pub fn writable_formats() -> Vec<Format> {
    Format::ALL.into_iter().filter(|f| caps(*f).write).collect()
}

/// Formats that can be written *and* read in this build.
pub fn rw_formats() -> Vec<Format> {
    Format::ALL.into_iter().filter(|f| caps(*f).write && caps(*f).read).collect()
}

/// A minimal valid little-endian TIFF-structured EXIF blob (one IFD with
/// an Orientation entry).
pub fn sample_exif() -> Vec<u8> {
    let mut v = b"II*\0\x08\0\0\0".to_vec();
    v.extend_from_slice(&1u16.to_le_bytes()); // 1 entry
    v.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
    v.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // next IFD
    v
}

/// Fake ICC profile of `len` bytes (header-ish prefix + deterministic noise).
pub fn sample_icc(len: usize) -> Vec<u8> {
    let mut v = Rng::new(len as u64).bytes(len);
    v[..4].copy_from_slice(&(len as u32).to_be_bytes());
    if len >= 40 {
        v[36..40].copy_from_slice(b"acsp");
    }
    v
}

pub const SAMPLE_XMP: &str = "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"/></x:xmpmeta><?xpacket end=\"w\"?>";
