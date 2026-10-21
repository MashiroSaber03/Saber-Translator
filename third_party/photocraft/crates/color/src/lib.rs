//! Pixel formats, colour models and blend-mode math.
//!
//! Bit depth and colour model are *runtime data* (see `plan/architecture.md` §1.1):
//! nothing in the engine assumes 8-bit or RGB.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod blend;
pub mod convert;

pub use blend::BlendMode;

/// Per-pixel gradient dither noise in `[0, 1)` from a position hash (decorrelated, no visible
/// pattern). Shared by the Gradient tool and Gradient Fill layers (and mirrored in the GPU
/// compositor), so a live gradient dithers exactly like the painted one.
pub fn dither_noise(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    f32::from((h & 0xFFFF) as u16) / 65535.0
}

use serde::{Deserialize, Serialize};

/// Storage type of one channel sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleType {
    U8,
    U16,
    F32,
}

impl SampleType {
    pub const ALL: [SampleType; 3] = [SampleType::U8, SampleType::U16, SampleType::F32];

    pub const fn bytes(self) -> usize {
        match self {
            SampleType::U8 => 1,
            SampleType::U16 => 2,
            SampleType::F32 => 4,
        }
    }
    pub const fn bits(self) -> u32 {
        self.bytes() as u32 * 8
    }
}

/// Document colour model (Photoshop "Image → Mode").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorMode {
    Bitmap,
    Grayscale,
    Indexed,
    Rgb,
    Cmyk,
    Lab,
    Multichannel,
    Duotone,
}

impl ColorMode {
    /// Number of colour (non-alpha) channels used for pixel storage.
    /// Indexed and Bitmap documents are stored expanded (RGB / Gray) for editing.
    pub const fn color_channels(self) -> usize {
        match self {
            ColorMode::Bitmap | ColorMode::Grayscale | ColorMode::Duotone => 1,
            ColorMode::Indexed | ColorMode::Rgb | ColorMode::Lab => 3,
            ColorMode::Cmyk => 4,
            // Multichannel documents carry their channel count separately; default to 3.
            ColorMode::Multichannel => 3,
        }
    }
}

/// Layout of a pixel: colour model, sample type, and whether an alpha channel follows the colour channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelFormat {
    pub mode: ColorMode,
    pub sample: SampleType,
    pub alpha: bool,
}

impl PixelFormat {
    pub const RGBA8: PixelFormat = PixelFormat { mode: ColorMode::Rgb, sample: SampleType::U8, alpha: true };
    pub const RGBA16: PixelFormat = PixelFormat { mode: ColorMode::Rgb, sample: SampleType::U16, alpha: true };
    pub const RGBA32F: PixelFormat = PixelFormat { mode: ColorMode::Rgb, sample: SampleType::F32, alpha: true };
    pub const GRAY8: PixelFormat = PixelFormat { mode: ColorMode::Grayscale, sample: SampleType::U8, alpha: false };
    pub const GRAYA8: PixelFormat = PixelFormat { mode: ColorMode::Grayscale, sample: SampleType::U8, alpha: true };
    pub const CMYKA8: PixelFormat = PixelFormat { mode: ColorMode::Cmyk, sample: SampleType::U8, alpha: true };

    pub const fn new(mode: ColorMode, sample: SampleType, alpha: bool) -> Self {
        Self { mode, sample, alpha }
    }
    pub const fn channels(&self) -> usize {
        self.mode.color_channels() + self.alpha as usize
    }
    pub const fn bytes_per_pixel(&self) -> usize {
        self.channels() * self.sample.bytes()
    }
    pub const fn with_sample(self, sample: SampleType) -> Self {
        Self { sample, ..self }
    }
}

/// Read one sample at `index` (sample index, not byte index) as normalised f32.
/// U8/U16 map to [0, 1]; F32 is returned as stored.
#[inline]
pub fn read_sample(bytes: &[u8], sample: SampleType, index: usize) -> f32 {
    match sample {
        SampleType::U8 => bytes[index] as f32 / 255.0,
        SampleType::U16 => {
            let o = index * 2;
            u16::from_ne_bytes([bytes[o], bytes[o + 1]]) as f32 / 65535.0
        }
        SampleType::F32 => {
            let o = index * 4;
            f32::from_ne_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        }
    }
}

/// Write a normalised f32 sample; integer types are clamped and rounded.
#[inline]
pub fn write_sample(bytes: &mut [u8], sample: SampleType, index: usize, v: f32) {
    match sample {
        SampleType::U8 => bytes[index] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        SampleType::U16 => {
            let o = index * 2;
            let q = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            bytes[o..o + 2].copy_from_slice(&q.to_ne_bytes());
        }
        SampleType::F32 => {
            let o = index * 4;
            bytes[o..o + 4].copy_from_slice(&v.to_ne_bytes());
        }
    }
}

/// A colour value in the document's model with alpha, as normalised floats.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub mode: ColorMode,
    /// Up to 4 colour components (unused ones are 0) followed by nothing; alpha kept separately.
    pub c: [f32; 4],
    pub alpha: f32,
}

impl Color {
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { mode: ColorMode::Rgb, c: [r, g, b, 0.0], alpha: 1.0 }
    }
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { mode: ColorMode::Rgb, c: [r, g, b, 0.0], alpha: a }
    }
    pub const fn gray(v: f32) -> Self {
        Self { mode: ColorMode::Grayscale, c: [v, 0.0, 0.0, 0.0], alpha: 1.0 }
    }
    pub const BLACK: Color = Color::rgb(0.0, 0.0, 0.0);
    pub const WHITE: Color = Color::rgb(1.0, 1.0, 1.0);
    pub const TRANSPARENT: Color = Color::rgba(0.0, 0.0, 0.0, 0.0);

    /// Convert to sRGB (CMYK through the built-in CMYK profile, Lab via D50 formulas).
    pub fn to_rgb(&self) -> [f32; 3] {
        match self.mode {
            ColorMode::Rgb | ColorMode::Indexed | ColorMode::Multichannel => [self.c[0], self.c[1], self.c[2]],
            ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => [self.c[0]; 3],
            ColorMode::Cmyk => convert::cmyk_to_rgb([self.c[0], self.c[1], self.c[2], self.c[3]]),
            ColorMode::Lab => convert::lab_to_srgb([self.c[0] * 100.0, self.c[1] * 255.0 - 128.0, self.c[2] * 255.0 - 128.0]),
        }
    }

    /// This colour in `mode`'s model, converted as pixels written to a surface of that mode
    /// are (through sRGB); unchanged when it is already in `mode`. Indexed and Multichannel
    /// documents keep RGB colours, as [`Color::to_rgb`] reads them.
    pub fn in_mode(self, mode: ColorMode) -> Color {
        let rgb_like = |m: ColorMode| matches!(m, ColorMode::Rgb | ColorMode::Indexed | ColorMode::Multichannel);
        if self.mode == mode || (rgb_like(self.mode) && rgb_like(mode)) {
            return self;
        }
        let rgb = self.to_rgb();
        let (mode, c) = match mode {
            ColorMode::Rgb | ColorMode::Indexed | ColorMode::Multichannel => (ColorMode::Rgb, [rgb[0], rgb[1], rgb[2], 0.0]),
            ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => (mode, [convert::rgb_to_gray(rgb), 0.0, 0.0, 0.0]),
            ColorMode::Cmyk => (mode, convert::rgb_to_cmyk(rgb)),
            ColorMode::Lab => {
                let l = convert::srgb_to_lab(rgb);
                (mode, [l[0] / 100.0, (l[1] + 128.0) / 255.0, (l[2] + 128.0) / 255.0, 0.0])
            }
        };
        Color { mode, c, alpha: self.alpha }
    }

    pub fn to_rgba8(&self) -> [u8; 4] {
        let [r, g, b] = self.to_rgb();
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        [q(r), q(g), q(b), q(self.alpha)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_convert_into_a_document_mode() {
        let blue = Color::rgba(0.0, 0.0, 1.0, 0.5);
        let g = blue.in_mode(ColorMode::Grayscale);
        assert_eq!((g.mode, g.alpha), (ColorMode::Grayscale, 0.5));
        assert!((g.c[0] - 0.114).abs() < 1e-6, "{g:?}");
        // RGB-based modes keep RGB colours; a colour already in the mode is unchanged.
        assert_eq!(blue.in_mode(ColorMode::Indexed), blue);
        assert_eq!(g.in_mode(ColorMode::Grayscale), g);
        for mode in [ColorMode::Cmyk, ColorMode::Lab] {
            let c = blue.in_mode(mode);
            assert_eq!(c.mode, mode);
            let back = c.in_mode(ColorMode::Rgb);
            assert_eq!(back.mode, ColorMode::Rgb);
            assert_eq!(back.to_rgb(), c.to_rgb(), "{mode:?}: back to RGB shows the same colour");
        }
    }

    #[test]
    fn pixel_format_sizes() {
        assert_eq!(PixelFormat::RGBA8.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::RGBA16.bytes_per_pixel(), 8);
        assert_eq!(PixelFormat::RGBA32F.bytes_per_pixel(), 16);
        assert_eq!(PixelFormat::GRAY8.bytes_per_pixel(), 1);
        assert_eq!(PixelFormat::CMYKA8.channels(), 5);
    }

    #[test]
    fn sample_roundtrip_all_depths() {
        for s in SampleType::ALL {
            let mut buf = vec![0u8; 16];
            for (i, v) in [0.0f32, 0.25, 0.5, 1.0].iter().enumerate() {
                write_sample(&mut buf, s, i, *v);
                let r = read_sample(&buf, s, i);
                let tol = match s {
                    SampleType::U8 => 1.0 / 255.0,
                    SampleType::U16 => 1.0 / 65535.0,
                    SampleType::F32 => 0.0,
                };
                assert!((r - v).abs() <= tol, "{s:?} {v} -> {r}");
            }
        }
    }

    #[test]
    fn integer_samples_clamp() {
        let mut b = [0u8; 2];
        write_sample(&mut b, SampleType::U8, 0, 1.7);
        write_sample(&mut b, SampleType::U8, 1, -0.3);
        assert_eq!(b, [255, 0]);
    }

    #[test]
    fn float_samples_keep_hdr() {
        let mut b = [0u8; 4];
        write_sample(&mut b, SampleType::F32, 0, 4.5);
        assert_eq!(read_sample(&b, SampleType::F32, 0), 4.5);
    }

    #[test]
    fn color_to_rgba8() {
        assert_eq!(Color::WHITE.to_rgba8(), [255, 255, 255, 255]);
        assert_eq!(Color::gray(0.5).to_rgba8(), [128, 128, 128, 255]);
        let w = Color { mode: ColorMode::Cmyk, c: [0.0; 4], alpha: 1.0 };
        assert_eq!(w.to_rgba8(), [255, 255, 255, 255]);
        let k = Color { mode: ColorMode::Cmyk, c: [0.75, 0.68, 0.67, 0.9], alpha: 1.0 };
        assert!(k.to_rgba8()[..3].iter().all(|v| *v < 20), "{:?}", k.to_rgba8());
    }
}
